import numpy as np
import numba
from numba import njit
from numba.typed import List
from typing import Iterable, Callable, Any

from particle_grid_simulator.src.topology.interfaces.utility import ITopologyUtility


# ==========================================
# FAST PATH: MULTI-STEP FRONTIER KERNEL
# ==========================================
@njit(fastmath=True)
def _njit_ping_pong_frontier(
        start_idx: int,
        target_steps: int,
        starts: np.ndarray,
        counts: np.ndarray,
        edges: List,
        buffer_a: List,
        buffer_b: List,
        seen_array: np.ndarray
):
    buffer_a.clear()
    buffer_a.append(np.int32(start_idx))

    for step in range(target_steps):
        buffer_b.clear()
        seen_array[:] = False

        for idx in buffer_a:
            start_pos = starts[idx]
            count = counts[idx]

            for i in range(count):
                neighbor = edges[start_pos + i]

                if not seen_array[neighbor]:
                    seen_array[neighbor] = True
                    buffer_b.append(neighbor)

        buffer_a.clear()
        for item in buffer_b:
            buffer_a.append(item)

    return buffer_a


# ==========================================
# FAST PATH: MULTI-STEP BASIN KERNEL
# ==========================================
@njit(fastmath=True)
def _njit_ping_pong_basin(
        start_idx: int,
        target_steps: int,
        starts: np.ndarray,
        counts: np.ndarray,
        edges: List,
        buffer_a: List,
        buffer_b: List,
        seen_array: np.ndarray
):
    buffer_a.clear()
    buffer_a.append(np.int32(start_idx))

    basin_out = List.empty_list(numba.types.int32)
    basin_out.append(start_idx)

    seen_array[:] = False
    seen_array[start_idx] = True

    for step in range(target_steps):
        buffer_b.clear()

        for idx in buffer_a:
            start_pos = starts[idx]
            count = counts[idx]

            for i in range(count):
                neighbor = edges[start_pos + i]

                if not seen_array[neighbor]:
                    seen_array[neighbor] = True
                    buffer_b.append(neighbor)
                    basin_out.append(neighbor)

        buffer_a.clear()
        for item in buffer_b:
            buffer_a.append(item)

    return basin_out


# ==========================================
# FACTORY COMPILER (THE BULLETPROOF FIX)
# ==========================================
_KERNEL_CACHE = {}


def _get_or_compile_core_kernel(dim: int):
    """
    Dynamically compiles the exact kernel needed for 1D, 2D, or 3D
    without relying on any deprecated Numba extensions.
    """
    if dim in _KERNEL_CACHE:
        return _KERNEL_CACHE[dim]

    @njit(fastmath=True)
    def _dynamic_graph_core(
            visited_map, handle_map,
            forward_starts, forward_counts, forward_edges, reverse_edges,
            neighbour_func, start_vector, target_steps, steps_prepared, seen_array
    ):
        seen_array[:] = False

        # --- DYNAMIC TUPLE EXTRACTION ---
        # Numba performs dead-code elimination here at compile time!
        if dim == 1:
            s_tuple = (start_vector[0],)
        elif dim == 2:
            s_tuple = (start_vector[0], start_vector[1])
        elif dim == 3:
            s_tuple = (start_vector[0], start_vector[1], start_vector[2])
        else:
            s_tuple = (start_vector[0],)

        if s_tuple not in visited_map:
            new_idx = np.int32(len(handle_map))
            visited_map[s_tuple] = new_idx
            handle_map.append(start_vector)

        start_idx = visited_map[s_tuple]

        if steps_prepared >= target_steps:
            return start_idx, steps_prepared

        current_frontier = List.empty_list(numba.types.int32)
        current_frontier.append(start_idx)

        for step in range(0, target_steps):
            next_frontier = List.empty_list(numba.types.int32)

            for i in range(len(current_frontier)):
                current_idx = current_frontier[i]

                if forward_counts[current_idx] > 0:
                    start_edge = forward_starts[current_idx]
                    count = forward_counts[current_idx]
                    for j in range(count):
                        n_idx = forward_edges[start_edge + j]
                        if forward_counts[n_idx] == 0:
                            next_frontier.append(n_idx)
                    continue

                state_vec = handle_map[current_idx]
                neighbors = neighbour_func(state_vec)

                forward_starts[current_idx] = np.int32(len(forward_edges))
                forward_counts[current_idx] = np.int32(len(neighbors))

                for n_idx_iter in range(len(neighbors)):
                    n_vec = neighbors[n_idx_iter]

                    # --- DYNAMIC TUPLE EXTRACTION FOR NEIGHBORS ---
                    if dim == 1:
                        n_tuple = (n_vec[0],)
                    elif dim == 2:
                        n_tuple = (n_vec[0], n_vec[1])
                    elif dim == 3:
                        n_tuple = (n_vec[0], n_vec[1], n_vec[2])
                    else:
                        n_tuple = (n_vec[0],)

                    if n_tuple not in visited_map:
                        n_idx_new = np.int32(len(handle_map))
                        visited_map[n_tuple] = n_idx_new
                        handle_map.append(n_vec)
                    else:
                        n_idx_new = visited_map[n_tuple]

                    forward_edges.append(n_idx_new)
                    reverse_edges.append(np.int32(current_idx))

                    if forward_counts[n_idx_new] == 0:
                        next_frontier.append(n_idx_new)

            current_frontier.clear()

            if len(next_frontier) > 0:
                for i in range(len(next_frontier)):
                    n_idx = next_frontier[i]
                    if not seen_array[n_idx]:
                        seen_array[n_idx] = True
                        current_frontier.append(n_idx)

                for i in range(len(current_frontier)):
                    seen_array[current_frontier[i]] = False
            else:
                break

        new_steps = max(steps_prepared, target_steps)
        return start_idx, new_steps

    _KERNEL_CACHE[dim] = _dynamic_graph_core
    return _dynamic_graph_core


class NumbaTopologyUtility(ITopologyUtility):
    @staticmethod
    def _ensure_graph_built(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            start_vector: np.ndarray,
            target_steps: int
    ) -> int:
        # FIX: We dynamically check the dimension in pure Python and
        # fetch the perfectly compiled Numba kernel for that specific dimension.
        dim = len(start_vector)
        core_kernel = _get_or_compile_core_kernel(dim)

        start_idx, new_steps_prepared = core_kernel(
            fast_ref.visited_map,
            fast_ref.handle_map,
            fast_ref.forward_starts,
            fast_ref.forward_counts,
            fast_ref.forward_edges,
            fast_ref.reverse_edges,
            neighbour_func,
            start_vector,
            target_steps,
            fast_ref.steps_prepared,
            fast_ref.seen_array
        )

        fast_ref.steps_prepared = new_steps_prepared
        return start_idx

    @staticmethod
    def get_reachable(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray
    ) -> Iterable[np.ndarray]:
        return NumbaTopologyUtility.get_reachable_multi_step_frontier(
            fast_ref, neighbour_func, state_vector_in, 1
        )

    @staticmethod
    def get_reachable_multi_step_frontier(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray,
            steps: int
    ) -> Iterable[np.ndarray]:
        start_idx = NumbaTopologyUtility._ensure_graph_built(
            fast_ref, neighbour_func, state_vector_in, steps
        )

        result_indices = _njit_ping_pong_frontier(
            start_idx, steps,
            fast_ref.forward_starts, fast_ref.forward_counts, fast_ref.forward_edges,
            fast_ref.buffer_a, fast_ref.buffer_b, fast_ref.seen_array
        )

        return [fast_ref.handle_map[i] for i in result_indices]

    @staticmethod
    def get_reachable_multi_step_basin(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray,
            steps: int
    ) -> Iterable[np.ndarray]:
        start_idx = NumbaTopologyUtility._ensure_graph_built(
            fast_ref, neighbour_func, state_vector_in, steps
        )

        result_indices = _njit_ping_pong_basin(
            start_idx, steps,
            fast_ref.forward_starts, fast_ref.forward_counts, fast_ref.forward_edges,
            fast_ref.buffer_a, fast_ref.buffer_b, fast_ref.seen_array
        )

        return [fast_ref.handle_map[i] for i in result_indices]

    @staticmethod
    def warmup(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray,
            steps: int
    ) -> None:
        dummy_idx = NumbaTopologyUtility._ensure_graph_built(
            fast_ref, neighbour_func, state_vector_in, 1
        )
        _njit_ping_pong_frontier(
            dummy_idx, 1,
            fast_ref.forward_starts, fast_ref.forward_counts, fast_ref.forward_edges,
            fast_ref.buffer_a, fast_ref.buffer_b, fast_ref.seen_array
        )

        if steps > 1:
            NumbaTopologyUtility._ensure_graph_built(
                fast_ref, neighbour_func, state_vector_in, steps
            )

    @staticmethod
    def get_reaching(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray
    ) -> Iterable[np.ndarray]:
        return NumbaTopologyUtility.get_reaching_multi_step_frontier(
            fast_ref, neighbour_func, state_vector_in, 1
        )

    @staticmethod
    def get_reaching_multi_step_frontier(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray,
            steps: int
    ) -> Iterable[np.ndarray]:
        start_idx = NumbaTopologyUtility._ensure_graph_built(
            fast_ref, neighbour_func, state_vector_in, steps
        )

        result_indices = _njit_ping_pong_frontier(
            start_idx, steps,
            fast_ref.reverse_starts, fast_ref.reverse_counts, fast_ref.reverse_edges,
            fast_ref.buffer_a, fast_ref.buffer_b, fast_ref.seen_array
        )

        return [fast_ref.handle_map[i] for i in result_indices]

    @staticmethod
    def get_reaching_multi_step_basin(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            state_vector_in: np.ndarray,
            steps: int
    ) -> Iterable[np.ndarray]:
        start_idx = NumbaTopologyUtility._ensure_graph_built(
            fast_ref, neighbour_func, state_vector_in, steps
        )

        result_indices = _njit_ping_pong_basin(
            start_idx, steps,
            fast_ref.reverse_starts, fast_ref.reverse_counts, fast_ref.reverse_edges,
            fast_ref.buffer_a, fast_ref.buffer_b, fast_ref.seen_array
        )

        return [fast_ref.handle_map[i] for i in result_indices]