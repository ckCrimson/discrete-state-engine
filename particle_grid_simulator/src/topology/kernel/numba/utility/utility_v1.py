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
    """Calculates the exact reachable states at Step N. Returns list of handle indices."""
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
    """Calculates ALL states reachable up to Step N. Returns list of handle indices."""
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




class NumbaTopologyUtility(ITopologyUtility):
    """
    STATELESS UTILITY: Purely functional graph execution.
    Operates strictly on the FastRef payload and compiled function pointers.
    Zero OOP state is held in this class.
    """

    # ==========================================
    # INTERNAL: THE GRAPH BUILDER
    # ==========================================
    @staticmethod
    def _ensure_graph_built(
            fast_ref: Any,
            neighbour_func: Callable[[np.ndarray], np.ndarray],
            start_vector: np.ndarray,
            target_steps: int
    ) -> int:
        start_tuple = tuple(start_vector)

        if start_tuple not in fast_ref.visited_map:
            new_idx = np.int32(len(fast_ref.handle_map))
            fast_ref.visited_map[start_tuple] = new_idx
            fast_ref.handle_map.append(start_vector)

        start_idx = fast_ref.visited_map[start_tuple]
        if fast_ref.steps_prepared >= target_steps:
            return start_idx

        # O(1) Integer Queue
        current_frontier = List.empty_list(numba.types.int32)
        current_frontier.append(np.int32(start_idx))  # Wrapped in int32 for safety

        for step in range(0, target_steps):
            next_frontier = List.empty_list(numba.types.int32)

            for i in range(len(current_frontier)):
                current_idx = current_frontier[i]

                # FIX: If already expanded, queue its known neighbors and continue!
                if fast_ref.forward_counts[current_idx] > 0:
                    start_edge = fast_ref.forward_starts[current_idx]
                    count = fast_ref.forward_counts[current_idx]
                    for j in range(count):
                        n_idx = fast_ref.forward_edges[start_edge + j]
                        if fast_ref.forward_counts[n_idx] == 0:
                            next_frontier.append(np.int32(n_idx))
                    continue

                # DISCOVERY
                state_vec = fast_ref.handle_map[current_idx]
                neighbors = neighbour_func(state_vec)

                fast_ref.forward_starts[current_idx] = np.int32(len(fast_ref.forward_edges))
                fast_ref.forward_counts[current_idx] = np.int32(len(neighbors))

                for n_vec in neighbors:
                    n_tuple = tuple(n_vec)

                    if n_tuple not in fast_ref.visited_map:
                        n_idx = np.int32(len(fast_ref.handle_map))
                        fast_ref.visited_map[n_tuple] = n_idx
                        fast_ref.handle_map.append(n_vec)
                    else:
                        n_idx = fast_ref.visited_map[n_tuple]

                    fast_ref.forward_edges.append(np.int32(n_idx))
                    fast_ref.reverse_edges.append(np.int32(current_idx))

                    # Queue for next step if it hasn't been expanded
                    if fast_ref.forward_counts[n_idx] == 0:
                        next_frontier.append(np.int32(n_idx))

            # THE CHEAT: O(1) Integer Deduplication via Boolean Mask
            current_frontier.clear()
            if len(next_frontier) > 0:
                is_queued = np.zeros(len(fast_ref.handle_map), dtype=np.bool_)
                for i in range(len(next_frontier)):
                    n_idx = next_frontier[i]
                    if not is_queued[n_idx]:
                        is_queued[n_idx] = True
                        current_frontier.append(np.int32(n_idx))
            else:
                break

        fast_ref.steps_prepared = max(fast_ref.steps_prepared, target_steps)
        return start_idx

    # ==========================================
    # API IMPLEMENTATION
    # ==========================================
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

        # Execute pure C-level traversal
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
        """
        Two-Phase Engine Ignition:
        1. Absorbs JIT compilation tax (1-step dummy).
        2. Pre-calculates the physical adjacency graph (Deep build).
        """
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

    # ==========================================
    # API IMPLEMENTATION: BACKWARD TRAVERSAL (REACHING)
    # ==========================================

        # ==========================================
        # API IMPLEMENTATION: BACKWARD TRAVERSAL (REACHING)
        # ==========================================

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

            # EXACT SAME KERNEL -> DIFFERENT MEMORY POINTERS
            result_indices = _njit_ping_pong_frontier(
                start_idx, steps,
                fast_ref.reverse_starts, fast_ref.reverse_counts, fast_ref.reverse_edges,  # <-- REVERSE CSR
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

            # EXACT SAME KERNEL -> DIFFERENT MEMORY POINTERS
            result_indices = _njit_ping_pong_basin(
                start_idx, steps,
                fast_ref.reverse_starts, fast_ref.reverse_counts, fast_ref.reverse_edges,  # <-- REVERSE CSR
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
        raise NotImplementedError(
            "Directed backward traversal requires reverse CSR tracking (backward_starts, backward_edges) in the Storage layer."
        )