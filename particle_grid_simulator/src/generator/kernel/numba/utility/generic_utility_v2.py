import numpy as np
import numba as nb
from typing import Callable, Any, Optional

from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef
from particle_grid_simulator.src.generator.iterfaces.utility import IGeneratorKernelUtility


@nb.njit(cache=True, fastmath=True)
def _find_coord_id(target_vec: np.ndarray, array_to_search: np.ndarray) -> int:
    dim = target_vec.shape[0]
    for i in range(len(array_to_search)):
        match = True
        for d in range(dim):
            if array_to_search[i][d] != target_vec[d]:
                match = False
                break
        if match:
            return i
    return -1


# ==========================================
# THE JIT FACTORY (Zero-Overhead Compilation)
# ==========================================
_KERNEL_CACHE = {}


def get_compiled_ping_pong_loop(math_multiply, math_norm, transition_func):
    """
    FACTORY PATTERN: Bakes the specific math functions directly into the compiled assembly.
    This eliminates 'Function Pointer' dynamic dispatch overhead completely.
    """
    cache_key = (id(math_multiply), id(math_norm), id(transition_func))
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    @nb.njit(fastmath=True)
    def _compiled_loop(
            steps: int,
            buf_A_states: np.ndarray, buf_A_fields: np.ndarray, active_count_A: int,
            buf_B_states: np.ndarray, buf_B_fields: np.ndarray,
            state_coords: np.ndarray, edge_offsets: np.ndarray, edge_targets: np.ndarray,
            global_states: np.ndarray, global_fields: np.ndarray, global_norm_fields: np.ndarray,
            do_implicit_norm: bool
    ) -> int:
        current_active_count = active_count_A
        read_states = buf_A_states
        read_fields = buf_A_fields
        write_states = buf_B_states
        write_fields = buf_B_fields

        capacity = len(buf_A_states)
        num_csr_nodes = len(state_coords)
        field_dim = read_fields.shape[1]

        read_ids = np.empty(capacity, dtype=np.int32)
        write_ids = np.empty(capacity, dtype=np.int32)

        for i in range(current_active_count):
            read_ids[i] = _find_coord_id(read_states[i], state_coords)

        # ---------------------------------------------------------
        # PRE-ALLOCATION: Allocate RAM exactly ONCE before the loop.
        # ---------------------------------------------------------
        acc_fields = np.zeros((num_csr_nodes, field_dim), dtype=read_fields.dtype)
        seen_nodes = np.zeros(num_csr_nodes, dtype=np.bool_)

        for step in range(steps):

            # FAST SPARSE CLEAR: Only zero out the memory we touched in the last step!
            for i in range(num_csr_nodes):
                if seen_nodes[i]:
                    seen_nodes[i] = False
                    for d in range(field_dim):
                        acc_fields[i, d] = 0.0

            for i in range(current_active_count):
                state_id = read_ids[i]
                if state_id == -1:
                    continue

                field_i = read_fields[i]
                start_edge = edge_offsets[state_id]
                end_edge = edge_offsets[state_id + 1]

                for edge in range(start_edge, end_edge):
                    target_state_id = edge_targets[edge]
                    target_global_field = global_fields[target_state_id]

                    s_j = state_coords[state_id]
                    s_i = state_coords[target_state_id]

                    # Numba successfully inlines these because of the Factory pattern
                    t_weight = transition_func(s_j, s_i)
                    env_field = math_multiply(field_i, target_global_field)
                    propagated_field = math_multiply(env_field, t_weight)

                    if do_implicit_norm:
                        mag_sq = 0.0
                        for d in range(field_dim):
                            mag_sq += propagated_field[d].real ** 2 + propagated_field[d].imag ** 2
                        n_val = np.sqrt(mag_sq)

                        if n_val > 0:
                            for d in range(field_dim):
                                acc_fields[target_state_id, d] += propagated_field[d] / n_val
                        else:
                            for d in range(field_dim):
                                acc_fields[target_state_id, d] += propagated_field[d]
                    else:
                        for d in range(field_dim):
                            acc_fields[target_state_id, d] += propagated_field[d]

                    seen_nodes[target_state_id] = True

            # Dense Compaction
            write_idx = 0
            for node_idx in range(num_csr_nodes):
                if seen_nodes[node_idx]:
                    write_ids[write_idx] = node_idx
                    write_states[write_idx] = state_coords[node_idx]
                    for d in range(field_dim):
                        write_fields[write_idx, d] = acc_fields[node_idx, d]
                    write_idx += 1

            current_active_count = write_idx

            # Pointer Swap
            temp_states, temp_fields, temp_ids = read_states, read_fields, read_ids
            read_states, read_fields, read_ids = write_states, write_fields, write_ids
            write_states, write_fields, write_ids = temp_states, temp_fields, temp_ids

        return current_active_count

    _KERNEL_CACHE[cache_key] = _compiled_loop
    return _compiled_loop


# ==========================================
# HARDWARE UTILITY WRAPPER
# ==========================================
class GenericGeneratorKernelUtility(IGeneratorKernelUtility):

    @staticmethod
    def execute_multi_step(
            fast_refs: 'GeneratorKernelFastRef',
            steps: int,
            transition_func: Callable,
            do_implicit_norm: bool,
            do_explicit_norm: bool,
            math_utility: Optional[Any] = None,
            **kwargs: Any
    ) -> str:

        compiled_kernel = get_compiled_ping_pong_loop(
            math_multiply=fast_refs.math_multiply,
            math_norm=fast_refs.math_norm,
            transition_func=transition_func
        )

        final_count = compiled_kernel(
            steps=steps,
            buf_A_states=fast_refs.buffer_A_states,
            buf_A_fields=fast_refs.buffer_A_fields,
            active_count_A=fast_refs.active_count_A,
            buf_B_states=fast_refs.buffer_B_states,
            buf_B_fields=fast_refs.buffer_B_fields,
            state_coords=fast_refs.state_coordinates,
            edge_offsets=fast_refs.edge_offsets,
            edge_targets=fast_refs.edge_targets,
            global_states=fast_refs.global_states,
            global_fields=fast_refs.global_fields,
            global_norm_fields=fast_refs.global_normalized_fields,
            do_implicit_norm=do_implicit_norm
        )

        final_buffer_flag = 'A' if steps % 2 == 0 else 'B'

        # =========================================================
        # THE FIX: GUARANTEE BUFFER 'A' IS ALWAYS READY FOR THE NEXT CALL
        # =========================================================
        if final_buffer_flag == 'A':
            fast_refs.active_count_A = final_count
            fast_refs.active_count_B = 0
        else:
            # Swap the underlying numpy array pointers directly in memory
            fast_refs.buffer_A_states, fast_refs.buffer_B_states = fast_refs.buffer_B_states, fast_refs.buffer_A_states
            fast_refs.buffer_A_fields, fast_refs.buffer_B_fields = fast_refs.buffer_B_fields, fast_refs.buffer_A_fields

            fast_refs.active_count_A = final_count
            fast_refs.active_count_B = 0

        # We always return 'A' because we forced the winning data into Buffer A
        return 'A'