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


def get_compiled_ping_pong_loop(math_multiply, math_norm, transition_func, field_dtype):
    cache_key = (id(math_multiply), id(math_norm), id(transition_func), str(field_dtype))
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    @nb.njit(fastmath=True)
    def _compiled_loop(
            steps: int,
            buf_A_states: np.ndarray, buf_A_fields: np.ndarray, active_count_A: int,
            buf_B_states: np.ndarray, buf_B_fields: np.ndarray,
            state_coords: np.ndarray, edge_offsets: np.ndarray, edge_targets: np.ndarray,
            global_states: np.ndarray, global_fields: np.ndarray, global_norm_fields: np.ndarray,
            do_implicit_norm: bool,
            do_explicit_norm: bool  # FIX: Parameter restored
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

        acc_fields = np.zeros((num_csr_nodes, field_dim), dtype=buf_A_fields.dtype)
        seen_nodes = np.zeros(num_csr_nodes, dtype=nb.boolean)

        for step in range(steps):
            zero_val = read_fields[0, 0] * 0

            for i in range(num_csr_nodes):
                if seen_nodes[i]:
                    seen_nodes[i] = False
                    for d in range(field_dim):
                        acc_fields[i, d] = zero_val

            # Phase 1: Expansion & Accumulation
            for i in range(current_active_count):
                state_id = read_ids[i]
                if state_id == -1:
                    continue

                field_i = read_fields[i]
                start_edge = edge_offsets[state_id]
                end_edge = edge_offsets[state_id + 1]
                s_j = state_coords[state_id]

                # --- FIX: IMPLICIT NORM (Local Conservation) ---
                total_local_mag = 0.0
                if do_implicit_norm:
                    for edge in range(start_edge, end_edge):
                        target_state_id = edge_targets[edge]
                        t_weight = transition_func(s_j, state_coords[target_state_id])
                        env_field = math_multiply(field_i, global_fields[target_state_id])
                        prop_field = math_multiply(env_field, t_weight)

                        # Scalar magnitude calculation across vector dimensions
                        for d in range(field_dim):
                            total_local_mag += np.sqrt(prop_field[d].real ** 2 + prop_field[d].imag ** 2)

                # Execute propagation
                for edge in range(start_edge, end_edge):
                    target_state_id = edge_targets[edge]
                    t_weight = transition_func(s_j, state_coords[target_state_id])
                    env_field = math_multiply(field_i, global_fields[target_state_id])
                    prop_field = math_multiply(env_field, t_weight)

                    if do_implicit_norm and total_local_mag > 0:
                        for d in range(field_dim):
                            # Scalar division on vector index
                            acc_fields[target_state_id, d] += prop_field[d] / total_local_mag
                    else:
                        for d in range(field_dim):
                            acc_fields[target_state_id, d] += prop_field[d]

                    seen_nodes[target_state_id] = True

            # --- FIX: EXPLICIT NORM (Global Conservation) ---
            if do_explicit_norm:
                global_mag = 0.0
                for node_idx in range(num_csr_nodes):
                    if seen_nodes[node_idx]:
                        for d in range(field_dim):
                            global_mag += np.sqrt(acc_fields[node_idx, d].real ** 2 + acc_fields[node_idx, d].imag ** 2)

                if global_mag > 0:
                    for node_idx in range(num_csr_nodes):
                        if seen_nodes[node_idx]:
                            for d in range(field_dim):
                                # Scalar division on vector index
                                acc_fields[node_idx, d] = acc_fields[node_idx, d] / global_mag

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
            transition_func=transition_func,
            field_dtype=fast_refs.buffer_A_fields.dtype
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
            do_implicit_norm=do_implicit_norm,
            do_explicit_norm=do_explicit_norm  # FIX: Restored parameter pass
        )

        final_buffer_flag = 'A' if steps % 2 == 0 else 'B'

        if final_buffer_flag == 'A':
            fast_refs.active_count_A = final_count
            fast_refs.active_count_B = 0
        else:
            fast_refs.buffer_A_states, fast_refs.buffer_B_states = fast_refs.buffer_B_states, fast_refs.buffer_A_states
            fast_refs.buffer_A_fields, fast_refs.buffer_B_fields = fast_refs.buffer_B_fields, fast_refs.buffer_A_fields
            fast_refs.active_count_A = final_count
            fast_refs.active_count_B = 0

        return 'A'