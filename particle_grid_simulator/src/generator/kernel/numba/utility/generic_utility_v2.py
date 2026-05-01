import numpy as np
import numba as nb
from typing import Callable, Any, Optional

# ==========================================
# 1. GLOBAL CACHES
# ==========================================
_KERNEL_CACHE = {}
_MAP_CACHE = {}


# ==========================================
# 2. SPATIAL LOOKUP (Strict 2D)
# ==========================================
def _build_numba_dict(state_coords: np.ndarray):
    # Strictly 2D to satisfy the Numba compiler types
    key_ty = nb.types.Tuple((nb.types.float64, nb.types.float64))
    val_ty = nb.types.int32
    coord_map = nb.typed.Dict.empty(key_type=key_ty, value_type=val_ty)
    _populate_dict(coord_map, state_coords)
    return coord_map


@nb.njit(fastmath=True)
def _populate_dict(coord_map, state_coords: np.ndarray):
    for i in range(len(state_coords)):
        # Rounding handles floating point noise from the Operator jumps
        key = (np.round(state_coords[i, 0], 8), np.round(state_coords[i, 1], 8))
        coord_map[key] = nb.int32(i)


# ==========================================
# 3. THE JIT FACTORY (Fixed Signature)
# ==========================================
def get_compiled_ping_pong_loop(math_multiply, math_norm, transition_func, field_dtype):
    # math_norm is restored to the signature to prevent TypeErrors
    cache_key = (id(math_multiply), id(math_norm), id(transition_func), str(field_dtype))
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    @nb.njit(fastmath=True)
    def _compiled_loop(
            steps: int,
            buf_A_states: np.ndarray, buf_A_fields: np.ndarray, active_count_A: int,
            buf_B_states: np.ndarray, buf_B_fields: np.ndarray,
            state_coords: np.ndarray, edge_offsets: np.ndarray, edge_targets: np.ndarray,
            global_fields: np.ndarray,
            do_implicit_norm: bool,
            do_explicit_norm: bool,
            coord_map: Any
    ) -> int:
        current_active_count = active_count_A
        read_states, read_fields = buf_A_states, buf_A_fields
        write_states, write_fields = buf_B_states, buf_B_fields

        num_csr_nodes = len(state_coords)
        field_dim = read_fields.shape[1]

        acc_fields = np.zeros((num_csr_nodes, field_dim), dtype=field_dtype)
        seen_nodes = np.zeros(num_csr_nodes, dtype=nb.boolean)
        active_target_nodes = np.empty(num_csr_nodes, dtype=nb.int32)

        for step in range(steps):
            active_target_count = 0

            for i in range(current_active_count):
                # O(1) Lookup
                kx = np.round(read_states[i, 0], 8)
                ky = np.round(read_states[i, 1], 8)
                key = (kx, ky)

                if key not in coord_map:
                    continue

                state_id = coord_map[key]
                field_i = read_fields[i]
                s_j = state_coords[state_id]

                for edge in range(edge_offsets[state_id], edge_offsets[state_id + 1]):
                    target_id = edge_targets[edge]
                    t_weight = transition_func(s_j, state_coords[target_id])

                    # Core Physics: Current * Global * Transition
                    prop_field = math_multiply(math_multiply(field_i, global_fields[target_id]), t_weight)

                    if not seen_nodes[target_id]:
                        seen_nodes[target_id] = True
                        active_target_nodes[active_target_count] = target_id
                        active_target_count += 1

                    for d in range(field_dim):
                        acc_fields[target_id, d] += prop_field[d]

            # Write-back and Reset
            for k in range(active_target_count):
                node_idx = active_target_nodes[k]
                write_states[k] = state_coords[node_idx]
                for d in range(field_dim):
                    write_fields[k, d] = acc_fields[node_idx, d]

                seen_nodes[node_idx] = False
                for d in range(field_dim):
                    acc_fields[node_idx, d] = 0.0

            current_active_count = active_target_count
            read_states, write_states = write_states, read_states
            read_fields, write_fields = write_fields, read_fields

        return current_active_count

    _KERNEL_CACHE[cache_key] = _compiled_loop
    return _compiled_loop


# ==========================================
# 4. HARDWARE UTILITY WRAPPER
# ==========================================
class GenericGeneratorKernelUtility:
    @staticmethod
    def execute_multi_step(fast_refs, steps, transition_func, do_implicit_norm, do_explicit_norm, **kwargs):
        map_id = id(fast_refs.state_coordinates)
        if map_id not in _MAP_CACHE:
            _MAP_CACHE[map_id] = _build_numba_dict(fast_refs.state_coordinates)
        coord_map = _MAP_CACHE[map_id]

        # Signature now matches exactly what execute_multi_step provides
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
            global_fields=fast_refs.global_fields,
            do_implicit_norm=do_implicit_norm,
            do_explicit_norm=do_explicit_norm,
            coord_map=coord_map
        )

        if steps % 2 != 0:
            fast_refs.buffer_A_states, fast_refs.buffer_B_states = fast_refs.buffer_B_states, fast_refs.buffer_A_states
            fast_refs.buffer_A_fields, fast_refs.buffer_B_fields = fast_refs.buffer_B_fields, fast_refs.buffer_A_fields

        fast_refs.active_count_A = final_count
        fast_refs.active_count_B = 0
        return 'A'