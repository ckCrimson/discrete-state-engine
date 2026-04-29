import numpy as np
import numba as nb

from typing import Callable, Any, Optional, Dict

from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef
from particle_grid_simulator.src.generator.iterfaces.utility import IGeneratorKernelUtility

# ==========================================
# 1. THE SPATIAL HASH MAP BUILDER
# ==========================================
def _build_numba_dict(state_coords: np.ndarray):
    """
    Python-space wrapper: Allocates the typed dictionary safely outside the JIT compiler.
    Uses the `nb` alias directly to bypass IDE import confusion and namespace clashes.
    """
    key_ty = nb.types.Tuple((nb.types.float64, nb.types.float64, nb.types.float64))
    val_ty = nb.types.int32

    # Safe allocation in Python using the global nb alias
    coord_map = nb.typed.Dict.empty(key_type=key_ty, value_type=val_ty)

    # Pass pointer to C-level for blazing fast population
    _populate_dict(coord_map, state_coords)
    return coord_map


@nb.njit(fastmath=True)
def _populate_dict(coord_map, state_coords: np.ndarray):
    """C-space execution: Populates the pre-allocated map at native speed."""
    for i in range(len(state_coords)):
        key = (state_coords[i, 0], state_coords[i, 1], state_coords[i, 2])
        coord_map[key] = nb.int32(i)

# ==========================================
# THE JIT FACTORY (Zero-Overhead Compilation)
# ==========================================
_KERNEL_CACHE = {}
_MAP_CACHE = {}  # Caches the Spatial Hash Map so we only build it once

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
            do_explicit_norm: bool,
            coord_map: Any
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

        # O(1) Dictionary Lookup
        for i in nb.prange(current_active_count):
            key = (read_states[i, 0], read_states[i, 1], read_states[i, 2])
            if key in coord_map:
                read_ids[i] = coord_map[key]
            else:
                read_ids[i] = -1

        acc_fields = np.zeros((num_csr_nodes, field_dim), dtype=buf_A_fields.dtype)
        seen_nodes = np.zeros(num_csr_nodes, dtype=nb.boolean)

        # THE FIX: Track only the nodes we actually touch
        active_target_nodes = np.empty(capacity, dtype=np.int32)

        for step in range(steps):
            zero_val = read_fields[0, 0] * 0
            active_target_count = 0

            # Phase 1: Expansion & Accumulation
            for i in range(current_active_count):
                state_id = read_ids[i]
                if state_id == -1:
                    continue

                field_i = read_fields[i]
                start_edge = edge_offsets[state_id]
                end_edge = edge_offsets[state_id + 1]
                s_j = state_coords[state_id]

                total_local_mag = 0.0
                if do_implicit_norm:
                    for edge in range(start_edge, end_edge):
                        target_state_id = edge_targets[edge]
                        t_weight = transition_func(s_j, state_coords[target_state_id])
                        env_field = math_multiply(field_i, global_fields[target_state_id])
                        prop_field = math_multiply(env_field, t_weight)
                        for d in range(field_dim):
                            total_local_mag += np.sqrt(prop_field[d].real ** 2 + prop_field[d].imag ** 2)

                for edge in range(start_edge, end_edge):
                    target_state_id = edge_targets[edge]
                    t_weight = transition_func(s_j, state_coords[target_state_id])
                    env_field = math_multiply(field_i, global_fields[target_state_id])
                    prop_field = math_multiply(env_field, t_weight)

                    # THE FIX: Log the node if it's the first time we've seen it this step
                    if not seen_nodes[target_state_id]:
                        seen_nodes[target_state_id] = True
                        active_target_nodes[active_target_count] = target_state_id
                        active_target_count += 1

                    if do_implicit_norm and total_local_mag > 0:
                        for d in range(field_dim):
                            acc_fields[target_state_id, d] += prop_field[d] / total_local_mag
                    else:
                        for d in range(field_dim):
                            acc_fields[target_state_id, d] += prop_field[d]

            # THE FIX: Only calculate global norm on the ~8,000 active nodes
            if do_explicit_norm:
                global_mag = 0.0
                for k in nb.prange(active_target_count):
                    node_idx = active_target_nodes[k]
                    for d in range(field_dim):
                        global_mag += np.sqrt(acc_fields[node_idx, d].real ** 2 + acc_fields[node_idx, d].imag ** 2)

                if global_mag > 0:
                    for k in nb.prange(active_target_count):
                        node_idx = active_target_nodes[k]
                        for d in range(field_dim):
                            acc_fields[node_idx, d] = acc_fields[node_idx, d] / global_mag

            # THE FIX: Only compact and clean the ~8,000 active nodes
            for k in range(active_target_count):
                node_idx = active_target_nodes[k]

                # Write to the next buffer
                write_ids[k] = node_idx
                write_states[k] = state_coords[node_idx]
                for d in range(field_dim):
                    write_fields[k, d] = acc_fields[node_idx, d]

                # Instantly clean up for the next step so we don't need the 150k zeroing loop
                seen_nodes[node_idx] = False
                for d in range(field_dim):
                    acc_fields[node_idx, d] = zero_val

            current_active_count = active_target_count

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

        # 1. Retrieve or build the O(1) Numba Hash Map
        map_id = id(fast_refs.state_coordinates)
        if map_id not in _MAP_CACHE:
            _MAP_CACHE[map_id] = _build_numba_dict(fast_refs.state_coordinates)
        coord_map = _MAP_CACHE[map_id]

        # 2. Get Compiled Engine
        compiled_kernel = get_compiled_ping_pong_loop(
            math_multiply=fast_refs.math_multiply,
            math_norm=fast_refs.math_norm,
            transition_func=transition_func,
            field_dtype=fast_refs.buffer_A_fields.dtype
        )

        # 3. Execute
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
            do_explicit_norm=do_explicit_norm,
            coord_map=coord_map  # <-- Pass the map directly into the hardware loop
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