import numpy as np
import numba
from numba import njit
from typing import Callable, Any

from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef
from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility
from particle_grid_simulator.src.generator.iterfaces.utility import IGeneratorKernelUtility
import math


# ==========================================
# JIT KERNELS (Pure C-Speed Execution)
# ==========================================

@njit(cache=True, fastmath=True)
def _find_vector_index(target_vec: np.ndarray, array_to_search: Any, current_count: int) -> int:
    """Used ONLY for initial 1-time bootstrapping, never in the hot loop."""
    dim = target_vec.shape[0]
    for i in range(current_count):
        match = True
        for d in range(dim):
            if array_to_search[i][d] != target_vec[d]:
                match = False
                break
        if match:
            return i
    return -1


_KERNEL_CACHE = {}


def get_single_step_kernel(transition_func: Callable):
    """
    FACTORY PATTERN: Wraps the Handle-Based DOD Kernel.
    Aggressively optimized for Zero-Allocation Scalar Math.
    """
    if id(transition_func) in _KERNEL_CACHE:
        return _KERNEL_CACHE[id(transition_func)]

    @njit(fastmath=True)
    def _generator_single_step_kernel_o1(
            r_active_indices: np.ndarray, r_fields: np.ndarray, r_count: int,
            w_active_indices: np.ndarray, w_states: np.ndarray, w_fields: np.ndarray,
            csr_coords: Any, csr_offsets: np.ndarray, csr_targets: Any,
            g_fields: np.ndarray, g_norm_fields: np.ndarray,
            do_implicit: bool, do_explicit: bool
    ) -> int:

        num_csr_nodes = len(csr_coords)
        field_dim = r_fields.shape[1]

        acc_fields = np.zeros((num_csr_nodes, field_dim), dtype=np.float64)
        seen_nodes = np.zeros(num_csr_nodes, dtype=np.bool_)

        # Pre-allocate a single flat scratchpad for the maximum possible neighbors
        # (Assuming max 128 neighbors for a single node)
        temp_frontier_fields = np.zeros((128, field_dim), dtype=np.float64)

        for i in range(r_count):
            node_j = r_active_indices[i]
            f_j = r_fields[i]
            s_j_vec = csr_coords[node_j]

            start_edge = csr_offsets[node_j]
            end_edge = csr_offsets[node_j + 1]
            num_neighbors = end_edge - start_edge

            if num_neighbors == 0:
                continue

            frontier_norm_sum = 0.0

            # -----------------------------------------------------
            # PHASE A: TRANSITION (Pure Scalar Math)
            # -----------------------------------------------------
            for n_idx in range(num_neighbors):
                target_node_idx = csr_targets[start_edge + n_idx]
                s_i_vec = csr_coords[target_node_idx]

                # Fetch global field
                f_g = g_norm_fields[target_node_idx] if do_explicit else g_fields[target_node_idx]

                # The only permitted allocation (from the domain physics rule)
                raw_t = transition_func(s_j_vec, s_i_vec)

                mag_sq = 0.0

                # 1. Scalar Multiplication & Magnitude Tracking
                for d in range(field_dim):
                    val = raw_t[d] * f_g[d]
                    temp_frontier_fields[n_idx, d] = val
                    mag_sq += val * val

                # 2. Implicit Normalization (Scalar)
                if do_implicit:
                    mag = math.sqrt(mag_sq)
                    if mag > 0:
                        mag_sq = 0.0  # Reset for explicit tracking
                        for d in range(field_dim):
                            val = temp_frontier_fields[n_idx, d] / mag
                            temp_frontier_fields[n_idx, d] = val
                            mag_sq += val * val  # Re-track magnitude if it changed
                    else:
                        mag_sq = 0.0
                        for d in range(field_dim):
                            temp_frontier_fields[n_idx, d] = 0.0

                # 3. Explicit Conservation Summing (Scalar)
                if do_explicit:
                    frontier_norm_sum += math.sqrt(mag_sq)

            # -----------------------------------------------------
            # PHASE B: CONSERVATION & ACCUMULATION (Pure Scalar Math)
            # -----------------------------------------------------

            # Pre-calculate the explicit norm multiplier to avoid division in the inner loop
            norm_multiplier = 1.0
            if do_explicit and frontier_norm_sum > 0:
                norm_multiplier = 1.0 / frontier_norm_sum

            for n_idx in range(num_neighbors):
                target_node_idx = csr_targets[start_edge + n_idx]

                # THE MICRO-OPTIMIZATION: Direct memory write, zero array creations!
                for d in range(field_dim):
                    f_s_d = temp_frontier_fields[n_idx, d] * norm_multiplier
                    acc_fields[target_node_idx, d] += f_j[d] * f_s_d

                seen_nodes[target_node_idx] = True

        # -----------------------------------------------------
        # PHASE C: DENSE COMPACTION
        # -----------------------------------------------------
        w_count = 0
        for node_idx in range(num_csr_nodes):
            if seen_nodes[node_idx]:
                w_active_indices[w_count] = node_idx
                s_vec = csr_coords[node_idx]
                for d in range(len(s_vec)):
                    w_states[w_count, d] = s_vec[d]
                for d in range(field_dim):
                    w_fields[w_count, d] = acc_fields[node_idx, d]
                w_count += 1

        # Clean remainder of the dense write buffer
        for idx in range(w_count, w_states.shape[0]):
            for d in range(w_states.shape[1]):
                w_states[idx, d] = 0.0
            for d in range(field_dim):
                w_fields[idx, d] = 0.0

        return w_count

    _KERNEL_CACHE[id(transition_func)] = _generator_single_step_kernel_o1
    return _generator_single_step_kernel_o1


# ==========================================
# HARDWARE UTILITY IMPLEMENTATION
# ==========================================

class NumbaGeneratorUtility(IGeneratorKernelUtility):

    @staticmethod
    def execute_multi_step(
            fast_refs: 'GeneratorKernelFastRef',
            steps: int,
            transition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            math_utility: Any,
            intrinsic_norm: bool,  # <-- Reverted to match your Component Manager
            extrinsic_norm: bool,  # <-- Reverted to match your Component Manager
            **kwargs: Any
    ) -> str:

        compiled_kernel = get_single_step_kernel(transition_func)

        # 1. Allocate Handle tracking buffers
        max_cap = fast_refs.buffer_A_states.shape[0]
        active_indices_A = np.zeros(max_cap, dtype=np.int32)
        active_indices_B = np.zeros(max_cap, dtype=np.int32)

        # 2. Bootstrap Initial Handles
        num_csr_nodes = len(fast_refs.state_coordinates)
        for i in range(fast_refs.active_count_A):
            idx = _find_vector_index(
                fast_refs.buffer_A_states[i],
                fast_refs.state_coordinates,
                num_csr_nodes
            )
            if idx == -1:
                raise ValueError("Initial Generator State not found in Topology Graph.")
            active_indices_A[i] = idx

        active_buffer = 'A'

        # 3. Hardware Ping-Pong Sequence
        for l in range(steps):
            if active_buffer == 'A':
                w_count = compiled_kernel(
                    active_indices_A, fast_refs.buffer_A_fields, fast_refs.active_count_A,
                    active_indices_B, fast_refs.buffer_B_states, fast_refs.buffer_B_fields,
                    fast_refs.state_coordinates, fast_refs.edge_offsets, fast_refs.edge_targets,
                    fast_refs.global_fields, fast_refs.global_normalized_fields,
                    intrinsic_norm, extrinsic_norm  # Passed to the kernel here
                )
                fast_refs.active_count_B = w_count
                active_buffer = 'B'
            else:
                w_count = compiled_kernel(
                    active_indices_B, fast_refs.buffer_B_fields, fast_refs.active_count_B,
                    active_indices_A, fast_refs.buffer_A_states, fast_refs.buffer_A_fields,
                    fast_refs.state_coordinates, fast_refs.edge_offsets, fast_refs.edge_targets,
                    fast_refs.global_fields, fast_refs.global_normalized_fields,
                    intrinsic_norm, extrinsic_norm  # Passed to the kernel here
                )
                fast_refs.active_count_A = w_count
                active_buffer = 'A'

            if w_count >= max_cap:
                raise MemoryError("Generator max_size exceeded hardware allocation limit.")

        return active_buffer