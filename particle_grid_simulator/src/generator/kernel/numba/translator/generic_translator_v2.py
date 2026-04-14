import numpy as np
import numba as nb
from typing import Any, List, Tuple

from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef
from particle_grid_simulator.src.generator.iterfaces.translator import IGeneratorTranslator

# ==========================================
# C-SPEED EXTRACTION HELPERS
# ==========================================
@nb.njit(cache=True)
def _fast_extract_edges(numba_list, out_array):
    """Compiles the 78,000 item loop into native C assembly."""
    for i in range(len(numba_list)):
        out_array[i] = numba_list[i]


class GenericGeneratorTranslator(IGeneratorTranslator):

    def bake(self, fast_refs: 'GeneratorKernelFastRef', initial_data: Tuple[np.ndarray, np.ndarray]) -> None:
        states, fields = initial_data
        count = len(states)
        fast_refs.buffer_A_states[:count] = states
        fast_refs.buffer_A_fields[:count] = fields
        fast_refs.active_count_A = count
        fast_refs.active_count_B = 0

    def bake_incremental(self, fast_refs: 'GeneratorKernelFastRef', queue: List[tuple], **kwargs: Any) -> None:
        pass

    def bake_topology_field(self, fast_refs: 'GeneratorKernelFastRef', topology_cm: Any, global_field_cm: Any) -> None:

        topo_refs = topology_cm.fast_refs
        raw_map = topo_refs.handle_map
        num_states = len(raw_map)

        # 1. Map Coordinates (Using list comprehension is acceptable here if raw_map is a Numba Dict,
        # but wrapping it in np.array directly handles contiguous memory allocation faster)
        coord_list = [raw_map[i] for i in range(num_states)]
        fast_refs.state_coordinates = np.array(coord_list, dtype=np.float64)

        # 2. THE CSR BRIDGE (64-Bit Memory Aligned)
        fast_refs.edge_offsets = np.ascontiguousarray(
            topo_refs.forward_starts[:num_states + 1],
            dtype=np.int64
        )

        edge_count = len(topo_refs.forward_edges)
        target_array = np.zeros(edge_count, dtype=np.int64)

        # THE FIX: Execute the 78,000 item copy via the compiled C-helper
        _fast_extract_edges(topo_refs.forward_edges, target_array)

        fast_refs.edge_targets = target_array

        # 3. Global Field Mapping
        g_refs = global_field_cm.fast_refs
        fast_refs.global_states = g_refs.state_array
        fast_refs.global_fields = g_refs.field_array
        fast_refs.global_norm_fields = g_refs.normalized_field_array

        # 4. Math Functions
        fast_refs.math_multiply = global_field_cm.multiply_vectors
        fast_refs.math_norm = global_field_cm.norm_vector

        print(f"   [Translator] Successfully bridged {num_states} states and {edge_count} edges.")

    def sync_to_domain(self, fast_refs: 'GeneratorKernelFastRef', active_buffer_flag: str) -> Tuple[np.ndarray, np.ndarray]:
        if active_buffer_flag == 'A':
            count = fast_refs.active_count_A
            return fast_refs.buffer_A_states[:count].copy(), fast_refs.buffer_A_fields[:count].copy()
        else:
            count = fast_refs.active_count_B
            return fast_refs.buffer_B_states[:count].copy(), fast_refs.buffer_B_fields[:count].copy()

    def sync(self, fast_refs: 'GeneratorKernelFastRef', **kwargs: Any) -> Any:
        active_flag = kwargs.get('active_buffer_flag', 'A')
        return self.sync_to_domain(fast_refs, active_flag)