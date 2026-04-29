# ==========================================
# 3. THE V2 TRANSLATOR (Pointer-less Routing)
# ==========================================
from ast import Tuple
from typing import Any, List

import numpy as np

from particle_grid_simulator.src.generator.iterfaces.translator import IGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.storage.dense_storage import DenseGeneratorFastRef


class DenseGeneratorTranslator(IGeneratorTranslator):
    def bake(self, fast_refs: DenseGeneratorFastRef, initial_data: Tuple[np.ndarray, np.ndarray]) -> None:
        states, fields = initial_data
        offset = fast_refs.grid_offset

        # Inject the sparse list directly into the dense 3D grid
        for i in range(len(states)):
            q = int(states[i, 0]) + offset
            r = int(states[i, 1]) + offset
            d_in = int(states[i, 2])
            fast_refs.grid_A[q, r, d_in] += fields[i, 0]

    def bake_incremental(self, fast_refs: Any, queue: List[tuple], **kwargs: Any) -> None:
        pass

    def bake_topology_field(self, fast_refs: DenseGeneratorFastRef, topology_cm: Any, global_field_cm: Any) -> None:
        # Reconstruct the Global Field as a Dense 2D array
        topo_map = topology_cm.fast_refs.handle_map
        g_fields = global_field_cm.fast_refs.field_array
        offset = fast_refs.grid_offset

        for i in range(len(topo_map)):
            q = int(topo_map[i][0]) + offset
            r = int(topo_map[i][1]) + offset
            fast_refs.global_field_grid[q, r] = g_fields[i, 0]

        print(f"   [Translator-V2] Dense Grid {fast_refs.grid_A.shape} mapped directly to L1 Cache.")

    def sync_to_domain(self, fast_refs: DenseGeneratorFastRef, active_buffer_flag: str) -> Tuple[
        np.ndarray, np.ndarray]:
        # Extract only the non-zero probabilities back into a sparse list for the Domain/Plotter
        grid = fast_refs.grid_A if fast_refs.active_buffer == 0 else fast_refs.grid_B
        offset = fast_refs.grid_offset

        # Find non-zero elements
        nz_q, nz_r, nz_d = np.nonzero(grid)
        count = len(nz_q)

        out_states = np.empty((count, 3), dtype=np.float64)
        out_fields = np.empty((count, 1), dtype=np.complex128)

        for i in range(count):
            out_states[i, 0] = float(nz_q[i] - offset)
            out_states[i, 1] = float(nz_r[i] - offset)
            out_states[i, 2] = float(nz_d[i])
            out_fields[i, 0] = grid[nz_q[i], nz_r[i], nz_d[i]]

        return out_states, out_fields

    def sync(self, fast_refs: Any, **kwargs: Any) -> Any:
        return self.sync_to_domain(fast_refs, "")
