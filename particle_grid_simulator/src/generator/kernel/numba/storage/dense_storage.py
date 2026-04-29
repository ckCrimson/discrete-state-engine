import numpy as np
import numba as nb
from typing import Any, Tuple, List
from dataclasses import dataclass

from particle_grid_simulator.src.generator.iterfaces.translator import IGeneratorTranslator
from particle_grid_simulator.src.generator.iterfaces.storage import IGeneratorKernelStorage


# ==========================================
# 1. THE V2 FAST REF (Dense 3D Grid)
# ==========================================
@nb.experimental.jitclass([
    ('grid_A', nb.complex128[:, :, :]),
    ('grid_B', nb.complex128[:, :, :]),
    ('global_field_grid', nb.complex128[:, :]),
    ('deltas', nb.float64[:, :]),
    ('grid_offset', nb.int64),
    ('active_buffer', nb.int64)  # 0 for A, 1 for B
])
class DenseGeneratorFastRef:
    def __init__(self, grid_size: int, num_directions: int, grid_offset: int):
        self.grid_A = np.zeros((grid_size, grid_size, num_directions), dtype=np.complex128)
        self.grid_B = np.zeros((grid_size, grid_size, num_directions), dtype=np.complex128)
        self.global_field_grid = np.zeros((grid_size, grid_size), dtype=np.complex128)
        self.deltas = np.zeros((num_directions, 2), dtype=np.float64)
        self.grid_offset = grid_offset
        self.active_buffer = 0


# ==========================================
# 2. THE V2 STORAGE (L1 Cache Optimized)
# ==========================================
class NumbaDenseGeneratorStorage(IGeneratorKernelStorage):
    def __init__(self, box_radius: int, deltas: np.ndarray):
        self.box_radius = box_radius
        self.grid_size = (2 * box_radius) + 1
        self.num_directions = len(deltas)

        # Offset shifts coordinates like (-5, 5) -> (array_index, array_index)
        self.fast_ref_obj = DenseGeneratorFastRef(self.grid_size, self.num_directions, self.box_radius)
        self.fast_ref_obj.deltas = deltas

    @property
    def fast_refs(self) -> DenseGeneratorFastRef:
        return self.fast_ref_obj

    def get_buffers(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fast_ref_obj.grid_A, self.fast_ref_obj.grid_B

    def clear(self) -> None:
        self.fast_ref_obj.grid_A.fill(0.0 + 0.0j)
        self.fast_ref_obj.grid_B.fill(0.0 + 0.0j)
        self.fast_ref_obj.active_buffer = 0

# ==========================================
    # INTERFACE COMPLIANCE (Dummy/Mapped Buffers)
    # ==========================================
    @property
    def buffer_A_states(self) -> np.ndarray:
        # V2 encodes state natively into the grid indices.
        # Returning an empty array to satisfy the strict interface contract.
        return np.empty((0, 3), dtype=np.float64)

    @property
    def buffer_A_fields(self) -> np.ndarray:
        # Return the actual grid memory block
        return self.fast_ref_obj.grid_A

    @property
    def buffer_B_states(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.float64)

    @property
    def buffer_B_fields(self) -> np.ndarray:
        return self.fast_ref_obj.grid_B



