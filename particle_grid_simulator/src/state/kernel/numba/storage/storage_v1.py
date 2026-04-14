import numpy as np
from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract
from particle_grid_simulator.src.state.interfaces.storage import IStateStorage, StateFastRefs


class NumbaStateStorage(IStateStorage):
    def __init__(self, contract: KernelDataContract) -> None:
        max_count = contract.config.get('max_count', 1000000)
        dims = contract.config.get('dimensions', 3)

        self._ids = np.empty(max_count, dtype=np.int64)
        self._active_mask = np.zeros(max_count, dtype=np.uint8)
        # One contiguous block for all spatial dimensions
        self._coords = np.zeros((max_count, dims), dtype=np.float32)

        self._fast_refs = {
            'ids': self._ids,
            'active_mask': self._active_mask,
            'coords': self._coords
        }

    @property
    def fast_refs(self): return self._fast_refs

    @property
    def count(self) -> int:
        # Numba Storage knows it uses NumPy, so it handles the np.sum
        return int(np.sum(self._active_mask))

    def clear(self) -> None:
        # Numba Storage knows it can do in-place NumPy mutation
        self._active_mask[:] = 0

    def get_valid_state_vectors(self) -> np.ndarray:
        """
        Extracts active states and returns them as a stacked 2D NumPy array.
        """
        # 1. Create a boolean mask of active slots
        active_idx = self._active_mask == 1

        # 2. Extract and reshape IDs to be a column vector (N, 1)
        valid_ids = self._ids[active_idx].astype(np.float32).reshape(-1, 1)

        # 3. Extract valid coordinates (N, dims)
        valid_coords = self._coords[active_idx]

        # 4. Bind them together into a single [ID, x, y, z] matrix
        return np.hstack((valid_ids, valid_coords))