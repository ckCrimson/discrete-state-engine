import numpy as np
from numba import njit, prange
from typing import Callable

from particle_grid_simulator.src.state.domain.state_domain import StateSpace
from particle_grid_simulator.src.state.interfaces.storage import StateFastRefs
from particle_grid_simulator.src.state.interfaces.utility import IStateUtility


# ==========================================
# ⚡ THE HARDWARE KERNELS (Pure Math)
# ==========================================

@njit(cache=True)
def _jit_union(mask, ids, coords, new_ids, new_coords) -> int:
    total_new = len(new_ids)
    new_dim = new_coords.shape[1]
    actual_inserted_or_updated = 0

    for j in range(total_new):
        target_id = new_ids[j]
        already_exists = False

        # 1. Check if ID exists (Update/Upsert path)
        for i in range(len(mask)):
            if mask[i] == 1 and ids[i] == target_id:
                coords[i, :new_dim] = new_coords[j]
                already_exists = True
                actual_inserted_or_updated += 1
                break

        if not already_exists:
            # 2. Find empty slot (Insert path)
            for i in range(len(mask)):
                if mask[i] == 0:
                    ids[i] = target_id
                    coords[i, :new_dim] = new_coords[j]
                    mask[i] = 1
                    actual_inserted_or_updated += 1
                    break

    return actual_inserted_or_updated


@njit(cache=True, parallel=True)
def _jit_intersection(mask, ids, valid_ids):
    """
    Parallelized Intersection:
    Keeps an ID active ONLY if it exists in the valid_ids batch.
    """
    for i in prange(len(mask)):
        if mask[i] == 1:
            current_id = ids[i]
            found = False

            # Scan the (small) batch of valid_ids
            for j in range(len(valid_ids)):
                if current_id == valid_ids[j]:
                    found = True
                    break

            # If the ID in our vault is NOT in the incoming batch, kill it
            if not found:
                mask[i] = 0


class NumbaStateUtility(IStateUtility):
    def union_inplace(self, fast_refs: dict, **kwargs) -> None:
        data = kwargs.get('data')
        if not data: return

        # 1. Parse incoming data
        if isinstance(data, dict) and 'ids' in data:
            new_ids = data['ids'].astype(np.int64)
            new_coords = data['coords'].astype(np.float32)
        else:
            items = data.states if hasattr(data, 'states') else data
            vectors = np.array([d.vector if hasattr(d, 'vector') else d for d in items])
            if len(vectors) == 0: return
            new_ids = vectors[:, 0].astype(np.int64)
            new_coords = vectors[:, 1:].astype(np.float32)

        new_dim = new_coords.shape[1]

        # 2. Grab current active memory
        active_indices = np.where(fast_refs['active_mask'] == 1)[0]
        active_ids = fast_refs['ids'][active_indices]

        # 3. Vectorized Intersection (O(N log N) speed)
        # Finds overlap between active IDs and new IDs instantly
        _, active_match_idx, new_match_idx = np.intersect1d(
            active_ids, new_ids, return_indices=True, assume_unique=True
        )

        # 4. UPSERT: Update coordinates for IDs that already exist
        if len(active_match_idx) > 0:
            actual_memory_idx = active_indices[active_match_idx]
            fast_refs['coords'][actual_memory_idx, :new_dim] = new_coords[new_match_idx]

        # 5. INSERT: Find the IDs that are truly new
        insert_mask = np.ones(len(new_ids), dtype=np.bool_)
        insert_mask[new_match_idx] = False
        new_insert_ids = new_ids[insert_mask]
        new_insert_coords = new_coords[insert_mask]

        # 6. Bulk write new states to empty slots
        if len(new_insert_ids) > 0:
            empty_indices = np.where(fast_refs['active_mask'] == 0)[0]
            if len(empty_indices) < len(new_insert_ids):
                raise MemoryError("StateStorage capacity exceeded during union_inplace.")

            target_slots = empty_indices[:len(new_insert_ids)]
            fast_refs['ids'][target_slots] = new_insert_ids
            fast_refs['coords'][target_slots, :new_dim] = new_insert_coords
            fast_refs['active_mask'][target_slots] = 1

    def intersection_inplace(self, fast_refs: dict, **kwargs) -> None:
        data = kwargs.get('data')
        if not data: return

        # 1. Parse incoming valid_ids
        if isinstance(data, dict) and 'ids' in data:
            valid_ids = data['ids'].astype(np.int64)
        else:
            items = data.states if hasattr(data, 'states') else data
            vectors = np.array([d.vector if hasattr(d, 'vector') else d for d in items])
            if len(vectors) == 0:
                # Intersecting with an empty set means everything dies
                fast_refs['active_mask'][:] = 0
                return
            valid_ids = vectors[:, 0].astype(np.int64)

        if len(valid_ids) == 0:
            fast_refs['active_mask'][:] = 0
            return

        # 2. Grab current active memory slots
        active_indices = np.where(fast_refs['active_mask'] == 1)[0]
        if len(active_indices) == 0:
            return

        active_ids = fast_refs['ids'][active_indices]

        # 3. Vectorized Check (O(N log N) or better via C-backend)
        # np.isin checks which active_ids exist in valid_ids instantly
        keep_mask = np.isin(active_ids, valid_ids, assume_unique=True)

        # 4. Find the memory slots that failed the check
        drop_indices = active_indices[~keep_mask]

        # 5. Execute the wipe in bulk
        if len(drop_indices) > 0:
            fast_refs['active_mask'][drop_indices] = 0


    def map_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        transform_func: Callable = kwargs.get('transform_func')
        if not transform_func: return

        active_idx = np.where(fast_refs['active_mask'] == 1)[0]
        if len(active_idx) == 0: return

        # 1. Get the new projected coordinates
        new_coords = transform_func(
            fast_refs['ids'][active_idx],
            fast_refs['coords'][active_idx]
        )

        # 2. Extract the new dimensionality (e.g., 2)
        new_dim = new_coords.shape[1]

        # 3. Update the manifold by slicing the target to match the new_dim
        # This allows us to contract 3D -> 2D without reallocating the whole array.
        fast_refs['coords'][active_idx, :new_dim] = new_coords

    def filter_inplace(self, fast_refs: dict, predicate_func: Callable) -> None:
        """
        Applies a vectorized boolean filter directly to the active hardware memory.
        """
        # 1. Find the exact memory slots of currently active states
        active_indices = np.where(fast_refs['active_mask'] == 1)[0]
        if len(active_indices) == 0:
            return

        # 2. Extract ONLY the pure coordinates for the active states
        active_coords = fast_refs['coords'][active_indices]

        # 3. Apply the user's vectorized math (returns boolean array)
        keep_mask = predicate_func(active_coords)

        # 4. Find which specific memory slots failed the physics test
        drop_indices = active_indices[~keep_mask]

        # 5. Kill them directly in the hardware mask (O(1) per dropped state)
        if len(drop_indices) > 0:
            fast_refs['active_mask'][drop_indices] = 0