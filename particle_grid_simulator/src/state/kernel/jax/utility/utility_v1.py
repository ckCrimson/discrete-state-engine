from typing import Callable
import jax
import jax.numpy as jnp
from functools import partial
from particle_grid_simulator.src.state.interfaces.storage import StateFastRefs
from particle_grid_simulator.src.state.interfaces.utility import IStateUtility


class JaxStateUtility(IStateUtility):

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _apply_filter(mask, coords, predicate_func):
        # Run math on GPU, then update mask
        keep = predicate_func(coords)
        return jnp.where(keep, mask, 0)

    def filter_inplace(self, fast_refs: StateFastRefs, predicate_func: Callable, **kwargs) -> None:
        # Re-assigning the GPU buffer reference
        fast_refs['active_mask'] = self._apply_filter(
            fast_refs['active_mask'],
            fast_refs['coords'],
            predicate_func
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))  # transform_func is now the 4th argument (index 3)
    def _apply_map(mask, ids, coords, transform_func):
        # Pass BOTH ids and coords to match the Numba contract!
        new_coords = transform_func(ids, coords)
        return jnp.where(mask[:, None] == 1, new_coords, coords)

    def map_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        transform_func: Callable = kwargs.get('transform_func')
        fast_refs['coords'] = self._apply_map(
            fast_refs['active_mask'],
            fast_refs['ids'],  # <--- Now passing IDs
            fast_refs['coords'],
            transform_func
        )

    # ---------------------------------------------------------
    # NEW IMPLEMENTATIONS: INTERSECTION & UNION
    # ---------------------------------------------------------

    @staticmethod
    @jax.jit
    def _apply_intersection(mask, ids, valid_ids):
        # jnp.isin is fully supported by JAX for static array shapes
        # It instantly checks which of our vault IDs exist in the valid_ids batch
        keep = jnp.isin(ids, valid_ids)
        return jnp.where(keep, mask, 0)

    def intersection_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        data = kwargs.get('data')
        if not data: return

        # Parse incoming valid_ids
        if isinstance(data, dict) and 'ids' in data:
            valid_ids = jnp.array(data['ids'], dtype=jnp.int32)
        else:
            items = data.states if hasattr(data, 'states') else data
            vectors = jnp.array([d.vector if hasattr(d, 'vector') else d for d in items])
            if len(vectors) == 0:
                fast_refs['active_mask'] = jnp.zeros_like(fast_refs['active_mask'])
                return
            valid_ids = vectors[:, 0].astype(jnp.int32)

        if len(valid_ids) == 0:
            fast_refs['active_mask'] = jnp.zeros_like(fast_refs['active_mask'])
            return

        fast_refs['active_mask'] = self._apply_intersection(
            fast_refs['active_mask'],
            fast_refs['ids'],
            valid_ids
        )

    @staticmethod
    @jax.jit
    def _apply_union(mask, ids, coords, new_ids, new_coords):
        """
        Vectorized Insert using JAX cumulative sums.
        Finds empty slots (mask == 0) and cleanly maps the new data into them.
        """
        batch_size = new_ids.shape[0]

        # 1. Create a boolean array of all currently empty slots
        empty_flags = (mask == 0)

        # 2. Assign a unique, ascending index (0, 1, 2, 3...) to every empty slot
        empty_indices = jnp.cumsum(empty_flags) - 1

        # 3. We only want to fill exactly 'batch_size' number of slots
        valid_targets = empty_flags & (empty_indices < batch_size)

        # 4. Map the new data into the vault.
        # (JAX safely clamps out-of-bounds indices, making this operation exceptionally fast)
        updated_ids = jnp.where(valid_targets, new_ids[empty_indices], ids)
        updated_coords = jnp.where(valid_targets[:, None], new_coords[empty_indices], coords)
        updated_mask = jnp.where(valid_targets, 1, mask)

        return updated_mask, updated_ids, updated_coords

    def union_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        data = kwargs.get('data')
        if not data: return

        # Parse incoming data
        if isinstance(data, dict) and 'ids' in data:
            new_ids = jnp.array(data['ids'], dtype=jnp.int32)
            new_coords = jnp.array(data['coords'], dtype=jnp.float32)
        else:
            items = data.states if hasattr(data, 'states') else data
            vectors = jnp.array([d.vector if hasattr(d, 'vector') else d for d in items])
            if len(vectors) == 0: return
            new_ids = vectors[:, 0].astype(jnp.int32)
            new_coords = vectors[:, 1:].astype(jnp.float32)

        updated_mask, updated_ids, updated_coords = self._apply_union(
            fast_refs['active_mask'],
            fast_refs['ids'],
            fast_refs['coords'],
            new_ids,
            new_coords
        )

        fast_refs['active_mask'] = updated_mask
        fast_refs['ids'] = updated_ids
        fast_refs['coords'] = updated_coords