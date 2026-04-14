from functools import partial
from typing import Set, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put

from particle_grid_simulator.src.state.domain.state_domain import State
from particle_grid_simulator.src.state.interfaces.storage import StateFastRefs
from particle_grid_simulator.src.state.interfaces.translator import IStateTranslator


class JaxStateTranslator(IStateTranslator):
    """
    Handles the high-speed transfer of data between CPU (Python)
    and GPU (JAX Device Arrays).
    """

    def bake(self, fast_refs: StateFastRefs, initial_data: Any = None) -> None:
        """Initial mass-upload to GPU VRAM."""
        if initial_data is None:
            return

        # 1. Extract IDs and Coords (handling our Fast Path dictionary)
        if isinstance(initial_data, dict):
            ids = jnp.array(initial_data['ids'], dtype=jnp.int32)
            coords = jnp.array(initial_data['coords'], dtype=jnp.float32)
        else:
            # Fallback for State objects (Slow Path)
            vectors = jnp.array([s.vector for s in initial_data])
            ids = vectors[:, 0].astype(jnp.int32)
            coords = vectors[:, 1:].astype(jnp.float32)

        count = len(ids)

        # 2. Update the Device Arrays (XLA replacement)
        # We use .at[].set() for functional updates in JAX
        fast_refs['ids'] = fast_refs['ids'].at[:count].set(ids)
        fast_refs['coords'] = fast_refs['coords'].at[:count, :].set(coords)
        fast_refs['active_mask'] = fast_refs['active_mask'].at[:count].set(1)

    @partial(jax.jit, static_argnums=(0,))
    def _jit_add_batch(self, fast_refs, new_ids, new_coords):
        """
        Compiled XLA kernel to find empty slots and insert data
        without a Python loop.
        """
        # Find indices where mask is 0
        empty_indices = jnp.where(fast_refs['active_mask'] == 0, size=len(new_ids))[0]

        # Bulk update via scatter-update
        new_ids_refs = fast_refs['ids'].at[empty_indices].set(new_ids)
        new_coords_refs = fast_refs['coords'].at[empty_indices].set(new_coords)
        new_mask_refs = fast_refs['active_mask'].at[empty_indices].set(1)

        return new_ids_refs, new_coords_refs, new_mask_refs

    def bake_incremental(self, fast_refs: StateFastRefs, command_queue: list, **kwargs) -> None:
        """
        Processes the Manager's queue and pushes updates to the GPU.
        """
        for _, cmd_name, data in command_queue:
            if cmd_name == 'ADD_BATCH':
                # Fast Path Extraction
                ids = jnp.array(data['ids'])
                coords = jnp.array(data['coords'])

                # Execute the JIT-compiled GPU insertion
                ids_res, coords_res, mask_res = self._jit_add_batch(
                    fast_refs, ids, coords
                )

                # Update the references in the Manager's dictionary
                fast_refs['ids'] = ids_res
                fast_refs['coords'] = coords_res
                fast_refs['active_mask'] = mask_res

    def sync(self, fast_refs: StateFastRefs) -> Set[Any]:
        """
        The 'Emergency Exit': Pulls data from GPU back to CPU
        and reconstructs Python objects.
        """
        # 1. Pull from Device to Host (This is the slow part!)
        ids = np.array(fast_refs['ids'])
        coords = np.array(fast_refs['coords'])
        mask = np.array(fast_refs['active_mask'])

        active = mask == 1
        return {State(np.concatenate([[ids[i]], coords[i]])) for i in np.where(active)[0]}