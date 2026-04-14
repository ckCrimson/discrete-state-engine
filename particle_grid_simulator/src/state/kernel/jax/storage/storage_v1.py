import jax.numpy as jnp
from jax import device_put

from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract
from particle_grid_simulator.src.state.interfaces.storage import IStateStorage, StateFastRefs


class JaxStateStorage(IStateStorage):
    def __init__(self, contract: KernelDataContract):
        self._max_count = contract.config['max_count']
        dims = contract.config['dimensions']

        # Initial GPU Allocation
        self._refs = {
            'ids': device_put(jnp.zeros(self._max_count, dtype=jnp.int32)),
            'active_mask': device_put(jnp.zeros(self._max_count, dtype=jnp.uint8)),
            'coords': device_put(jnp.zeros((self._max_count, dims), dtype=jnp.float32))
        }

    @property
    def fast_refs(self) -> StateFastRefs:
        return self._refs

    @property
    def count(self) -> int:
        return int(jnp.sum(self._refs['active_mask']))

    def get_valid_state_vectors(self) -> jnp.ndarray:
        # Vectorized extraction using JAX JIT
        mask = self._refs['active_mask'] == 1
        return jnp.column_stack((
            self._refs['ids'][mask].astype(jnp.float32),
            self._refs['coords'][mask]
        ))

    def get_active_data(self) -> dict:
        mask = self._refs['active_mask'] == 1
        return {
            'ids': self._refs['ids'][mask],
            'coords': self._refs['coords'][mask]
        }

    def clear(self) -> None:
        self._refs['active_mask'] = jnp.zeros_like(self._refs['active_mask'])