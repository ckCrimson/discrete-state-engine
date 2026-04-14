from typing import Iterable, Union, Any, Callable

import numpy as np

from hpc_ecs_core.src.hpc_ecs_core.manager import BaseComponentManager
from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract, SyncState

from ..domain.state_domain import State, StateSpace
from ..interfaces.storage import IStateStorage, StateFastRefs
from ..interfaces.translator import IStateTranslator
from ..interfaces.utility import IStateUtility
from .component_enums import CommandType

StateData = Union[State, Any]


class StateComponentManager(BaseComponentManager):
    """
    The concrete API for the State Module.
    Handles queueing and synchronization. Math operations are deferred
    directly to the injected IStateUtility.
    """

    def __init__(
            self,
            contract: KernelDataContract,
            storage: IStateStorage,
            translator: IStateTranslator,
            utility: IStateUtility,
            initial_data: Iterable[StateData] = None
    ) -> None:
        super().__init__(utility,contract, storage, translator, initial_data)

        # Expose the strongly-typed utility so the user can call math operations directly
        self.translator: IStateTranslator = translator
        self.utility: IStateUtility = utility
        self.fast_refs: StateFastRefs = self.fast_refs

    @classmethod
    def create_raw_state_space(
            cls,
            contract: KernelDataContract,
            storage: IStateStorage,
            translator: IStateTranslator,
            utility: IStateUtility,
            raw_data: Iterable[Any]
    ) -> 'StateComponentManager':
        """Alternative constructor to bypass domain overhead."""
        return cls(contract, storage, translator, utility, initial_data=raw_data)

    def add_state(self, data: Iterable[StateData]) -> None:
        # Generate a simple sequential ID
        cmd_id = len(self.command_buffer.queue)
        self.command_buffer.add_command(cmd_id, CommandType.ADD_BATCH.name, data)
        self.store.sync_state = SyncState.DOMAIN_DIRTY

    def delete_state(self, data: Iterable[StateData]) -> None:
        cmd_id = len(self.command_buffer.queue)
        self.command_buffer.add_command(cmd_id, CommandType.DELETE_BATCH.name, data)
        self.store.sync_state = SyncState.DOMAIN_DIRTY

    def get_state_space(self) -> StateSpace:
        """Pays the translation tax only if the hardware changed the data."""
        if self.store.sync_state == SyncState.EXECUTION_DIRTY:
            synced_states = self.translator.sync(self.fast_refs)
            self.store.domain_data = StateSpace(states=synced_states)
            self.store.sync_state = SyncState.CLEAN

        return self.store.domain_data

    def get_raw_state_space(self) -> StateFastRefs:
        """Returns the naked arrays for absolute maximum speed."""
        return self.store.storage.get_valid_state_vectors()

    def union_in_place(self, data: Union[Iterable[StateData], StateSpace]) -> None:
        """Merges data and automatically flags the execution state as dirty."""
        self.utility.union_inplace(self.fast_refs, data=data)
        self.store.sync_state = SyncState.EXECUTION_DIRTY

    def intersection_in_place(self, data: Union[Iterable[StateData], StateSpace]) -> None:
        """Filters data and automatically flags the execution state as dirty."""
        self.utility.intersection_inplace(self.fast_refs, data=data)
        self.store.sync_state = SyncState.EXECUTION_DIRTY

    def map_in_place(self, transform_func: Callable) -> None:
        """Applies transformation and automatically flags the execution state as dirty."""
        self.utility.map_inplace(self.fast_refs, transform_func=transform_func)
        self.store.sync_state = SyncState.EXECUTION_DIRTY

    def clear(self) -> None:
        """
        Public API to reset the state space.
        Hard-resets the hardware masks and clears the command buffer.
        """
        # 1. Reset the hardware (The "Fast" part)
        self.store.storage.clear()

    @property
    def count(self) -> int:
        """
        Returns the number of active states in the manifold.
        Calculated at hardware speed via the active_mask.
        """
        return self.store.storage.count

    @classmethod
    def build(
            cls,
            initial_data: Any = None,
            dimensions: int = 2,
            max_count: int = 100000,
            kernel: str = 'Numba'
    ) -> 'StateComponentManager':
        """
        The primary factory method for end-users to initialize the Field Dynamic System.
        Automatically provisions the correct hardware backends and scales memory.
        """
        # 1. Determine the length of the incoming data to ensure safe capacity
        data_len = 0
        if initial_data is not None:
            if isinstance(initial_data, dict) and 'ids' in initial_data:
                data_len = len(initial_data['ids'])
            elif hasattr(initial_data, 'states'):
                data_len = len(initial_data.states)
            else:
                try:
                    data_len = len(initial_data)
                except TypeError:
                    pass  # Fallback if it's an exhaustible generator

        # Scale up automatically if data exceeds max_count.
        # We add a 10% buffer so the user has room to ADD states later without crashing.
        safe_max_count = max(max_count, int(data_len * 1.1))

        # 2. Setup the Data Contract
        contract = KernelDataContract(max_count=safe_max_count, dimensions=dimensions)

        # 3. Provision the Hardware Backend
        kernel_target = kernel.lower()

        if kernel_target == 'numba':
            # Local imports prevent circular dependency issues and speed up boot times
            # if a user only wants to use a specific backend.
            from particle_grid_simulator.src.state.kernel.numba.storage.storage_v1 import NumbaStateStorage
            from particle_grid_simulator.src.state.kernel.numba.translator.translator_v1 import NumbaStateTranslator
            from particle_grid_simulator.src.state.kernel.numba.utility.utility_v1 import NumbaStateUtility

            storage = NumbaStateStorage(contract)
            translator = NumbaStateTranslator()
            utility = NumbaStateUtility()

        elif kernel_target == 'jax':
            from particle_grid_simulator.src.state.kernel.jax.storage.storage_v1 import JaxStateStorage
            from particle_grid_simulator.src.state.kernel.jax.translator.translator_v1 import JaxStateTranslator
            from particle_grid_simulator.src.state.kernel.jax.utility.utility_v1 import JaxStateUtility

            storage = JaxStateStorage(contract)
            translator = JaxStateTranslator()
            utility = JaxStateUtility()

        else:
            raise ValueError(f"Unknown compute kernel: '{kernel}'. Supported: 'Numba', 'Jax'")

        # 4. Instantiate the Manager and Bake Data
        manager = cls(
            contract=contract,
            storage=storage,
            translator=translator,
            utility=utility,
            initial_data=initial_data
        )

        # Note: If your __init__ doesn't automatically bake initial_data,
        # do it here:
        # if initial_data is not None:
        #     manager.add_state(initial_data)
        #     manager.commit_frame()

        return manager

    def filter_in_place(self, predicate_func: Callable) -> None:
        """
        Filters the state space in-place using a vectorized condition.
        Delegates array manipulation entirely to the compute kernel.
        """
        self.utility.filter_inplace(self.fast_refs, predicate_func)

