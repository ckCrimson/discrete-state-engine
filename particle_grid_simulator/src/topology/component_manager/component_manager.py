from typing import Optional, Any, Iterable, Type, Callable, Union
import numpy as np

# Assuming the BaseComponentManager and Interfaces are imported here
from hpc_ecs_core.src.hpc_ecs_core.manager import BaseComponentManager
from hpc_ecs_core.src.hpc_ecs_core.interfaces import SyncState

from particle_grid_simulator.src.topology.interfaces.storage import ITopologyStorage
from particle_grid_simulator.src.topology.interfaces.translator import ITopologyTranslator
from particle_grid_simulator.src.topology.interfaces.utility import ITopologyUtility
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import TopologyKernelDataContract


class TopologyComponentManager(BaseComponentManager):
    def __init__(
            self,
            contract: TopologyKernelDataContract,  # Standardized name
            storage: ITopologyStorage,
            translator: ITopologyTranslator,
            utility: ITopologyUtility,
            topology: Any = None
    ) -> None:

        # 1. Safely route to the Base Manager using explicit kwargs
        super().__init__(
            utility=utility,
            contract=contract,
            raw_storage=storage,
            translator=translator,
            initial_data=topology
        )

        # 2. Only setup stateful elements if this is not a static math bridge
        if not self.is_static:
            self._contract = contract

            # Extract metadata straight from the DOD contract
            self._neighbour_func = contract.neighbour_function
            self._state_class_ref = contract.state_class_reference

            # Store the domain object explicitly for the pull-sync
            self.store.domain_object = topology
            self.store.sync_state = SyncState.CLEAN

    @classmethod
    def create_from_raw_data(
            cls,
            data_contract: TopologyKernelDataContract,
            storage: ITopologyStorage,
            translator: ITopologyTranslator,
            utility: ITopologyUtility
    ) -> "TopologyComponentManager":
        """
        RAW CONSTRUCTOR:
        Directly injects the kernel logic. No 'Topology' domain object is created or used.
        The manager relies entirely on the Contract for class refs and hardware functions.
        """
        return cls(
            contract=data_contract,
            storage=storage,
            translator=translator,
            utility=utility,
            topology=None
        )

    # ==========================================
    # DOMAIN SYNC (The Pull Architecture)
    # ==========================================
    def get_domain(self) -> Any:
        """
        Returns the high-level Python domain object.
        If the kernel has expanded the graph, it hydrates the cache first.
        """
        self._ensure_stateful()
        if self.store.sync_state == SyncState.DOMAIN_DIRTY:
            self.translator.sync(self.fast_refs, topology=self.store.domain_object)
            self.store.sync_state = SyncState.CLEAN

        return self.store.domain_object

    # ==========================================
    # INTERNAL API HELPERS
    # ==========================================
    def _prepare_input(self, state_in: Union[Any, np.ndarray]) -> np.ndarray:
        if isinstance(state_in, np.ndarray):
            return state_in
        return self.translator.to_raw_vector(state_in)

    def _prepare_output(self, raw_vectors: Iterable[np.ndarray], return_state_class: bool) -> Union[
        Iterable[np.ndarray], Iterable[Any]]:
        # Any time we successfully query and potentially expand the graph,
        # the Python domain cache is now out of date.
        self.store.sync_state = SyncState.DOMAIN_DIRTY

        if return_state_class:
            return self.translator.to_state_objects(raw_vectors, self._state_class_ref)
        return raw_vectors

    # ==========================================
    # THE QUERIES (Stateless Execution Routing)
    # ==========================================

    def get_reachable(self, state_in: Union[Any, np.ndarray], return_state_class: bool = False) -> Iterable[Any]:
        self._ensure_stateful()
        raw_in = self._prepare_input(state_in)
        # Passed fast_refs and neighbour_func to the stateless utility
        raw_out = self.utility.get_reachable(self.fast_refs, self._neighbour_func, raw_in)
        return self._prepare_output(raw_out, return_state_class)

    def get_reachable_multi_step_frontier(self, state_in: Union[Any, np.ndarray], steps: int,
                                          return_state_class: bool = False) -> Iterable[Any]:
        self._ensure_stateful()
        raw_in = self._prepare_input(state_in)
        raw_out = self.utility.get_reachable_multi_step_frontier(self.fast_refs, self._neighbour_func, raw_in, steps)
        return self._prepare_output(raw_out, return_state_class)

    def get_reachable_multi_step_basin(self, state_in: Union[Any, np.ndarray], steps: int,
                                       return_state_class: bool = False) -> Iterable[Any]:
        self._ensure_stateful()
        raw_in = self._prepare_input(state_in)
        raw_out = self.utility.get_reachable_multi_step_basin(self.fast_refs, self._neighbour_func, raw_in, steps)
        return self._prepare_output(raw_out, return_state_class)

    def get_reaching_multi_step_frontier(self, state_in: Union[Any, np.ndarray], steps: int,
                                         return_state_class: bool = False) -> Iterable[Any]:
        self._ensure_stateful()
        raw_in = self._prepare_input(state_in)
        raw_out = self.utility.get_reaching_multi_step_frontier(self.fast_refs, self._neighbour_func, raw_in, steps)
        return self._prepare_output(raw_out, return_state_class)

    def get_reaching_multi_step_basin(self, state_in: Union[Any, np.ndarray], steps: int,
                                      return_state_class: bool = False) -> Iterable[Any]:
        self._ensure_stateful()
        raw_in = self._prepare_input(state_in)
        raw_out = self.utility.get_reaching_multi_step_basin(self.fast_refs, self._neighbour_func, raw_in, steps)
        return self._prepare_output(raw_out, return_state_class)

    # ==========================================
    # GRAPH GENERATION
    # ==========================================

    def warmup(self, initial_states: Iterable[Any], steps: int) -> None:
        """
        Pre-bakes the adjacency graph for a given set of starting states.
        Run this before the main simulation loop to guarantee zero-latency execution.
        """
        self._ensure_stateful()
        for state in initial_states:
            raw_in = self._prepare_input(state)
            self.utility.warmup(self.fast_refs, self._neighbour_func, raw_in, steps)

        # The graph has expanded, so the Python cache is now out of date
        self.store.sync_state = SyncState.DOMAIN_DIRTY

    def prepare_graph(self, initial_states: Iterable[Any], steps: int) -> None:
        """Alias for warmup to match the Orchestrator API expectations."""
        self.warmup(initial_states, steps)

    def compile_storage(self) -> None:
        print("\n" + "=" * 50)
        print("   FATAL DEBUG: INSPECTING NUMBA STORAGE")
        print("=" * 50)

        # Based on your previous traceback, this is the object we want
        actual_storage = self.store.storage

        print(f"1. Object Type: {type(actual_storage)}")

        # This will list EVERY single variable and method inside the object
        print(f"2. All internal attributes:")
        for attr in dir(actual_storage):
            if not attr.startswith('__'):  # Hide the built-in Python junk
                print(f"   -> {attr}")

        print("=" * 50 + "\n")

        # Crash the program on purpose so we can read the console
        raise RuntimeError("DEBUG HALT: Look at the console output above to see the real array names.")