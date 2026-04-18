from typing import Any, Tuple, Optional

from hpc_ecs_core.src.hpc_ecs_core import BaseComponentManager
from particle_grid_simulator.src.operator.interfaces.utility import OperatorContext, IOperatorCMUtility


# Assuming BaseComponentManager, IOperatorCMUtility, and OperatorContext are imported

class OperatorComponentManager(BaseComponentManager):
    """
    The Operator Pipeline.
    Zero allocation, zero indirection.
    """

    def __init__(self, operator_data: Any, utility: 'IOperatorCMUtility'):
        super().__init__(
            utility=utility,
            contract=None,
            raw_storage=None,
            translator=None
        )

        # ==========================================
        # DOD OPTIMIZATION: CACHE POINTERS
        # Extract properties immediately to kill indirection overhead in hot loops.
        # ==========================================
        self._evolution_function = operator_data.evolution_function
        self._state_class_ref = operator_data.state_class_ref

        # Pre-allocated DOD memory map
        self._context = OperatorContext()

    @classmethod
    def create_raw(
            cls,
            evolution_func: Any,
            utility: 'IOperatorCMUtility',
            state_class_ref: Optional[Any] = None
    ) -> 'OperatorComponentManager':
        """
        Fast-path instantiation. Bypasses the Domain Data wrapper entirely.
        Perfect for raw mathematical execution and quick script testing.
        """
        # 1. Allocate without triggering the base class __init__
        instance = cls.__new__(cls)

        # 2. Manually satisfy the BaseComponentManager static requirements
        instance.utility = utility
        instance.is_static = True

        # 3. Inject the Operator pointers directly
        instance._evolution_function = evolution_func
        instance._state_class_ref = state_class_ref
        instance._context = OperatorContext()

        return instance

    @classmethod
    def create_utility_cm(cls, utility_class: type) -> 'OperatorComponentManager':
        raise NotImplementedError(
            "OperatorComponentManager requires operator_data. "
            "Instantiate it directly via __init__."
        )

    def _map_context(self, primary: Any, context_args: Tuple[Any, ...]) -> None:
        self._ensure_static()
        self._context.primary_data = primary
        self._context.context_data = context_args

    # ==========================================
    # THE 4 PURE APIs (Now using direct pointers)
    # ==========================================
    def evolve(self, state: Any, *context: Any) -> Any:
        self._map_context(state, context)
        raw_result = self.utility.evolve(self._evolution_function, self._context)

        # FIX: Check if we actually have a class ref before trying to wrap it
        if self._state_class_ref is not None and not isinstance(raw_result, self._state_class_ref):
            return self._state_class_ref(raw_result)

        # If instantiated via create_raw without a class ref, just return the raw array
        return raw_result

    def evolve_inplace(self, state: Any, *context: Any) -> None:
        self._map_context(state, context)
        self.utility.evolve_inplace(self._evolution_function, self._context)

    def evolve_batch(self, states: Any, *context: Any) -> Any:
        self._map_context(states, context)
        return self.utility.evolve_batch(self._evolution_function, self._context)

    def evolve_batch_inplace(self, states: Any, *context: Any) -> None:
        self._map_context(states, context)
        self.utility.evolve_batch_inplace(self._evolution_function, self._context)