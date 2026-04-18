from typing import Any

from particle_grid_simulator.src.operator.domain.data.operator import  T
from particle_grid_simulator.src.operator.domain.interfaces.operator import IOperatorUtility, IOperatorData


class GenericOperatorUtility(IOperatorUtility):
    """
        Concrete implementation satisfying IOperatorUtility.
    """
    @staticmethod
    def evolve(operator: IOperatorData[T], data_context: Any) -> T:
        # 1. Execute the mathematical rule using the single data_context
        raw_result = operator.evolution_function(data_context)

        # 2. Enforce the mathematical contract (ensure S is returned)
        if not isinstance(raw_result, operator.state_class_ref):
            return operator.state_class_ref(raw_result)

        return raw_result