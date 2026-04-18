from typing import TypeVar, Callable, Any, Type

from particle_grid_simulator.src.operator.domain.interfaces.operator import IOperatorData

T = TypeVar('T')

class GenericOperatorData(IOperatorData):
    """
    A single, universal Operator implementation that satisfies IOperatorData.
    It can wrap ANY evolution function, whether Classic or Field-based.
    """
    def __init__(self, evolution_function: Callable[..., T], state_class_ref: Type[T]):
        # Use private variables to store the data
        self._evolution_function = evolution_function
        self._state_class_ref = state_class_ref

    @property
    def evolution_function(self) -> Callable[..., T]:
        return self._evolution_function

    @property
    def state_class_ref(self) -> Type[T]:
        return self._state_class_ref