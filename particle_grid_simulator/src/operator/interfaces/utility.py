from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility


@dataclass
class OperatorContext:
    """
    PRIVATE DTO: The universal data payload.
    This acts as the void* or generic struct to pass data across the abstraction barrier.
    """
    primary_data: Any = None
    context_data: Tuple[Any, ...] = ()

class IOperatorCMUtility(IKernelUtility):
    """
    The Pure Interface.
    Strictly defines the 4 execution pathways without dictating the underlying technology.
    """
    @abstractmethod
    def evolve(self, func: Any, context: OperatorContext) -> Any:
        pass

    @abstractmethod
    def evolve_inplace(self, func: Any, context: OperatorContext) -> None:
        pass

    @abstractmethod
    def evolve_batch(self, func: Any, context: OperatorContext) -> Any:
        pass

    @abstractmethod
    def evolve_batch_inplace(self, func: Any, context: OperatorContext) -> None:
        pass