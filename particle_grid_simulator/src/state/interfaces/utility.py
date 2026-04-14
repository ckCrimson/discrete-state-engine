from abc import abstractmethod
from typing import Any, Callable
from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility
from .storage import StateFastRefs


class IStateUtility(IKernelUtility):
    """
    Strict contract for the State Module's high-speed hardware math.
    These methods map directly to the domain's StateSpaceUtility,
    but operate purely on raw memory arrays (fast_refs).
    """

    @abstractmethod
    def union_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        """
        Hardware equivalent of adding new states.
        Must find empty slots (active_mask == 0) and overwrite them with new data.
        """
        pass

    @abstractmethod
    def intersection_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        """
        Hardware equivalent of filtering valid states.
        Must set active_mask to 0 for any ID not present in the valid set.
        """
        pass

    @abstractmethod
    def map_inplace(self, fast_refs: StateFastRefs, **kwargs) -> None:
        """
        Hardware equivalent of mapping a transformation.
        e.g., applying velocity to position for all particles where active_mask == 1.
        """
        pass

    @abstractmethod
    def filter_inplace(self, fast_refs: dict, predicate_func: Callable, **kwargs) -> None:
        """
        Reads the hardware arrays and reconstructs the immutable
        Python ParticleState objects for the renderer/domain logic.
        """
        pass