from abc import abstractmethod
from typing import Any, List, Tuple, Iterable, Set, Callable
from hpc_ecs_core.src.hpc_ecs_core.interfaces import ITranslator

# Import your domain realities
from ..domain.state_domain import State
from ..component_manager.component_enums import CommandType
from .storage import StateFastRefs


class IStateTranslator(ITranslator):
    """
    Contract for translating between pure Python ParticleStates
    and the raw hardware arrays.
    """

    @abstractmethod
    def bake(self, fast_refs: StateFastRefs, initial_data: Iterable[State] = None) -> None:
        """Injects initial ParticleStates directly into the C-arrays."""
        pass

    @abstractmethod
    def bake_incremental(
            self,
            fast_refs: StateFastRefs,
            command_queue: List[Tuple[int, CommandType, Any]],
            **kwargs
    ) -> None:
        """
        Parses the domain CommandQueue (ADD_BATCH, DELETE_BATCH)
        and mutates the fast references directly.
        """
        pass

    @abstractmethod
    def sync(self, fast_refs: StateFastRefs, **kwargs) -> Set[State]:
        """
        Reads the hardware arrays and reconstructs the immutable
        Python ParticleState objects for the renderer/domain logic.
        """
        pass