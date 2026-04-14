from abc import abstractmethod
from typing import Any, List

from hpc_ecs_core.src.hpc_ecs_core.interfaces import ITranslator


class IFieldTranslator(ITranslator):
    """
    CONTRACT: Bridges the Domain Object model and the Hardware Array model.
    Strictly implements the core ITranslator, expecting FieldKernelFastRef 
    for the fast_refs argument.
    """

    @abstractmethod
    def bake(self, fast_refs: Any, initial_data: Any) -> None:
        """
        FULL REBUILD: Parses the domain mapper and populates the pre-allocated arrays.
        - Expected fast_refs type: FieldKernelFastRef
        - Expected initial_data type: IFieldMapper

        *Architectural Note:* The concrete implementation should cache a reference 
        to `initial_data.algebra` here so `bake_incremental` can use it later.
        """
        pass

    @abstractmethod
    def bake_incremental(self, fast_refs: Any, queue: List[tuple] , **kwargs) -> None:
        """
        SURGICAL UPDATE: Flushes queued dynamic changes (SET, ADD, CLEAR).
        - Expected fast_refs type: FieldKernelFastRef
        - Modifies fast_refs.field_array in-place.
        """
        pass

    @abstractmethod
    def sync(self, fast_refs: Any, **kwargs: Any) -> None:
        """
        REVERSE TRANSLATION: Pushes hardware array changes back up to the Domain.
        - Expected fast_refs type: FieldKernelFastRef
        - Expected kwargs: `domain_mapper=IFieldMapper`
        """
        pass

    @abstractmethod
    def get_hardware_indices(self, states: Any) -> Any:
        pass