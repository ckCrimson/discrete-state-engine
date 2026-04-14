# 2. Translator now implements ITranslator
from abc import abstractmethod
from typing import Dict, Any, List, Iterable, Type

import numpy as np

from hpc_ecs_core.src.hpc_ecs_core.interfaces import ITranslator


class ITopologyTranslator(ITranslator):

    # We will use bake() to handle the initial translation of Python states
    # into our Numba FastRef Handle Map
    @abstractmethod
    def bake(self, fast_refs: Dict[str, Any], initial_data: Any) -> None:
        pass

    # Skipped as requested!
    def bake_incremental(self, fast_refs: Dict[str, Any], queue: List[tuple]) -> None:
        pass

    @abstractmethod
    def sync(self, fast_refs: Any, **kwargs) -> None:
        pass

    # Our specific topology translation helpers
    @abstractmethod
    def to_raw_vector(self, state_obj: Any) -> np.ndarray:
        pass

    @abstractmethod
    def to_state_objects(self, raw_vectors: Iterable[np.ndarray], state_class: Type) -> Iterable[Any]:
        pass