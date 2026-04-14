from enum import Enum, auto
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class SyncState(Enum):
    CLEAN = auto()
    DOMAIN_DIRTY = auto()
    EXECUTION_DIRTY = auto()

class KernelDataContract:
    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

class IKernelStorage(ABC):
    @property
    @abstractmethod
    def fast_refs(self) -> Any:
        """Expose raw memory pointers (fast_refs) to prevent cache thrashing."""
        pass

class ITranslator(ABC):
    @abstractmethod
    def bake(self, fast_refs: Any, initial_data:Any) -> None:
        pass

    @abstractmethod
    def bake_incremental(self, fast_refs: Any, queue: List[tuple], **kwargs) -> None:
        pass

    @abstractmethod
    def sync(self, fast_refs: Any, **kwargs) -> None:
        pass

class IKernelUtility(ABC):
    """Empty marker interface for kernel utilities."""
    pass