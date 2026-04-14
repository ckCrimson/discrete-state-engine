from abc import abstractmethod, ABC
from typing import Any, TypedDict
from hpc_ecs_core.src.hpc_ecs_core.store import IKernelStorage

class StateFastRefs(TypedDict):
    """
    Strict typing for the raw memory pointers.
    Any backend (Numba/JAX) MUST expose a dictionary with exactly these keys
    pointing to their respective contiguous arrays.
    """
    ids: Any          # e.g., 1D NumPy/JAX array of integers
    active_mask: Any  # e.g., 1D array of booleans or ints (0=dead, 1=alive)
    velocities: Any   # e.g., 2D array of floats (N, dimensions)
    masses: Any       # e.g., 1D array of floats

class IStateStorage(IKernelStorage, ABC):
    """
    Contract for State Module hardware memory allocation.
    """
    @property
    @abstractmethod
    def fast_refs(self) -> StateFastRefs:
        """Must return the specific StateFastRefs dictionary."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Returns the number of active entities in this specific storage backend."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Wipes the active memory for this specific backend."""
        pass

    @abstractmethod
    def get_valid_state_vectors(self) -> Any:
        """
        Returns a dense 2D array of all active state vectors.
        Format per row: [ID, coord_1, coord_2, ...]
        """
        pass
