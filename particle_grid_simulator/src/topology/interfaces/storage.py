# Contract for Topology Module hardware memory allocation
from dataclasses import dataclass
from typing import  Type, Callable
from abc import  abstractmethod


from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelStorage


@dataclass
class ITopologyFastRef:
    """
    The raw, contiguous memory layout that Numba @njit functions will process.
    """



class ITopologyStorage(IKernelStorage):

    @abstractmethod
    def get_fast_ref(self) -> ITopologyFastRef:
        """Returns the raw memory struct for fast Numba/C++ traversal."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all arrays and resets steps_prepared to 0, keeping capacity."""
        pass

    @abstractmethod
    def get_number_of_states_prepared(self) -> int:
        """Returns how deeply the adjacency matrix has been built."""
        pass

    @abstractmethod
    def get_neighbour_function(self) -> Callable:
        """Returns the domain logic function (e.g., move_1d, move_2d)."""
        pass

    @abstractmethod
    def get_state_class_reference(self) -> Type:
        """Returns the Python class used to instantiate states (e.g., State)."""
        pass