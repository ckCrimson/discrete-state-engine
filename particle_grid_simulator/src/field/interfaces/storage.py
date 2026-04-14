# Contract for Field Module hardware memory allocation
from abc import ABC, abstractmethod
from typing import NamedTuple, Callable, Optional, Tuple, Any
import numpy as np

from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract, IKernelStorage
from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebra


class FieldKernelFastRef(NamedTuple):
    """
    STRUCT: Zero-overhead, immutable data container for Numba.
    """
    state_array: np.ndarray
    field_array: np.ndarray
    is_mapped_array: np.ndarray
    field_function: Optional[Callable[[np.ndarray], np.ndarray]]
    normalized_field_array: np.ndarray



class IFieldKernelDataContract(KernelDataContract):
    """
    CONTRACT: Marker interface for the Field's hardware blueprint.
    """
    pass


from typing import Any, Callable, Optional, Type


# from your_interfaces import IFieldKernelDataContract, IFieldAlgebra

class FieldKernelDataContract(IFieldKernelDataContract):
    def __init__(
            self,
            state_dimensions: int,
            field_dimensions: int,
            algebra: 'IFieldAlgebra',
            state_class_ref: Type,
            mapper_func: Optional[Callable] = None,  # <--- Added
            initial_capacity: int = 100_000,
            growth_factor: float = 1.5,
            max_capacity: int = 10_000_000,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.state_dimensions = state_dimensions
        self.field_dimensions = field_dimensions
        self.algebra = algebra

        self.state_class_ref = state_class_ref  # <--- Blueprinted
        self.mapper_func = mapper_func  # <--- Blueprinted

        self.initial_capacity = initial_capacity
        self.growth_factor = growth_factor
        self.max_capacity = max_capacity


class IFieldKernelStorage(IKernelStorage):
    """
    CONTRACT: The hardware-friendly memory layout for the Field.
    A pure, stripped-down reflection of the Domain's IFieldMapper.
    """

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Metadata: Returns the shape of the state array without accessing the array itself."""
        pass

    @property
    @abstractmethod
    def field_shape(self) -> Tuple[int, ...]:
        """Metadata: Returns the shape of the field array without accessing the array itself."""
        pass

    @property
    @abstractmethod
    def state_array(self) -> np.ndarray:
        """Shape: Should match self.state_shape"""
        pass

    @property
    @abstractmethod
    def field_array(self) -> np.ndarray:
        """Shape: Should match self.field_shape"""
        pass

    @property
    @abstractmethod
    def normalized_field_array(self) -> np.ndarray:
        """Shape: Should match self.field_shape. Pre-allocated memory for normalization."""
        pass

    @property
    @abstractmethod
    def field_function(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """The JIT-compiled mathematical rule for lazy evaluation."""
        pass

    @abstractmethod
    def get_fast_ref(self) -> FieldKernelFastRef:
        """
        Instantly returns the execution struct required by the Numba Utility.
        Returns: FieldKernelFastRef(state_array, field_array, field_function)
        """
        pass

    @abstractmethod
    def fast_refs(self) -> Any:
        self.get_fast_ref()



    @abstractmethod
    def set_field_function(self, func: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        """Sets the compiled JIT function into the storage."""
        pass