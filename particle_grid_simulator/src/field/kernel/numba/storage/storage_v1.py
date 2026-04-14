from typing import Optional, Tuple, Type, Callable
import numpy as np
from numba import njit

from particle_grid_simulator.src.field.interfaces.storage import IFieldKernelStorage, FieldKernelDataContract, \
    FieldKernelFastRef


class NumbaFieldKernelStorage(IFieldKernelStorage):
    def __init__(self, contract: 'FieldKernelDataContract') -> None:
        self._contract = contract
        self._capacity = contract.initial_capacity

        # Pre-allocate Arrays
        self._state_array = np.zeros((self._capacity, contract.state_dimensions), dtype=np.float64)
        self._field_array = np.zeros((self._capacity, contract.field_dimensions), dtype=np.float64)

        # ---> THE MISSING ALLOCATION <---
        self._normalized_field_array = np.zeros((self._capacity, contract.field_dimensions), dtype=np.float64)

        self._is_mapped_array = np.zeros(self._capacity, dtype=np.bool_)

        # Save State Reference
        self._state_class_ref = contract.state_class_ref

        # JIT Compile the function securely inside the Numba layer
        self._field_function: Optional[Callable] = None
        if contract.mapper_func is not None:
            self._field_function = njit(contract.mapper_func)

    def get_state_class_reference(self) -> Type:
        return self._state_class_ref

    @property
    def fast_refs(self) -> 'FieldKernelFastRef':
        """Constructs the immutable execution struct perfectly on the first try."""
        return FieldKernelFastRef(
            state_array=self._state_array,
            field_array=self._field_array,
            is_mapped_array=self._is_mapped_array,
            field_function=self._field_function,
            normalized_field_array=self._normalized_field_array  # ---> THE MISSING ARGUMENT <---
        )

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self._capacity, self._contract.state_dimensions)

    @property
    def field_shape(self) -> Tuple[int, ...]:
        return (self._capacity, self._contract.field_dimensions)

    @property
    def state_array(self) -> np.ndarray:
        return self._state_array

    @property
    def field_array(self) -> np.ndarray:
        return self._field_array

    @property
    def is_mapped_array(self) -> np.ndarray:
        return self._is_mapped_array

    @property
    def normalized_field_array(self) -> np.ndarray:
        return self._normalized_field_array

    @property
    def field_function(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        return self._field_function

    def get_fast_ref(self) -> FieldKernelFastRef:
        """Instantly returns the execution struct for the Numba Utility."""
        return self.fast_refs

    # ==========================================
    # HARDWARE MEMORY MANAGEMENT
    # ==========================================

    def set_field_function(self, func: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        """Allows the Translator to attach the compiled JIT function during bake()."""
        self._field_function = func

    def resize(self, required_capacity: int) -> None:
        """
        Reallocates memory using a geometric growth factor.
        Called by the Translator if an incoming command buffer exceeds bounds.
        """
        if required_capacity <= self._capacity:
            return

        new_capacity = max(
            int(self._capacity * self._contract.growth_factor),
            required_capacity
        )

        if new_capacity > self._contract.max_capacity:
            raise MemoryError(
                f"Field Storage resize ({new_capacity}) exceeds Max Capacity ({self._contract.max_capacity})."
            )

        # Allocate new continuous blocks
        new_states = np.zeros((new_capacity, self._contract.state_dimensions), dtype=np.float64)
        new_fields = np.zeros((new_capacity, self._contract.field_dimensions), dtype=np.float64)
        new_mapped = np.zeros(new_capacity, dtype=np.bool_)
        new_normalized = np.zeros((new_capacity, self._contract.field_dimensions), dtype=np.float64)

        # Fast memory copy from old to new
        new_states[:self._capacity] = self._state_array
        new_fields[:self._capacity] = self._field_array
        new_mapped[:self._capacity] = self._is_mapped_array
        new_normalized[:self._capacity] = self._normalized_field_array

        # Update pointers and capacity
        self._state_array = new_states
        self._field_array = new_fields
        self._is_mapped_array = new_mapped
        self._normalized_field_array = new_normalized
        self._capacity = new_capacity