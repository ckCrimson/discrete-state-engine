import numpy as np
from typing import Any, Tuple, Optional, Callable, Type

from particle_grid_simulator.src.field.interfaces.storage import (
    IFieldKernelStorage,
    FieldKernelDataContract,
    FieldKernelFastRef
)


class NumbaComplexFieldKernelStorage(IFieldKernelStorage):
    """
    CONCRETE STORAGE: Fulfills IFieldKernelStorage for Quantum/Wave fields.
    Strictly handles the immutability of the FieldKernelFastRef NamedTuple.
    """

    def __init__(self, contract: FieldKernelDataContract) -> None:
        self._contract = contract
        self._steps_prepared = 0
        self._field_function: Optional[Callable[[np.ndarray], np.ndarray]] = None

        init_capacity = contract.initial_capacity
        s_dim = contract.state_dimensions
        f_dim = contract.field_dimensions

        # 1. Allocate the physical arrays
        s_array = np.zeros((init_capacity, s_dim), dtype=np.float64)
        f_array = np.zeros((init_capacity, f_dim), dtype=np.complex128)
        norm_array = np.zeros((init_capacity, f_dim), dtype=np.complex128)

        # ADDED: The missing boolean array for mapping tracking
        is_mapped = np.zeros(init_capacity, dtype=np.bool_)

        # 2. Strict NamedTuple Instantiation
        self._fast_refs = FieldKernelFastRef(
            state_array=s_array,
            field_array=f_array,
            is_mapped_array=is_mapped,
            field_function=self._field_function,
            normalized_field_array=norm_array
        )

    # ==========================================
    # IFIELDKERNELSTORAGE PROPERTY FULFILLMENT
    # ==========================================
    @property
    def state_shape(self) -> Tuple[int, ...]: return self._fast_refs.state_array.shape

    @property
    def field_shape(self) -> Tuple[int, ...]: return self._fast_refs.field_array.shape

    @property
    def state_array(self) -> np.ndarray: return self._fast_refs.state_array

    @property
    def field_array(self) -> np.ndarray: return self._fast_refs.field_array

    @property
    def normalized_field_array(self) -> np.ndarray: return self._fast_refs.normalized_field_array

    @property
    def field_function(self) -> Optional[Callable[[np.ndarray], np.ndarray]]: return self._fast_refs.field_function

    @property
    def fast_refs(self) -> FieldKernelFastRef: return self._fast_refs

    # ==========================================
    # IFIELDKERNELSTORAGE METHOD FULFILLMENT
    # ==========================================
    def get_fast_ref(self) -> FieldKernelFastRef:
        return self._fast_refs

    def set_field_function(self, func: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        self._field_function = func
        # FIX: Because NamedTuple is immutable, we must recreate it to update the function pointer
        self._fast_refs = FieldKernelFastRef(
            state_array=self._fast_refs.state_array,
            field_array=self._fast_refs.field_array,
            is_mapped_array=self._fast_refs.is_mapped_array,
            field_function=self._field_function,
            normalized_field_array=self._fast_refs.normalized_field_array
        )

    # ==========================================
    # IKERNELSTORAGE METHOD FULFILLMENT
    # ==========================================
    def clear(self) -> None:
        # We can mutate the data INSIDE the arrays without breaking Tuple immutability
        self._fast_refs.state_array.fill(0.0)
        self._fast_refs.field_array.fill(0.0 + 0.0j)
        self._fast_refs.normalized_field_array.fill(0.0 + 0.0j)
        self._fast_refs.is_mapped_array.fill(False)
        self._steps_prepared = 0

    def resize(self, new_capacity: int) -> None:
        old_states = self._fast_refs.state_array
        old_fields = self._fast_refs.field_array
        old_norms = self._fast_refs.normalized_field_array
        old_mapped = self._fast_refs.is_mapped_array

        s_dim = old_states.shape[1]
        f_dim = old_fields.shape[1]

        # Allocate new, larger blocks
        new_states = np.zeros((new_capacity, s_dim), dtype=np.float64)
        new_fields = np.zeros((new_capacity, f_dim), dtype=np.complex128)
        new_norms = np.zeros((new_capacity, f_dim), dtype=np.complex128)
        new_mapped = np.zeros(new_capacity, dtype=np.bool_)

        # Copy existing data
        curr_size = min(len(old_states), new_capacity)
        new_states[:curr_size] = old_states[:curr_size]
        new_fields[:curr_size] = old_fields[:curr_size]
        new_norms[:curr_size] = old_norms[:curr_size]
        new_mapped[:curr_size] = old_mapped[:curr_size]

        # FIX: Reconstruct the NamedTuple with the new array pointers
        self._fast_refs = FieldKernelFastRef(
            state_array=new_states,
            field_array=new_fields,
            is_mapped_array=new_mapped,
            field_function=self._field_function,
            normalized_field_array=new_norms
        )

    def get_state_class_reference(self) -> Type:
        return self._contract.state_class_ref

    def get_number_of_states_prepared(self) -> int:
        return self._steps_prepared