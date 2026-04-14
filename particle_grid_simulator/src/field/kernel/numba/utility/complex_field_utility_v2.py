import numpy as np
import numba as nb
from typing import Any
from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelFastRef
from particle_grid_simulator.src.field.interfaces.utility import IFieldKernelUtility


# ==========================================
# RAW JIT KERNELS (C-Speed Complex Math)
# ==========================================

@nb.njit(cache=True, fastmath=True)
def _complex_add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Element-wise addition of complex vectors."""
    return v1 + v2


@nb.njit(cache=True, fastmath=True)
def _complex_multiply(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Element-wise complex multiplication.
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    Numba natively handles np.complex128 multiplication.
    """
    return v1 * v2


@nb.njit(cache=True, fastmath=True)
def _complex_norm(v: np.ndarray) -> float:
    """
    Calculates the L2 norm (magnitude) of a complex vector.
    For complex z, |z|^2 = z.real^2 + z.imag^2
    """
    mag_sq = 0.0
    for i in range(len(v)):
        mag_sq += v[i].real ** 2 + v[i].imag ** 2
    return np.sqrt(mag_sq)


@nb.njit(cache=True, fastmath=True)
def _batch_complex_add_inplace(t_fields: np.ndarray, s_fields: np.ndarray) -> None:
    for i in range(len(t_fields)):
        for d in range(t_fields.shape[1]):
            t_fields[i, d] += s_fields[i, d]


@nb.njit(cache=True, fastmath=True)
def _batch_complex_mul_inplace(t_fields: np.ndarray, s_fields: np.ndarray) -> None:
    for i in range(len(t_fields)):
        for d in range(t_fields.shape[1]):
            t_fields[i, d] *= s_fields[i, d]


# ==========================================
# THE CLASS WRAPPER
# ==========================================

class NumbaComplexUtility(IFieldKernelUtility):
    """
    Hardware-accelerated complex number operations mapping directly
    to the IFieldKernelUtility contract.
    """

    # --- VECTOR MATH ---
    @staticmethod
    def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return _complex_add(v1, v2)

    @staticmethod
    def multiply_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return _complex_multiply(v1, v2)

    @staticmethod
    def norm_vector(v: np.ndarray) -> float:
        return _complex_norm(v)

    # --- BULK MATH (ALLOCATING) ---
    @staticmethod
    def add_mappers(ref1: FieldKernelFastRef, ref2: FieldKernelFastRef) -> FieldKernelFastRef:
        new_fields = ref1.field_array + ref2.field_array
        return FieldKernelFastRef(
            state_array=ref1.state_array.copy(),
            field_array=new_fields,
            normalized_field_array=np.zeros_like(new_fields, dtype=np.complex128)
        )

    @staticmethod
    def multiply_mappers(ref1: FieldKernelFastRef, ref2: FieldKernelFastRef) -> FieldKernelFastRef:
        new_fields = ref1.field_array * ref2.field_array
        return FieldKernelFastRef(
            state_array=ref1.state_array.copy(),
            field_array=new_fields,
            normalized_field_array=np.zeros_like(new_fields, dtype=np.complex128)
        )

    @staticmethod
    def norm_mapper(ref: FieldKernelFastRef) -> FieldKernelFastRef:
        # Complex norm reduces dimensionality from complex to real scalars
        out_norms = np.zeros((ref.field_array.shape[0], 1), dtype=np.float64)
        NumbaComplexUtility.batch_norm_bridge(ref.state_array, ref.field_array, out_norms)

        return FieldKernelFastRef(
            state_array=ref.state_array.copy(),
            field_array=out_norms,  # Casted to real numbers
            normalized_field_array=out_norms.copy()
        )

    # --- BULK MATH (IN-PLACE) ---
    @staticmethod
    def add_mappers_inplace(target_ref: FieldKernelFastRef, source_ref: FieldKernelFastRef) -> None:
        _batch_complex_add_inplace(target_ref.field_array, source_ref.field_array)

    @staticmethod
    def multiply_mappers_inplace(target_ref: FieldKernelFastRef, source_ref: FieldKernelFastRef) -> None:
        _batch_complex_mul_inplace(target_ref.field_array, source_ref.field_array)

    @staticmethod
    def norm_mapper_inplace(target_ref: FieldKernelFastRef) -> None:
        raise NotImplementedError(
            "In-place complex norm violates dtype rules (complex128 -> float64). Use batch_norm_bridge.")

    # --- DATA RETRIEVAL ---
    @staticmethod
    def get_fields(ref: FieldKernelFastRef, target_states: np.ndarray) -> np.ndarray:
        # Assumes state IDs line up with indices for O(1) fetch
        return ref.field_array[:len(target_states)]

    @staticmethod
    def normalize_field(fast_ref: FieldKernelFastRef) -> None:
        NumbaComplexUtility.batch_normalize_bridge_inplace(fast_ref.state_array, fast_ref.field_array)

    # --- RAW DATA BRIDGE METHODS (The Static Pipeline) ---
    @staticmethod
    def batch_add_bridge_inplace(
            target_states: np.ndarray, target_fields: np.ndarray,
            source_states: np.ndarray, source_fields: np.ndarray, **kwargs: Any
    ) -> None:
        _batch_complex_add_inplace(target_fields, source_fields)

    @staticmethod
    def batch_multiply_bridge_inplace(
            target_states: np.ndarray, target_fields: np.ndarray,
            source_states: np.ndarray, source_fields: np.ndarray, **kwargs: Any
    ) -> None:
        _batch_complex_mul_inplace(target_fields, source_fields)

    @staticmethod
    def batch_norm_bridge(
            states: np.ndarray, fields: np.ndarray, out_norms: np.ndarray, **kwargs: Any
    ) -> None:
        """
        Proof of the architecture's foresight: fields is complex128, out_norms is float64.
        """

        @nb.njit(cache=True, fastmath=True)
        def _calc_norms(f_arr, out_arr):
            for i in range(len(f_arr)):
                mag_sq = 0.0
                for d in range(f_arr.shape[1]):
                    mag_sq += f_arr[i, d].real ** 2 + f_arr[i, d].imag ** 2
                out_arr[i, 0] = np.sqrt(mag_sq)

        _calc_norms(fields, out_norms)

    @staticmethod
    def batch_normalize_bridge_inplace(states: np.ndarray, fields: np.ndarray, **kwargs: Any) -> None:
        @nb.njit(cache=True, fastmath=True)
        def _norm_inplace(f_arr):
            total_sum = 0.0
            for i in range(len(f_arr)):
                for d in range(f_arr.shape[1]):
                    total_sum += np.sqrt(f_arr[i, d].real ** 2 + f_arr[i, d].imag ** 2)

            if total_sum > 0:
                for i in range(len(f_arr)):
                    for d in range(f_arr.shape[1]):
                        # Float division scales both real and imaginary parts
                        f_arr[i, d] /= total_sum

        _norm_inplace(fields)

    # --- JIT DISPATCHERS FOR GENERATOR INJECTION ---
    @staticmethod
    def get_add_kernel() -> Any:
        return _complex_add

    @staticmethod
    def get_multiply_kernel() -> Any:
        return _complex_multiply

    @staticmethod
    def get_norm_kernel() -> Any:
        return _complex_norm