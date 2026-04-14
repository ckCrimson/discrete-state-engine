# Numba utility implementation for Field
import numpy as np
from numba import njit
from typing import Any, Tuple

from particle_grid_simulator.src.field.interfaces.storage import FieldKernelFastRef
from particle_grid_simulator.src.field.interfaces.utility import IFieldKernelUtility


# Assuming interfaces are imported
# from particle_grid_simulator.src.field.interfaces.utility import IFieldKernelUtility
# from particle_grid_simulator.src.field.interfaces.storage import FieldKernelFastRef

# ==========================================
# PURE JIT KERNELS (Module Level)
# ==========================================

@njit(cache=True, fastmath=True)
def _add_vectors_kernel(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 + v2

@njit(cache=True, fastmath=True)
def _multiply_vectors_kernel(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 * v2

@njit(cache=True, fastmath=True)
def _norm_vector_kernel(v: np.ndarray) -> np.ndarray:
    """Returns the normalized vector array (required for the Generator loop)."""
    mag = np.linalg.norm(v)
    return v / (mag + 1e-12)




@njit(cache=True, fastmath=True)
def _get_fields_kernel(
        state_array: np.ndarray,
        field_array: np.ndarray,
        is_mapped_array: np.ndarray,
        target_states: np.ndarray
) -> np.ndarray:
    """
    Blazing fast C-level linear search.
    Finds the target state vectors in the state_array and returns corresponding field vectors.
    """
    num_targets = target_states.shape[0]
    state_dim = target_states.shape[1]
    field_dim = field_array.shape[1]
    num_states = state_array.shape[0]

    # Pre-allocate the result array
    result = np.empty((num_targets, field_dim), dtype=np.float64)

    for i in range(num_targets):
        t_state = target_states[i]
        found = False

        for j in range(num_states):
            if not is_mapped_array[j]:
                continue

            # Check for coordinate match (Vector equality)
            match = True
            for d in range(state_dim):
                if state_array[j, d] != t_state[d]:
                    match = False
                    break

            if match:
                # Copy the field vector into the result
                for f in range(field_dim):
                    result[i, f] = field_array[j, f]
                found = True
                break

        if not found:
            # If a user asks for an unmapped state, we return NaNs
            # to clearly indicate the mathematical absence of data.
            for f in range(field_dim):
                result[i, f] = np.nan

    return result


@njit(cache=True, fastmath=True)
def _norm_mapper_kernel(
        field_array: np.ndarray,
        is_mapped_array: np.ndarray
) -> np.ndarray:
    """
    Calculates the L2 norm (magnitude) of every valid vector in the field array.
    """
    num_states = field_array.shape[0]
    field_dim = field_array.shape[1]

    # Pre-allocate a 1D column vector (shape: N, 1) to hold the scalar magnitudes
    norms = np.zeros((num_states, 1), dtype=np.float64)

    for i in range(num_states):
        if is_mapped_array[i]:
            # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
            sum_sq = 0.0
            for j in range(field_dim):
                sum_sq += field_array[i, j] ** 2
            norms[i, 0] = np.sqrt(sum_sq)

    return norms


@njit(cache=True, fastmath=True)
def _add_mappers_kernel(
        f1: np.ndarray, m1: np.ndarray,
        f2: np.ndarray, m2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = f1.shape[0]
    field_dim = f1.shape[1]

    new_fields = np.zeros_like(f1)
    new_mapped = np.zeros_like(m1)

    for i in range(num_rows):
        if m1[i] and m2[i]:
            for j in range(field_dim):
                new_fields[i, j] = f1[i, j] + f2[i, j]
            new_mapped[i] = True
        elif m1[i]:
            for j in range(field_dim):
                new_fields[i, j] = f1[i, j]
            new_mapped[i] = True
        elif m2[i]:
            for j in range(field_dim):
                new_fields[i, j] = f2[i, j]
            new_mapped[i] = True

    return new_fields, new_mapped


@njit(cache=True, fastmath=True)
def _multiply_mappers_kernel(
        f1: np.ndarray, m1: np.ndarray,
        f2: np.ndarray, m2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = f1.shape[0]
    field_dim = f1.shape[1]

    new_fields = np.zeros_like(f1)
    new_mapped = np.zeros_like(m1)

    for i in range(num_rows):
        if m1[i] and m2[i]:
            for j in range(field_dim):
                new_fields[i, j] = f1[i, j] * f2[i, j]
            new_mapped[i] = True
        elif m1[i]:
            for j in range(field_dim):
                new_fields[i, j] = f1[i, j]  # f2 is unmapped (identity is 1)
            new_mapped[i] = True
        elif m2[i]:
            for j in range(field_dim):
                new_fields[i, j] = f2[i, j]  # f1 is unmapped (identity is 1)
            new_mapped[i] = True

    return new_fields, new_mapped


@njit(cache=True, fastmath=True)
def _add_mappers_inplace_kernel(
        t_f: np.ndarray, t_m: np.ndarray,
        s_f: np.ndarray, s_m: np.ndarray
) -> None:
    num_rows = t_f.shape[0]
    field_dim = t_f.shape[1]

    for i in range(num_rows):
        if s_m[i]:
            if t_m[i]:
                for j in range(field_dim):
                    t_f[i, j] += s_f[i, j]
            else:
                for j in range(field_dim):
                    t_f[i, j] = s_f[i, j]
                t_m[i] = True


@njit(cache=True, fastmath=True)
def _multiply_mappers_inplace_kernel(
        t_f: np.ndarray, t_m: np.ndarray,
        s_f: np.ndarray, s_m: np.ndarray
) -> None:
    num_rows = t_f.shape[0]
    field_dim = t_f.shape[1]

    for i in range(num_rows):
        if s_m[i]:
            if t_m[i]:
                for j in range(field_dim):
                    t_f[i, j] *= s_f[i, j]
            else:
                for j in range(field_dim):
                    t_f[i, j] = s_f[i, j]
                t_m[i] = True


@njit(cache=True, fastmath=True)
def _norm_mapper_inplace_kernel(
        f: np.ndarray, m: np.ndarray
) -> None:
    num_rows = f.shape[0]
    field_dim = f.shape[1]

    for i in range(num_rows):
        if m[i]:
            sum_sq = 0.0
            for j in range(field_dim):
                sum_sq += f[i, j] ** 2

            # Write magnitude to column 0, zero out the rest
            f[i, 0] = np.sqrt(sum_sq)
            for j in range(1, field_dim):
                f[i, j] = 0.0


@njit(cache=True, fastmath=True)
def _jit_normalize_field_in_place(
        field_array: np.ndarray,
        is_mapped_array: np.ndarray,
        normalized_field_array: np.ndarray
) -> None:
    """
    JIT-compiled core logic for field PDF normalization.
    Executes entirely in C-speed without allocating new arrays.
    Delegates to _norm_vector_kernel to respect the generic field algebra.
    """
    capacity = field_array.shape[0]
    field_dim = field_array.shape[1]

    total_norm = 0.0

    # 1. Sequentially find the norm by calling the generalized utility kernel
    for i in range(capacity):
        if is_mapped_array[i]:
            # Slice the 1D field vector for this state and calculate its norm
            total_norm += _norm_vector_kernel(field_array[i])

    # 2. Divide the whole array with this sum serially
    if total_norm > 0:
        for i in range(capacity):
            if is_mapped_array[i]:
                for j in range(field_dim):
                    normalized_field_array[i, j] = field_array[i, j] / total_norm
            else:
                # Ensure unmapped states remain zeroed in the normalized array
                for j in range(field_dim):
                    normalized_field_array[i, j] = 0.0
    else:
        # Fallback to prevent division by zero if the field is completely empty
        for i in range(capacity):
            for j in range(field_dim):
                normalized_field_array[i, j] = 0.0


# ==========================================
# RAW DATA BRIDGE KERNELS (Static Pipeline)
# ==========================================

@njit(cache=True, fastmath=True)
def _batch_add_indexed_kernel(t_f: np.ndarray, s_f: np.ndarray, t_ids: np.ndarray, s_ids: np.ndarray) -> None:
    """O(1) Addition using pre-computed graph indices."""
    num_elements = len(t_ids)
    field_dim = t_f.shape[1]
    for i in range(num_elements):
        t_idx = t_ids[i]
        s_idx = s_ids[i]
        for j in range(field_dim):
            t_f[t_idx, j] += s_f[s_idx, j]


@njit(cache=True, fastmath=True)
def _batch_add_coord_kernel(t_s: np.ndarray, t_f: np.ndarray, s_s: np.ndarray, s_f: np.ndarray) -> None:
    """Fallback O(N*M) spatial coordinate matching."""
    num_s = s_s.shape[0]
    num_t = t_s.shape[0]
    state_dim = s_s.shape[1]
    field_dim = s_f.shape[1]

    for i in range(num_s):
        for j in range(num_t):
            match = True
            for d in range(state_dim):
                if t_s[j, d] != s_s[i, d]:
                    match = False
                    break
            if match:
                for f in range(field_dim):
                    t_f[j, f] += s_f[i, f]
                break


@njit(cache=True, fastmath=True)
def _batch_multiply_indexed_kernel(t_f: np.ndarray, s_f: np.ndarray, t_ids: np.ndarray, s_ids: np.ndarray) -> None:
    num_elements = len(t_ids)
    field_dim = t_f.shape[1]
    for i in range(num_elements):
        t_idx = t_ids[i]
        s_idx = s_ids[i]
        for j in range(field_dim):
            t_f[t_idx, j] *= s_f[s_idx, j]


@njit(cache=True, fastmath=True)
def _batch_multiply_coord_kernel(t_s: np.ndarray, t_f: np.ndarray, s_s: np.ndarray, s_f: np.ndarray) -> None:
    num_s = s_s.shape[0]
    num_t = t_s.shape[0]
    state_dim = s_s.shape[1]
    field_dim = s_f.shape[1]

    for i in range(num_s):
        for j in range(num_t):
            match = True
            for d in range(state_dim):
                if t_s[j, d] != s_s[i, d]:
                    match = False
                    break
            if match:
                for f in range(field_dim):
                    t_f[j, f] *= s_f[i, f]
                break


@njit(cache=True, fastmath=True)
def _batch_norm_bridge_kernel(fields: np.ndarray, out_norms: np.ndarray) -> None:
    num_rows = fields.shape[0]
    for i in range(num_rows):
        out_norms[i, 0] = _norm_vector_kernel(fields[i])


@njit(cache=True, fastmath=True)
def _batch_normalize_bridge_inplace_kernel(fields: np.ndarray) -> None:
    num_rows = fields.shape[0]
    field_dim = fields.shape[1]

    total_norm = 0.0
    for i in range(num_rows):
        total_norm += _norm_vector_kernel(fields[i])

    if total_norm > 0:
        for i in range(num_rows):
            for j in range(field_dim):
                fields[i, j] = fields[i, j] / total_norm
    else:
        for i in range(num_rows):
            for j in range(field_dim):
                fields[i, j] = 0.0

class NumbaKernelFieldUtility(IFieldKernelUtility):

    @staticmethod
    def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return _add_vectors_kernel(v1, v2)

    @staticmethod
    def multiply_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return _multiply_vectors_kernel(v1, v2)

    @staticmethod
    def norm_vector(v: np.ndarray) -> Any:
        return _norm_vector_kernel(v)

    @staticmethod
    def get_fields(ref: FieldKernelFastRef, target_indices: np.ndarray) -> np.ndarray:
        """
        O(1) Data Retrieval.
        Uses advanced NumPy indexing to extract fields instantly.
        """
        # Create a boolean mask of which indices are valid (not -1 and mapped)
        valid_mask = (target_indices >= 0) & ref.is_mapped_array[target_indices]

        # Pre-allocate output with NaNs (shape: [num_targets, field_dim])
        field_dim = ref.field_array.shape[1]
        result = np.full((len(target_indices), field_dim), np.nan, dtype=np.float64)

        # O(1) C-level memory copy for valid rows
        valid_indices = target_indices[valid_mask]
        result[valid_mask] = ref.field_array[valid_indices]

        return result

    @staticmethod
    def norm_mapper(ref: 'FieldKernelFastRef') -> 'FieldKernelFastRef':
        """
        Returns a new FastRef containing the scalar magnitudes of the fields.
        The new field_array will have shape (N, 1).
        """
        new_field_array = _norm_mapper_kernel(ref.field_array, ref.is_mapped_array)

        # Return a new struct.
        # State and Mapped arrays are passed by reference (zero overhead).
        # We drop the field_function because the resulting scalar field is fully materialized.
        return FieldKernelFastRef(
            state_array=ref.state_array,
            field_array=new_field_array,
            is_mapped_array=ref.is_mapped_array,
            field_function=None
        )

    @staticmethod
    def add_mappers(ref1: 'FieldKernelFastRef', ref2: 'FieldKernelFastRef') -> 'FieldKernelFastRef':
        new_f, new_m = _add_mappers_kernel(
            ref1.field_array, ref1.is_mapped_array,
            ref2.field_array, ref2.is_mapped_array
        )
        return FieldKernelFastRef(
            state_array=ref1.state_array,  # Share the state topology
            field_array=new_f,
            is_mapped_array=new_m,
            field_function=None
        )

    @staticmethod
    def multiply_mappers(ref1: 'FieldKernelFastRef', ref2: 'FieldKernelFastRef') -> 'FieldKernelFastRef':
        new_f, new_m = _multiply_mappers_kernel(
            ref1.field_array, ref1.is_mapped_array,
            ref2.field_array, ref2.is_mapped_array
        )
        return FieldKernelFastRef(
            state_array=ref1.state_array,
            field_array=new_f,
            is_mapped_array=new_m,
            field_function=None
        )

    # (Note: norm_mapper was implemented in the previous step)

    # ==========================================
    # 3. BULK MATH - IN-PLACE (Zero Allocation)
    # ==========================================

    @staticmethod
    def add_mappers_inplace(target_ref: 'FieldKernelFastRef', source_ref: 'FieldKernelFastRef') -> None:
        _add_mappers_inplace_kernel(
            target_ref.field_array, target_ref.is_mapped_array,
            source_ref.field_array, source_ref.is_mapped_array
        )

    @staticmethod
    def multiply_mappers_inplace(target_ref: 'FieldKernelFastRef', source_ref: 'FieldKernelFastRef') -> None:
        _multiply_mappers_inplace_kernel(
            target_ref.field_array, target_ref.is_mapped_array,
            source_ref.field_array, source_ref.is_mapped_array
        )

    @staticmethod
    def norm_mapper_inplace(target_ref: 'FieldKernelFastRef') -> None:
        """
        Overwrites the target reference with its magnitude.
        Pads columns [1:] with 0.0 to safely maintain the C-array shape.
        """
        _norm_mapper_inplace_kernel(
            target_ref.field_array,
            target_ref.is_mapped_array
        )

    @staticmethod
    def normalize_field(fast_ref: FieldKernelFastRef) -> None:
        """
        Executes the in-place PDF normalization using the predefined field algebra.
        Reads from fast_ref.field_array and writes directly to fast_ref.normalized_field_array.
        """
        _jit_normalize_field_in_place(
            fast_ref.field_array,
            fast_ref.is_mapped_array,
            fast_ref.normalized_field_array
        )
# ==========================================
    # 5. RAW DATA BRIDGE METHODS (The Static Pipeline)
    # ==========================================

    @staticmethod
    def batch_add_bridge_inplace(
        target_states: np.ndarray,
        target_fields: np.ndarray,
        source_states: np.ndarray,
        source_fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        target_ids = kwargs.get('target_ids')
        source_ids = kwargs.get('source_ids')

        # Fast Path: Graph indices are provided, bypassing spatial search
        if target_ids is not None and source_ids is not None:
            _batch_add_indexed_kernel(target_fields, source_fields, target_ids, source_ids)
        # Slow Path: Fallback to spatial matching
        else:
            _batch_add_coord_kernel(target_states, target_fields, source_states, source_fields)

    @staticmethod
    def batch_multiply_bridge_inplace(
        target_states: np.ndarray,
        target_fields: np.ndarray,
        source_states: np.ndarray,
        source_fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        target_ids = kwargs.get('target_ids')
        source_ids = kwargs.get('source_ids')

        if target_ids is not None and source_ids is not None:
            _batch_multiply_indexed_kernel(target_fields, source_fields, target_ids, source_ids)
        else:
            _batch_multiply_coord_kernel(target_states, target_fields, source_states, source_fields)

    @staticmethod
    def batch_norm_bridge(
        states: np.ndarray,
        fields: np.ndarray,
        out_norms: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Fills the pre-allocated out_norms array with scalar magnitudes.
        """
        _batch_norm_bridge_kernel(fields, out_norms)

    @staticmethod
    def batch_normalize_bridge_inplace(
        states: np.ndarray,
        fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Calculates the Z-factor over the raw batch and normalizes in-place.
        """
        _batch_normalize_bridge_inplace_kernel(fields)

    @staticmethod
    def get_add_kernel() -> Any:
        return _add_vectors_kernel

    @staticmethod
    def get_multiply_kernel() -> Any:
        return _multiply_vectors_kernel
    @staticmethod
    def get_norm_kernel() -> Any:
        return _norm_vector_kernel
