from abc import ABC, abstractmethod
import numpy as np
from typing import Any

from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelFastRef

class IFieldKernelUtility(IKernelUtility):
    """
    CONTRACT: Highly optimized mathematical operations for the Field Kernel.
    Strictly stateless. Operates on either FastRefs (for standard CM routing)
    or raw numpy arrays (for the Static CM bridge pathway).
    """

    # ==========================================
    # 1. VECTOR MATH (Operates on single 1D field vectors)
    # ==========================================
    @staticmethod
    @abstractmethod
    def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def multiply_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def norm_vector(v: np.ndarray) -> Any:
        pass

    # ==========================================
    # 2. BULK MATH - ALLOCATING (Creates new FastRef)
    # ==========================================
    @staticmethod
    @abstractmethod
    def add_mappers(ref1: FieldKernelFastRef, ref2: FieldKernelFastRef) -> FieldKernelFastRef:
        """Returns a brand new FastRef containing the sum."""
        pass

    @staticmethod
    @abstractmethod
    def multiply_mappers(ref1: FieldKernelFastRef, ref2: FieldKernelFastRef) -> FieldKernelFastRef:
        pass

    @staticmethod
    @abstractmethod
    def norm_mapper(ref: FieldKernelFastRef) -> FieldKernelFastRef:
        pass

    # ==========================================
    # 3. BULK MATH - IN-PLACE (Zero Allocation)
    # target_ref is mutated directly.
    # ==========================================
    @staticmethod
    @abstractmethod
    def add_mappers_inplace(target_ref: FieldKernelFastRef, source_ref: FieldKernelFastRef) -> None:
        """
        Adds source_ref into target_ref.
        Operation: target_ref.field_array += source_ref.field_array
        """
        pass

    @staticmethod
    @abstractmethod
    def multiply_mappers_inplace(target_ref: FieldKernelFastRef, source_ref: FieldKernelFastRef) -> None:
        """
        Multiplies source_ref into target_ref.
        Operation: target_ref.field_array *= source_ref.field_array
        """
        pass

    @staticmethod
    @abstractmethod
    def norm_mapper_inplace(target_ref: FieldKernelFastRef) -> None:
        """
        Calculates the norm of target_ref and overwrites its own data.
        Note: If the field was 3D, it will safely overwrite with 1D scalars padded or
        restructured according to the specific hardware implementation.
        """
        pass

    # ==========================================
    # 4. DATA RETRIEVAL
    # ==========================================
    @staticmethod
    @abstractmethod
    def get_fields(ref: FieldKernelFastRef, target_states: np.ndarray) -> np.ndarray:
        """
        Retrieves field vectors for a given array of state vectors.
        Hardware implementations (like Numba) will use optimized internal mapping
        to find the target_states within ref.state_array and return the corresponding ref.field_array rows.
        """
        pass

    @staticmethod
    @abstractmethod
    def normalize_field(fast_ref: FieldKernelFastRef) -> None:
        """
        Executes the in-place PDF normalization.
        1. Sequentially calculates the norm for each valid state from fast_ref.field_array.
        2. Sums these norms.
        3. Divides the field values by the sum and stores in fast_ref.normalized_field_array.
        """
        pass

    # ==========================================
    # 5. RAW DATA BRIDGE METHODS (The Static Pipeline)
    # Operates strictly on raw iterables/arrays. Zero OOP overhead.
    # ==========================================
    @staticmethod
    @abstractmethod
    def batch_add_bridge_inplace(
        target_states: np.ndarray,
        target_fields: np.ndarray,
        source_states: np.ndarray,
        source_fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Adds source fields into target fields using raw arrays.
        kwargs can include 'target_ids' and 'source_ids' for O(1) matching.
        Mutates target_fields directly.
        """
        pass

    @staticmethod
    @abstractmethod
    def batch_multiply_bridge_inplace(
        target_states: np.ndarray,
        target_fields: np.ndarray,
        source_states: np.ndarray,
        source_fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Multiplies source fields into target fields using raw arrays.
        Mutates target_fields directly.
        """
        pass

    @staticmethod
    @abstractmethod
    def batch_norm_bridge(
        states: np.ndarray,
        fields: np.ndarray,
        out_norms: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Calculates the norm (magnitude) over a raw batch of fields.
        Because this changes the geometric shape (e.g., from N,3 to N,1),
        the orchestrator MUST provide a pre-allocated 'out_norms' array.
        Returns None to enforce the zero-allocation rule.
        """
        pass

    @staticmethod
    @abstractmethod
    def batch_normalize_bridge_inplace(
        states: np.ndarray,
        fields: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Executes in-place PDF normalization over a raw batch of fields.
        Assumes scalar division (fields /= Z) which preserves array shape.
        Mutates the 'fields' array directly to conserve memory.
        """
        pass

# ==========================================
    @staticmethod
    @abstractmethod
    def get_add_kernel() -> Any:
        """Returns the raw JIT-compiled dispatcher for vector addition."""
        pass

    @staticmethod
    @abstractmethod
    def get_multiply_kernel() -> Any:
        """Returns the raw JIT-compiled dispatcher for vector multiplication."""
        pass

    @staticmethod
    @abstractmethod
    def get_norm_kernel() -> Any:
        """Returns the raw JIT-compiled dispatcher for array normalization."""
        pass