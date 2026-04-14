import numpy as np
from typing import Any

from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebra


# Assuming IFieldAlgebra is imported from your interfaces
# from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebra

class FieldAlgebra(IFieldAlgebra):
    """
    DATA: Concrete implementation of the Inner Product Vector Space (F).
    Defines the boundaries, identities, and mathematical operations of the field.
    """

    def __init__(self, dimensions: int, dtype: Any = np.float64):
        self._dimensions = dimensions
        self._dtype = dtype

        # Pre-allocate the identities so we aren't recreating them on every call
        self._null_vector = np.zeros(dimensions, dtype=dtype)
        self._unity_vector = np.ones(dimensions, dtype=dtype)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def null_vector(self) -> np.ndarray:
        """The Additive Identity (0)."""
        return self._null_vector

    @property
    def unity_vector(self) -> np.ndarray:
        """The Multiplicative Identity (1)."""
        return self._unity_vector

    # ==========================================
    # MATHEMATICAL OPERATIONS (Decoupled Logic)
    # ==========================================

    def add(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Standard vector addition.
        """
        return v1 + v2

    def multiply(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Standard vector multiplication (Hadamard / element-wise product).
        """
        return v1 * v2

    def norm(self, v: np.ndarray) -> Any:
        """
        Calculates the L2 Norm (magnitude) of the vector.
        Returns a scalar value.
        """
        return np.linalg.norm(v)