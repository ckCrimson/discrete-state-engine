import numpy as np
from typing import Any
from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebraUtility


class FieldAlgebraUtility(IFieldAlgebraUtility):
    """
    UTILITY: Concrete binary operations over the Vector Space.
    Executes raw NumPy C-level operations.
    """

    @staticmethod
    def add(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """Vector addition."""
        return f1 + f2

    @staticmethod
    def multiply(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """
        Element-wise multiplication (Hadamard product).
        This acts as the standard scaling operation in our Field Algebra.
        """
        return f1 * f2

    @staticmethod
    def inner_product(f1: np.ndarray, f2: np.ndarray) -> Any:
        """
        The Inner Product <f1, f2>.
        Returns a numpy scalar matching the Algebra's dtype.
        """
        return np.dot(f1, f2)

    @staticmethod
    def norm(f: np.ndarray) -> Any:
        """
        The L2 Norm ||f|| (Magnitude).
        Returns a positive numpy scalar matching the Algebra's dtype.
        """
        # np.linalg.norm naturally handles the square root of the sum of squares
        return np.linalg.norm(f)