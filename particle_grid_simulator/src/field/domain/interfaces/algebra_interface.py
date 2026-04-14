from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class IFieldAlgebra(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int: pass

    @property
    @abstractmethod
    def dtype(self) -> Any: pass

    @property
    @abstractmethod
    def null_vector(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def unity_vector(self) -> np.ndarray: pass

    # --- ADD THE MATH OPERATIONS HERE ---
    @abstractmethod
    def add(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def multiply(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def norm(self, v: np.ndarray) -> Any: pass

class IFieldAlgebraUtility(ABC):
    """
    CONTRACT: Binary operations allowed over the Vector Space.
    """
    @staticmethod
    @abstractmethod
    def add(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def multiply(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def inner_product(f1: np.ndarray, f2: np.ndarray) -> Any:
        """
        The Inner Product <f1, f2>.
        Returns a scalar (type depends on the Algebra's dtype).
        """
        pass

    @staticmethod
    @abstractmethod
    def norm(f: np.ndarray) -> Any:
        """
        The Norm ||f|| (Magnitude of the vector).
        Returns a positive scalar (type depends on the Algebra's dtype).
        """
        pass