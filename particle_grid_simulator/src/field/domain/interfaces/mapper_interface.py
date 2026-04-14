from abc import ABC, abstractmethod
from typing import Optional, Union, Iterable, Type, Any, Tuple

import numpy as np

from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebra


class IFieldMapper(ABC):
    """
    CONTRACT: Maps discrete state coordinates to continuous field vectors.
    """
    @property
    @abstractmethod
    def algebra(self) -> IFieldAlgebra:
        pass

    @property
    @abstractmethod
    def state_class_ref(self) -> Type:
        pass

    @property
    @abstractmethod
    def mapper_func(self) -> Type:
        pass

    @abstractmethod
    def set_fields_at(
        self,
        states: Union[Iterable[Any], Iterable[np.ndarray], Any, np.ndarray],
        fields: Union[Iterable[np.ndarray], np.ndarray]
    ) -> None:
        """Injects discrete data into the field cache."""
        pass

    @abstractmethod
    def get_field_vector(self, state_vec: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns the field vector for a coordinate.
        Must return None if the field is mathematically undefined at this coordinate.
        """
        pass

    @abstractmethod
    def get_raw_data(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Extracts the internal cache for hardware baking.
        Returns: (List of state vectors, List of corresponding field vectors)
        """
        pass


class IFieldMapperUtility(ABC):
    """
    CONTRACT: Operations that compose or transform Field Mappers.
    """
    @staticmethod
    @abstractmethod
    def add_mappers(m1: IFieldMapper, m2: IFieldMapper) -> IFieldMapper:
        pass

    @staticmethod
    @abstractmethod
    def multiply_mappers(m1: IFieldMapper, m2: IFieldMapper) -> IFieldMapper:
        pass

    @staticmethod
    @abstractmethod
    def norm(m: IFieldMapper) -> IFieldMapper:
        pass