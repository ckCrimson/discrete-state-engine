import numpy as np
from typing import Callable, Dict, Tuple, Type, Any, Optional, Iterable, Union

from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebra
from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper


class FieldMapper(IFieldMapper):
    """
    DATA: Concrete implementation of the Field Mapper.
    Maps discrete state coordinates to continuous field vectors.
    Supports lazy evaluation and manual discrete population.
    """

    def __init__(
            self,
            algebra: IFieldAlgebra,
            state_class_ref: Type,
            states: Optional[Union[Iterable[Any], Iterable[np.ndarray], Any, np.ndarray]] = None,
            field_vectors: Optional[Union[Iterable[np.ndarray], np.ndarray]] = None,
            mapper_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):        # Private backing fields for the interface properties
        self._algebra = algebra
        self._state_class_ref = state_class_ref

        self.mapper_func = mapper_func

        # Fast Access Mapping: tuple(state_vec) -> np.ndarray(field_vec)
        self.mapping_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

        # The known State Space (S) intersecting with this field
        self.state_space_vectors: list[np.ndarray] = []

        # If initial discrete data is provided, populate the field immediately
        if states is not None and field_vectors is not None:
            self.set_fields_at(states, field_vectors)

    @property
    def algebra(self) -> IFieldAlgebra:
        return self._algebra

    @property
    def mapper_func(self) -> Type:
        return self._mapper_func


    @property
    def state_class_ref(self) -> Type:
        return self._state_class_ref

    def set_fields_at(
            self,
            states: Union[Iterable[Any], Iterable[np.ndarray], Any, np.ndarray],
            fields: Union[Iterable[np.ndarray], np.ndarray]
    ) -> None:
        """
        Populates the field at specific states.
        Automatically strips OOP wrappers (like State objects) to store pure NumPy arrays.
        Supports broadcasting a single field vector to multiple states.
        """
        # 1. Parse States into a list of raw 1D numpy arrays
        raw_states = []
        if isinstance(states, self._state_class_ref):
            raw_states = [states.vector]
        elif isinstance(states, np.ndarray) and states.ndim == 1:
            raw_states = [states]
        elif isinstance(states, Iterable):
            for s in states:
                if isinstance(s, self._state_class_ref):
                    raw_states.append(s.vector)
                elif isinstance(s, np.ndarray):
                    raw_states.append(s)
                else:
                    raise TypeError(f"Unsupported state type in iterable: {type(s)}")
        else:
            raise TypeError("Unsupported states format.")

        # 2. Parse Fields into a list of raw 1D numpy arrays
        raw_fields = []
        if isinstance(fields, np.ndarray):
            if fields.ndim == 1:
                # Broadcast single vector to all provided states
                raw_fields = [fields] * len(raw_states)
            elif fields.ndim == 2:
                # Iterable of vectors
                raw_fields = list(fields)
        elif isinstance(fields, Iterable):
            raw_fields = list(fields)
        else:
            raise TypeError("Unsupported fields format.")

        # Validation
        if len(raw_fields) != len(raw_states):
            raise ValueError(
                f"Mismatch: Provided {len(raw_states)} states but {len(raw_fields)} field vectors."
            )

        # 3. Populate the mathematical cache
        for s_vec, f_vec in zip(raw_states, raw_fields):
            vec_tuple = tuple(s_vec.tolist())

            # Only append to state_space_vectors if it's a completely new state
            if vec_tuple not in self.mapping_cache:
                self.state_space_vectors.append(s_vec)

            self.mapping_cache[vec_tuple] = f_vec

    def get_field_vector(self, state_vec: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieves the field vector.
        If unmapped, attempts lazy evaluation via mapper_func.
        If no function exists, returns None (mathematically undefined/vacuum).
        """
        vec_tuple = tuple(state_vec.tolist())

        # 1. Check discrete cache
        if vec_tuple in self.mapping_cache:
            return self.mapping_cache[vec_tuple]

        # 2. Lazy evaluation (if a functional rule was provided)
        if self.mapper_func is not None:
            f_vec = self.mapper_func(state_vec)
            self.mapping_cache[vec_tuple] = f_vec
            self.state_space_vectors.append(state_vec)
            return f_vec

        # 3. The Unmapped Vacuum
        # This state is not in the cache and we have no function to compute it.
        return None

    def get_raw_data(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        states = []
        fields = []
        # self.mapping_cache is Dict[Tuple, np.ndarray]
        for state_tuple, field_vec in self.mapping_cache.items():
            states.append(np.array(state_tuple, dtype=self.algebra.dtype))
            fields.append(field_vec)
        return states, fields

    @mapper_func.setter
    def mapper_func(self, value):
        self._mapper_func = value