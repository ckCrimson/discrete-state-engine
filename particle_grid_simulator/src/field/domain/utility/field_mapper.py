import numpy as np
from typing import Any

from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.field.domain.interfaces.algebra_interface import IFieldAlgebraUtility
from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapperUtility, IFieldMapper


class FieldMapperUtility(IFieldMapperUtility):
    """
    UTILITY: Concrete implementation for composing and transforming Field Mappers.
    Perfectly decoupled: Relies entirely on the IFieldAlgebra for mathematical operations.
    """

    @staticmethod
    def _validate_compatibility(m1: IFieldMapper, m2: IFieldMapper) -> None:
        if m1.state_class_ref != m2.state_class_ref:
            raise TypeError("Compatibility Error: Mappers operate on different State Spaces.")
        if m1.algebra.dimensions != m2.algebra.dimensions:
            raise ValueError(f"Compatibility Error: Dimensionality mismatch.")
        if m1.algebra.dtype != m2.algebra.dtype:
            raise TypeError("Compatibility Error: Field Algebras have different data types.")

    @staticmethod
    def add_mappers(m1: IFieldMapper, m2: IFieldMapper) -> IFieldMapper:
        FieldMapperUtility._validate_compatibility(m1, m2)

        def composed_add(state_vec: np.ndarray) -> np.ndarray:
            v1 = m1.get_field_vector(state_vec)
            v2 = m2.get_field_vector(state_vec)

            val1 = v1 if v1 is not None else m1.algebra.null_vector
            val2 = v2 if v2 is not None else m2.algebra.null_vector

            # DECOUPLED: The algebra handles its own addition
            return m1.algebra.add(val1, val2)

        return FieldMapper(
            algebra=m1.algebra,
            state_class_ref=m1.state_class_ref,
            mapper_func=composed_add
        )

    @staticmethod
    def multiply_mappers(m1: IFieldMapper, m2: IFieldMapper) -> IFieldMapper:
        FieldMapperUtility._validate_compatibility(m1, m2)

        def composed_mult(state_vec: np.ndarray) -> np.ndarray:
            v1 = m1.get_field_vector(state_vec)
            v2 = m2.get_field_vector(state_vec)

            val1 = v1 if v1 is not None else m1.algebra.unity_vector
            val2 = v2 if v2 is not None else m2.algebra.unity_vector

            # DECOUPLED: The algebra handles its own multiplication
            return m1.algebra.multiply(val1, val2)

        return FieldMapper(
            algebra=m1.algebra,
            state_class_ref=m1.state_class_ref,
            mapper_func=composed_mult
        )

    @staticmethod
    def norm(m: IFieldMapper) -> IFieldMapper:
        # Create a new 1D scalar algebra matching the source dtype
        # (Assuming your domain allows instantiating a base scalar algebra here)
        norm_algebra = FieldAlgebra(dimensions=1, dtype=m.algebra.dtype)

        def composed_norm(state_vec: np.ndarray) -> np.ndarray:
            v = m.get_field_vector(state_vec)
            if v is None:
                return norm_algebra.null_vector

            # DECOUPLED: The source algebra calculates its own norm
            scalar_norm = m.algebra.norm(v)
            return np.array([scalar_norm], dtype=norm_algebra.dtype)

        return FieldMapper(
            algebra=norm_algebra,
            state_class_ref=m.state_class_ref,
            mapper_func=composed_norm
        )