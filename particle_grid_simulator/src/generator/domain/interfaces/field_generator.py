from typing import Protocol, Callable, Type, Tuple, Any, List
import numpy as np

from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper


# Assuming imports for your domain interfaces
# from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper

class IFieldGeneratorData(Protocol):
    """
    DATA: The configuration, boundaries, and fast-references for the Field Generator.
    Acts as the immutable blueprint for the 'prepare and blast' strategy.
    """

    @property
    def max_size(self) -> int:
        """The maximum expected dimensionality/capacity used to pre-allocate blast arrays."""
        ...

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """The geometric shape of a single state vector."""
        ...

    @property
    def field_vector_shape(self) -> Tuple[int, ...]:
        """The mathematical shape of a single field vector."""
        ...

    @property
    def state_class_ref(self) -> Type:
        """Reference to the concrete State class."""
        ...

    @property
    def mapper_class_ref(self) -> Type:
        """Reference to the concrete FieldMapper class."""
        ...

    @property
    def transition_function(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        T(s_j -> s_i): The raw transition field generator function.
        Takes (source_state_vector, target_state_vector) -> raw_transition_field_vector
        """
        ...

    # ==========================================
    # FAST ALGEBRA REFERENCES (1 Pointer Chase)
    # Extracted directly from the base FieldMapper's Algebra
    # ==========================================

    @property
    def algebra_add(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        ...

    @property
    def algebra_multiply(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        ...

    @property
    def algebra_norm(self) -> Callable[[np.ndarray], Any]:
        ...

    @property
    def algebra_null_vector(self) -> np.ndarray:
        ...

    @property
    def algebra_unity_vector(self) -> np.ndarray:
        ...



# Assuming IFieldGeneratorData and IFieldMapper are imported

class IFieldGeneratorUtility(Protocol):
    """
    UTILITY: The pure mathematical execution of the Generic Markovian Field Generator.
    Operates strictly on raw C-arrays using hoisted functions for maximum throughput.
    """

    def calculate_affected_transition_field(
        self,
        source_state: np.ndarray,
        target_state: np.ndarray,
        global_field_vector: np.ndarray,
        transition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        multiply_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """F_a(s_j -> s_i) = T(s_j -> s_i) ⊗ F_g(s_i)"""
        ...

    def normalize_transition_frontier(
        self,
        affected_fields: List[np.ndarray],
        add_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        norm_func: Callable[[np.ndarray], Any]
    ) -> List[np.ndarray]:
        """Calculates Z_i over the reachable space."""
        ...

    def _generate_next_step(
        self,
        current_states: np.ndarray,
        current_fields: np.ndarray,
        active_count: int,
        target_states_buffer: np.ndarray,
        target_fields_buffer: np.ndarray,
        global_mapper: 'IFieldMapper',
        generator_data: 'IFieldGeneratorData',
        **kwargs: Any
    ) -> int:
        """
        PRIVATE: Executes a single generation step.
        Reads from current arrays and accumulates into the target pre-allocated buffers.
        Returns the number of active states written to the target buffers.
        """
        ...

    def generate_multi_step_field(
        self,
        initial_mapper: 'IFieldMapper',
        global_mapper: 'IFieldMapper',
        generator_data: 'IFieldGeneratorData',
        steps: int,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PUBLIC: Drives the L-step evolution.
        Manages the ping-pong buffer allocation and executes the loop.
        Returns the densely packed, active slices of the final states and fields.
        """
        ...