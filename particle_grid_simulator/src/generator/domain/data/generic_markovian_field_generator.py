from typing import Callable, Type, Tuple, Any
import numpy as np

from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility


# Assuming imports for IFieldGeneratorData, Topology, TopologyUtility, IFieldMapper, State
# from particle_grid_simulator.src.field.domain.interfaces.generator_interface import IFieldGeneratorData
# from particle_grid_simulator.src.topology.domain.topology_domain import Topology, TopologyUtility
# from particle_grid_simulator.src.state.domain import State

class GenericMarkovianFieldGeneratorData:  # Implements IFieldGeneratorData
    """
    DATA: The immutable configuration and fast-reference blueprint for the Generator.
    Pre-extracts algebra pointers and eagerly bakes graph edges to guarantee
    O(1) topology lookups during the recurrence relation loops.
    """

    def __init__(
            self,
            mapper: 'IFieldMapper',
            topology: 'Topology',
            transition_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
            maximum_step_baking: int,
            max_size: int,
            state_shape: Tuple[int, ...],
            implicit_norm: bool = False,
            explicit_norm: bool = True
    ):
        self._mapper_class_ref = type(mapper)
        self._state_class_ref = mapper.state_class_ref

        self._implicit_norm = implicit_norm
        self._explicit_norm = explicit_norm

        self._max_size = max_size
        self._state_shape = state_shape
        self._field_vector_shape = (mapper.algebra.dimensions,)

        self._topology = topology
        self._transition_function = transition_function
        self._maximum_step_baking = maximum_step_baking

        # ==========================================
        # 1. ALGEBRA POINTER EXTRACTION
        # Eliminates multi-level pointer chasing during the simulation loop.
        # ==========================================
        self._algebra_add = mapper.algebra.add
        self._algebra_multiply = mapper.algebra.multiply
        self._algebra_norm = mapper.algebra.norm
        self._algebra_null_vector = mapper.algebra.null_vector
        self._algebra_unity_vector = mapper.algebra.unity_vector

        # ==========================================
        # 2. TOPOLOGY BAKING (Eager Cache Population)
        # ==========================================
        self._bake_topology(mapper)

    def _bake_topology(self, mapper: 'IFieldMapper') -> None:
        """
        Eagerly traverses the graph up to 'maximum_step_baking' starting from
        all currently active states in the mapper. This forces the TopologyUtility
        to populate the adjacency_cache, ensuring O(1) lookups later.
        """
        if self._maximum_step_baking <= 0 or not self._topology.use_cache:
            return

        # 1. FIX: Unpack the tuple to isolate the list of state vectors
        raw_states, _ = mapper.get_raw_data()

        # 2. Reconstruct active states from the raw numpy arrays
        active_states = set(
            self._state_class_ref(vec) for vec in raw_states
        )

        seen = set(active_states)
        current_frontier = set(active_states)

        for _ in range(self._maximum_step_baking):
            next_frontier = set()
            for current_state in current_frontier:
                # This call intrinsically caches the result inside the topology object
                neighbors = TopologyUtility._get_cached_reachable(self._topology, current_state)

                for next_state in neighbors:
                    if next_state not in seen:
                        next_frontier.add(next_state)
                        seen.add(next_state)

            current_frontier = next_frontier

            if not current_frontier:
                break  # Grid exhausted early

    # ==========================================
    # IFieldGeneratorData PROTOCOL IMPLEMENTATION
    # ==========================================

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self._state_shape

    @property
    def field_vector_shape(self) -> Tuple[int, ...]:
        return self._field_vector_shape

    @property
    def state_class_ref(self) -> Type:
        return self._state_class_ref

    @property
    def mapper_class_ref(self) -> Type:
        return self._mapper_class_ref

    @property
    def transition_function(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._transition_function

    @property
    def algebra_add(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._algebra_add

    @property
    def algebra_multiply(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._algebra_multiply

    @property
    def algebra_norm(self) -> Callable[[np.ndarray], Any]:
        return self._algebra_norm

    @property
    def algebra_null_vector(self) -> np.ndarray:
        return self._algebra_null_vector

    @property
    def algebra_unity_vector(self) -> np.ndarray:
        return self._algebra_unity_vector

    @property
    def implicit_norm(self) -> bool:
        return self._implicit_norm

    @property
    def explicit_norm(self) -> bool:
        return self._explicit_norm

    # ==========================================
    # DOMAIN SPECIFIC PROPERTIES
    # ==========================================

    @property
    def topology(self) -> 'Topology':
        return self._topology

    @property
    def maximum_step_baking(self) -> int:
        return self._maximum_step_baking