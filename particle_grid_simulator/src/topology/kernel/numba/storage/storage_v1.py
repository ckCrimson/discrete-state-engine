from dataclasses import dataclass
from typing import Any, Callable, Type
import numpy as np
import numba
from numba import types
from numba.typed import List, Dict as NumbaDict

from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract
from particle_grid_simulator.src.topology.interfaces.storage import ITopologyFastRef

@dataclass
class TopologyFastRef(ITopologyFastRef):
    """The raw memory payload passed directly into the Numba @njit kernel."""

    # Identity Maps (Sparse Set approach)
    handle_map: List         # List of np.ndarray vectors (Dense Array)
    visited_map: NumbaDict   # Map: Tuple(coords) -> Integer Handle Index (Hashmap)

    # Forward Graph (Dynamic CSR)
    forward_starts: np.ndarray
    forward_counts: np.ndarray
    forward_edges: List      # Append-only integer list

    # Reverse Graph (Dynamic CSR)
    reverse_starts: np.ndarray
    reverse_counts: np.ndarray
    reverse_edges: List      # Append-only integer list

    # Query Buffers
    buffer_a: List
    buffer_b: List
    seen_array: np.ndarray

    # Meta
    steps_prepared: int


from particle_grid_simulator.src.topology.interfaces.storage import ITopologyStorage

class TopologyKernelDataContract(KernelDataContract):
    def __init__(
        self,
        neighbour_function: Callable,
        state_class_reference: Type,
        initial_capacity: int = 100000,
        dimensions: int = 1,
        vector_dtype: Any = np.int32,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.neighbour_function = neighbour_function
        self.state_class_reference = state_class_reference
        self.initial_capacity = initial_capacity
        self.dimensions = dimensions
        self.vector_dtype = vector_dtype

class NumbaTopologyStorage(ITopologyStorage):
    def __init__(self, contract: TopologyKernelDataContract) -> None:
        self.contract = contract
        self._fast_ref = self._allocate_memory(
            capacity=self.contract.initial_capacity,
            dimensions=self.contract.dimensions,
            vector_dtype=self.contract.vector_dtype
        )

    @staticmethod
    def _allocate_memory(capacity: int, dimensions: int, vector_dtype: Any) -> TopologyFastRef:
        vec_type = numba.typeof(np.zeros(1, dtype=vector_dtype))

        # FIX 1: Dynamically map the tuple type to match the provided vector_dtype (float64)
        numba_scalar_type = numba.from_dtype(np.dtype(vector_dtype))
        tuple_key_type = types.UniTuple(numba_scalar_type, count=dimensions)

        return TopologyFastRef(
            handle_map=List.empty_list(vec_type),

            # FIX 2: value_type MUST be int32 (it stores the graph index, not the coordinate!)
            visited_map=NumbaDict.empty(key_type=tuple_key_type, value_type=types.int32),

            forward_starts=np.zeros(capacity, dtype=np.int32),
            forward_counts=np.zeros(capacity, dtype=np.int32),
            forward_edges=List.empty_list(types.int32),
            reverse_starts=np.zeros(capacity, dtype=np.int32),
            reverse_counts=np.zeros(capacity, dtype=np.int32),
            reverse_edges=List.empty_list(types.int32),
            buffer_a=List.empty_list(types.int32),
            buffer_b=List.empty_list(types.int32),
            seen_array=np.zeros(capacity, dtype=np.bool_),
            steps_prepared=0
        )

    # --- THE FIX 1: Return the object directly, NOT a dictionary ---
    def get_fast_ref(self) -> TopologyFastRef:
        return self._fast_ref

    @property
    def fast_refs(self) -> TopologyFastRef:
        return self._fast_ref

    # --- Interface Fulfillments ---
    def get_neighbour_function(self) -> Callable:
        return self.contract.neighbour_function

    def get_state_class_reference(self) -> Type:
        return self.contract.state_class_reference

    def get_number_of_states_prepared(self) -> int:
        return self._fast_ref.steps_prepared

    # --- THE FIX 2: Check size BEFORE clearing the lists ---
    def clear(self) -> None:
        """Zeros out the data without returning memory to the OS."""
        ref = self._fast_ref

        # Get the size BEFORE we wipe the lists
        current_size = len(ref.handle_map)

        # O(1) clears for dynamic lists/dicts
        ref.handle_map.clear()
        ref.visited_map.clear()
        ref.forward_edges.clear()
        ref.reverse_edges.clear()
        ref.buffer_a.clear()
        ref.buffer_b.clear()

        # Zero out only the used portions of the pre-allocated Numpy arrays
        if current_size > 0:
            ref.forward_starts[:current_size] = 0
            ref.forward_counts[:current_size] = 0
            ref.reverse_starts[:current_size] = 0
            ref.reverse_counts[:current_size] = 0
            ref.seen_array[:current_size] = False

        ref.steps_prepared = 0