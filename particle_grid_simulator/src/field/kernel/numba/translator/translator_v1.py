# Numba translator implementation for Field
import numpy as np
from typing import Any, Dict, Tuple, List

from particle_grid_simulator.src.field.component_manager.component_enums import FieldCommandType
from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelFastRef
from particle_grid_simulator.src.field.interfaces.translator import IFieldTranslator


# Assuming interfaces and enums are imported:
# from particle_grid_simulator.src.field.interfaces.translator import IFieldTranslator, FieldCommandType
# from particle_grid_simulator.src.field.kernel.interfaces.storage_interface import FieldKernelFastRef
# from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper




class NumbaFieldTranslator(IFieldTranslator):
    """
    CONCRETE TRANSLATOR: Bridges the Domain and Numba Hardware.
    Maintains the Python-to-Hardware ID mapping to ensure the Numba Kernel
    never has to perform a linear search, and handles incremental buffer flushing.
    """

    def __init__(self) -> None:
        # The bridge: Maps a hashable state coordinate to a hardware row index
        self._state_to_id_map: Dict[Tuple[float, ...], int] = {}

        # Tracks the next available empty row in the pre-allocated arrays
        self._next_available_id: int = 0

        # Cached for CLEAR commands during incremental updates
        self._algebra = None

    def bake(self, fast_refs: Any, initial_data: Any) -> None:
        """
        FULL REBUILD: Extracts OOP data and packs it continuously into C-arrays.
        """
        if initial_data is None:
            return

        mapper: 'IFieldMapper' = initial_data
        ref: 'FieldKernelFastRef' = fast_refs

        # 1. Cache the algebra for later hardware use
        self._algebra = mapper.algebra

        # 2. Extract the flat lists from the Domain Cache
        states_list, fields_list = mapper.get_raw_data()
        num_items = len(states_list)

        if num_items == 0:
            return

        # 3. Convert lists to bulk NumPy arrays
        states_arr = np.array(states_list, dtype=np.float64)
        fields_arr = np.array(fields_list, dtype=np.float64)

        # 4. In-place injection into the hardware vaults (Zero Allocation)
        ref.state_array[:num_items] = states_arr
        ref.field_array[:num_items] = fields_arr
        ref.is_mapped_array[:num_items] = True  # Mark these specific rows as valid

        # 5. Build the ID mapping for O(1) dynamic lookups later
        for i, state_vec in enumerate(states_list):
            state_key = tuple(state_vec.tolist())
            self._state_to_id_map[state_key] = i

        # 6. Advance the hardware pointer
        self._next_available_id = num_items

    def bake_incremental(self, ref: 'FieldKernelFastRef', queue: list, algebra: 'IFieldAlgebra') -> None:
        """
        Flushes the command queue to the hardware arrays using C-speed bulk operations.
        Bypasses the Python GIL for array mutations entirely.
        """
        for cmd_id, cmd_type, payload in queue:
            states, fields = payload

            # 1. Coerce to fast, contiguous C-arrays
            raw_states = np.asarray(states, dtype=np.float64)
            num_states = len(raw_states)

            # 2. C-Speed Tuple Conversion
            state_tuples = list(map(tuple, raw_states.tolist()))

            # 3. Fast ID Mapping (Stateless ID Generation)
            target_indices = np.empty(num_states, dtype=np.int32)

            for i, state_key in enumerate(state_tuples):
                idx = self._state_to_id_map.get(state_key, -1)
                if idx == -1:
                    # FIX: The next ID is exactly the current size of the dictionary.
                    idx = len(self._state_to_id_map)
                    self._state_to_id_map[state_key] = idx
                target_indices[i] = idx

            # 4. BULK HARDWARE WRITES (Zero Python Looping)
            if cmd_type == FieldCommandType.SET.value:
                raw_fields = np.asarray(fields)
                ref.field_array[target_indices] = raw_fields
                ref.is_mapped_array[target_indices] = True

            elif cmd_type == FieldCommandType.ADD.value:
                raw_fields = np.asarray(fields, dtype=np.float64)
                is_mapped = ref.is_mapped_array[target_indices]

                # Mapped States: Bulk Addition
                ref.field_array[target_indices[is_mapped]] += raw_fields[is_mapped]

                # Unmapped States: Bulk Overwrite
                unmapped = ~is_mapped
                ref.field_array[target_indices[unmapped]] = raw_fields[unmapped]
                ref.is_mapped_array[target_indices] = True

            elif cmd_type == FieldCommandType.CLEAR.value:
                ref.field_array[target_indices] = algebra.null_vector

    def sync(self, fast_refs: Any, **kwargs: Any) -> None:
        """
        REVERSE TRANSLATION: Pushes hardware array changes back up to the Domain.
        Slices out only the active (mapped) memory and pushes it in bulk.
        """
        domain_mapper: 'IFieldMapper' = kwargs.get('domain_mapper')
        if not domain_mapper:
            raise ValueError("Sync failed: 'domain_mapper' must be provided in kwargs.")

        ref: 'FieldKernelFastRef' = fast_refs

        # 1. Use the boolean mask to find all rows that contain valid data.
        # We only check up to `_next_available_id` to avoid scanning empty pre-allocated space.
        valid_indices = np.where(ref.is_mapped_array[:self._next_available_id])[0]

        if len(valid_indices) == 0:
            return

        # 2. Slice the arrays (Vectorized, O(1) C-level extraction)
        active_states = ref.state_array[valid_indices]
        active_fields = ref.field_array[valid_indices]

        # 3. Bulk push back to the pure Python Domain Mapper
        domain_mapper.set_fields_at(active_states, active_fields)

    def get_hardware_indices(self, states: Any) -> np.ndarray:
        """
        O(1) lookup bridging Domain coordinates to Hardware row IDs.
        """
        # Normalize to iterable
        if isinstance(states, np.ndarray) and states.ndim == 1:
            states = [states]

        indices = []
        for state in states:
            state_key = tuple(np.array(state).tolist())
            # O(1) Dictionary Lookup
            idx = self._state_to_id_map.get(state_key, -1)
            indices.append(idx)

        return np.array(indices, dtype=np.int32)