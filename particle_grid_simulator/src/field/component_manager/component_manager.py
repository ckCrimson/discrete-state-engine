# The concrete API for the Field Module
from typing import Any, Iterable, Union, Optional, Callable
import numpy as np
from numba import njit

from hpc_ecs_core.src.hpc_ecs_core.manager import BaseComponentManager
from hpc_ecs_core.src.hpc_ecs_core.interfaces import SyncState
from particle_grid_simulator.src.field.component_manager.component_enums import FieldCommandType
from particle_grid_simulator.src.field.domain.interfaces.mapper_interface import IFieldMapper
from particle_grid_simulator.src.field.interfaces.storage import IFieldKernelDataContract, IFieldKernelStorage
from particle_grid_simulator.src.field.interfaces.translator import IFieldTranslator
from particle_grid_simulator.src.field.interfaces.utility import IFieldKernelUtility
# Assuming FieldCommandType is defined in your transformation/interfaces


class FieldComponentManager(BaseComponentManager):
    def __init__(
            self,
            utility: 'IFieldKernelUtility',
            contract: Optional['IFieldKernelDataContract'] = None,
            storage: Optional['IFieldKernelStorage'] = None,
            translator: Optional['IFieldTranslator'] = None,
            domain_mapper: Optional['IFieldMapper'] = None
    ) -> None:

        # The base class cleanly initializes the storage, extracts the FastRef,
        # and calls translator.bake(). No hacks required.
        super().__init__(
            contract=contract,
            raw_storage=storage,
            translator=translator,
            utility=utility,
            initial_data=domain_mapper
        )
        # FIX: Explicitly save what we need for later operations!
        self._algebra = contract.algebra
        self._domain_mapper = domain_mapper
        self._is_normalized: bool = False

    # ==========================================
    # DYNAMIC COMMAND PIPELINE
    # ==========================================

    def _queue_operation(
            self,
            command_type: FieldCommandType,
            states: Union[Iterable[Any], Iterable[np.ndarray], Any, np.ndarray],
            fields: Optional[Union[Iterable[np.ndarray], np.ndarray]] = None
    ) -> int:
        """
        Helper to safely push field updates into the generic command buffer.
        """
        cmd_id = len(self.command_buffer.queue)

        # Pack the raw inputs; the Translator will parse states -> indices
        payload = (states, fields)
        self.command_buffer.add_command(cmd_id, command_type.value, payload)

        # Mark the state as execution dirty, indicating hardware needs an incremental bake
        self.store.sync_state = SyncState.EXECUTION_DIRTY

        return cmd_id

    def set_fields(self, states: Any, fields: Any) -> int:
        """Absolute overwrite operation."""
        return self._queue_operation(FieldCommandType.SET, states, fields)

    def add_fields(self, states: Any, fields: Any) -> int:
        """Accumulation operation (e.g., adding a temporary heat source)."""
        return self._queue_operation(FieldCommandType.ADD, states, fields)

    def clear_fields(self, states: Any) -> int:
        """Reset-to-zero operation for specific states."""
        return self._queue_operation(FieldCommandType.CLEAR, states, None)

    def commit_frame(self) -> None:
        """
        Flushes all queued surgical updates (SET, ADD, CLEAR) into the C-arrays.
        """
        queue = self.command_buffer.queue
        if queue:
            # FIX: Use the safely stored algebra from the contract!
            self.translator.bake_incremental(
                self.fast_refs,
                queue,
                self._algebra
            )
            self.command_buffer.clear()

        self.store.sync_state = SyncState.CLEAN
        self._is_normalized = False

    def get_fields(self, target_states: Any) -> np.ndarray:
        indices = self.translator.get_hardware_indices(target_states)
        return self.utility.get_fields(self.fast_refs, indices)

    def sync_to_domain(self) -> None:
        """
        Pulls hardware array data back up into the Python OOP domain mapper cache.
        """
        # FIX: Use the safely stored domain mapper reference!
        if self._domain_mapper:
            self.translator.sync(self.fast_refs, domain_mapper=self._domain_mapper)

    @classmethod
    def create_from_raw(
            cls,
            utility: 'IFieldKernelUtility',
            contract: 'IFieldKernelDataContract',
            storage: 'IFieldKernelStorage',
            translator: 'IFieldTranslator',
            states: Union[Iterable[np.ndarray], np.ndarray],
            fields: Union[Iterable[np.ndarray], np.ndarray]
    ) -> 'FieldComponentManager':
        """
        FACTORY: Instantiates an empty manager, warms up the JIT compiler,
        and immediately populates it via the incremental bake pipeline.
        """
        # 1. Matches the new __init__ signature
        manager = cls(
            utility=utility,
            contract=contract,
            storage=storage,
            translator=translator,
            domain_mapper=None
        )

        manager.warmup_hardware()
        manager.set_fields(states, fields)
        manager.commit_frame()

        return manager

    def warmup_hardware(self) -> None:
        """
        FORCED DISPATCH: We trigger the specific NumPy fancy-indexing paths
        that cause the initial 600ms stutter.
        """
        # Fetch actual dims from contract
        s_dim = self.store.contract.state_dimensions
        f_dim = self.store.contract.field_dimensions

        # Use a small but 'Real' looking slice (e.g. 5 rows)
        # Numba treats (1, D) and (5, D) differently sometimes; (N, D) is the goal.
        d_states = np.zeros((5, s_dim), dtype=np.float64)
        d_fields = np.zeros((5, f_dim), dtype=np.float64)

        # Trigger the INCREMENTAL BAKE (The slow part)
        self.set_fields(d_states, d_fields)
        self.commit_frame()  # This forces the Translator's logic to compile

        # Trigger the UTILITY (The math part)
        self.utility.add_mappers_inplace(self.fast_refs, self.fast_refs)

        # Wipe the warmup data so the IDs don't conflict with the real simulation
        self.translator._state_to_id_map.clear()
        self.fast_refs.is_mapped_array.fill(False)

    def get_normalized_field(self) -> np.ndarray:
        """
        Returns the normalized field array, executing the in-place calculation
        only if the field has changed since the last normalization.
        """
        # Ensure the hardware arrays reflect any pending surgical updates
        if self.store.sync_state == SyncState.EXECUTION_DIRTY:
            self.commit_frame()

        # If the data changed since last normalization, trigger the utility
        if not self._is_normalized:
            self.utility.normalize_field(self.fast_refs)
            self._is_normalized = True

        return self.fast_refs.normalized_field_array

    # ==========================================
    # STATIC BRIDGE API (Lightweight Instance Methods)
    # ==========================================

    # def batch_add_raw(self, target_states: np.ndarray, target_fields: np.ndarray, source_states: np.ndarray, source_fields: np.ndarray, **kwargs: Any) -> None:
    #     self._ensure_static()
    #     self.utility.batch_add_bridge_inplace(target_states, target_fields, source_states, source_fields, **kwargs)

    def batch_multiply_raw(self, target_states: np.ndarray, target_fields: np.ndarray, source_states: np.ndarray, source_fields: np.ndarray, **kwargs: Any) -> None:
        self._ensure_static()
        self.utility.batch_multiply_bridge_inplace(target_states, target_fields, source_states, source_fields, **kwargs)

    def batch_norm_raw(self, states: np.ndarray, fields: np.ndarray, out_norms: np.ndarray, **kwargs: Any) -> None:
        self._ensure_static()
        self.utility.batch_norm_bridge(states, fields, out_norms, **kwargs)

    def batch_normalize_raw(self, states: np.ndarray, fields: np.ndarray, **kwargs: Any) -> None:
        self._ensure_static()
        self.utility.batch_normalize_bridge_inplace(states, fields, **kwargs)


    @property
    def add_vectors(self) -> Callable:
        """Returns the raw JIT-compiled vector addition function for external hardware loops."""
        return self.utility.get_add_kernel()

    @property
    def multiply_vectors(self) -> Callable:
        """Returns the raw JIT-compiled vector multiplication function for external hardware loops."""
        return self.utility.get_multiply_kernel()

    @property
    def norm_vector(self) -> Callable:
        """Returns the raw JIT-compiled vector norm function for external hardware loops."""
        return self.utility.get_norm_kernel()

    def batch_add_raw(self, target_states: np.ndarray, target_fields: np.ndarray, source_states: np.ndarray, source_fields: np.ndarray, **kwargs: Any) -> None:
        self._ensure_static()
        self.utility.batch_add_bridge_inplace(target_states, target_fields, source_states, source_fields, **kwargs)

    def fill(self, value: float) -> None:
        """Sets the entire hardware field memory to a specific scalar value."""
        self.fast_refs.field_array.fill(value)
        self.fast_refs.normalized_field_array.fill(value)
        # Mark as normalized since we just set the whole thing to a constant
        #self._is_normalized = True