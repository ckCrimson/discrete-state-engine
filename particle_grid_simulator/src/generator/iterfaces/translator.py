from abc import abstractmethod
from typing import Any, List, Tuple
import numpy as np

# Assuming base ITranslator is imported from your core module
from hpc_ecs_core.src.hpc_ecs_core.interfaces import ITranslator
from particle_grid_simulator.src.generator.domain.interfaces.field_generator import IFieldGeneratorData
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef


# Type checking imports
# from particle_grid_simulator.src.generator.contracts import GeneratorKernelFastRef
# from particle_grid_simulator.src.field.domain.interfaces.generator_interface import IFieldGeneratorData

class IGeneratorTranslator(ITranslator):
    """
    CONTRACT: The Data-Oriented router for the Generator module.
    Strictly responsible for translating Domain objects into C-arrays and
    mapping external environment pointers into the Generator's FastRef.
    """

    @abstractmethod
    def bake(

            self,
            fast_refs: 'GeneratorKernelFastRef',
            initial_data: Tuple[np.ndarray, np.ndarray]  # <-- THE FIX
    ) -> None:
        """
        Loads the initial active particle states and fields into Buffer A.
        initial_data is strictly a tuple of (raw_states, raw_fields).
        """
        pass

    @abstractmethod
    def bake_incremental(self, fast_refs: 'GeneratorKernelFastRef', queue: List[tuple], **kwargs: Any) -> None:
        """
        Required by the base ITranslator contract.
        For the continuous Ping-Pong loop of a generator, this is typically a no-op
        unless specific mid-simulation surgical injection is required later.
        """
        pass

    # ==========================================
    # GENERATOR SPECIFIC TRANSLATION
    # ==========================================

    @abstractmethod
    def bake_topology_field(
            self,
            fast_refs: 'GeneratorKernelFastRef',
            topology_cm: Any,
            global_field_cm: Any
    ) -> None:
        """
        THE INJECTION ROUTER.
        Extracts the raw memory pointers from the external Topology and Global Field
        Utility Component Managers, and binds them to the empty placeholder
        slots in the Generator's FastRef.

        This executes with zero memory allocation.
        """
        pass

    @abstractmethod
    def sync_to_domain(
            self,
            fast_refs: 'GeneratorKernelFastRef',
            active_buffer_flag: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the final computed data from the winning Numba hardware buffer.

        Args:
            fast_refs: The structural payload containing the buffers and counts.
            active_buffer_flag: "A" or "B", indicating which buffer holds the final state.

        Returns:
            A tuple of (final_states_array, final_fields_array) stripped of all
            empty allocated memory (sliced exactly to the active_count),
            ready to be handed back to the Object-Oriented Domain.
        """
        pass

    @abstractmethod
    def sync(self, fast_refs: 'GeneratorKernelFastRef', **kwargs: Any) -> Any:
        """
        The base sync method. For the Generator, this should act as a wrapper
        that delegates directly to `sync_to_domain` based on kwargs.
        """
        pass