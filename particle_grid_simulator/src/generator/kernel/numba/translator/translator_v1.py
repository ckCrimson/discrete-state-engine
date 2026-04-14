import numpy as np
from typing import Any, List, Tuple

# Assuming these are imported from your architecture:
from hpc_ecs_core.src.hpc_ecs_core.interfaces import ITranslator
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef
from particle_grid_simulator.src.generator.iterfaces.translator import IGeneratorTranslator


from typing import Any, List, Tuple
import numpy as np


class NumbaGeneratorTranslator(IGeneratorTranslator):
    """
    CONCRETE TRANSLATOR: The Data-Oriented router for the Generator.
    Maps Domain objects and external Component Manager memory directly
    into the Generator's FastRef payload.
    """

    def bake(
            self,
            fast_refs: 'GeneratorKernelFastRef',
            initial_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:

        # 1. Unpack the naked tuple
        raw_states, raw_fields = initial_data

        num_active = len(raw_states)
        max_capacity = fast_refs.buffer_A_states.shape[0]

        # 2. Safety Guard
        if num_active > max_capacity:
            raise ValueError(
                f"Initial states ({num_active}) exceed allocated Numba capacity "
                f"({max_capacity}). Increase 'max_active_states' in the Domain Contract."
            )

        # 3. Write data to the primary buffer
        if num_active > 0:
            fast_refs.buffer_A_states[:num_active] = raw_states
            fast_refs.buffer_A_fields[:num_active] = raw_fields

        # 4. Set the High-Water Marks
        fast_refs.active_count_A = num_active
        fast_refs.active_count_B = 0

    def bake_incremental(self, fast_refs: 'GeneratorKernelFastRef', queue: List[tuple], **kwargs: Any) -> None:
        """
        No-op for the Generator. Continuous Ping-Pong loops do not
        accept surgical mid-frame command queues.
        """
        pass

    # ==========================================
    # GENERATOR SPECIFIC TRANSLATION
    # ==========================================

    def bake_topology_field(
            self,
            fast_refs: 'GeneratorKernelFastRef',
            topology_cm: Any,
            global_field_cm: Any
    ) -> None:
        """
        THE INJECTION ROUTER.
        Binds external memory pointers AND behavior directly into the FastRef.
        """
        # 1. Bind Topology CSR Arrays
        fast_refs.state_coordinates = topology_cm.fast_refs.handle_map
        fast_refs.edge_offsets = topology_cm.fast_refs.forward_starts
        fast_refs.edge_targets = topology_cm.fast_refs.forward_edges

        # 2. Bind Global Field Arrays
        fast_refs.global_states = global_field_cm.fast_refs.state_array
        fast_refs.global_fields = global_field_cm.fast_refs.field_array
        fast_refs.global_normalized_fields = global_field_cm.fast_refs.normalized_field_array

        # 3. Bind the specific Math Algebra for these fields
        fast_refs.math_multiply = global_field_cm.multiply_vectors
        fast_refs.math_norm = global_field_cm.norm_vector

    def sync_to_domain(
        self,
        fast_refs: 'GeneratorKernelFastRef',
        active_buffer_flag: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZERO-COPY SYNC: Returns a direct memory view into the hardware buffer.
        Do not trigger another generation step while holding these views!
        """
        if active_buffer_flag == 'A':
            count = fast_refs.active_count_A
            # Returning a direct slice (view), no allocation!
            final_states = fast_refs.buffer_A_states[:count]
            final_fields = fast_refs.buffer_A_fields[:count]
        elif active_buffer_flag == 'B':
            count = fast_refs.active_count_B
            final_states = fast_refs.buffer_B_states[:count]
            final_fields = fast_refs.buffer_B_fields[:count]
        else:
            raise ValueError("Invalid buffer flag.")

        return final_states, final_fields

    def sync(self, fast_refs: 'GeneratorKernelFastRef', **kwargs: Any) -> Any:
        """
        Standard interface wrapper that routes to the Ping-Pong specific sync.
        """
        winning_buffer = kwargs.get('winning_buffer', 'A')
        return self.sync_to_domain(fast_refs, active_buffer_flag=winning_buffer)