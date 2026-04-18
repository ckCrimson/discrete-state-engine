import numpy as np

from particle_grid_simulator.src.generator.iterfaces.storage import (
    ICSRGeneratorStorage,
    GeneratorKernelDataContract,
    GeneratorKernelFastRef
)


class NumbaComplexCSRGeneratorStorage(ICSRGeneratorStorage):
    """
    CONCRETE STORAGE: Fulfills ICSRGeneratorStorage.
    Allocates 64-bit float Ping-Pong buffers for states, and 128-bit complex buffers for fields.
    Acts as the single source of truth for the injected FastRef pointers.
    """

    def __init__(self, contract: GeneratorKernelDataContract) -> None:
        self._contract = contract

        max_states = contract.max_active_states
        s_dim = contract.state_dimensions
        f_dim = contract.input_field_size
        g_dim = contract.global_field_size

        # --- 1. OWNED MEMORY (Ping-Pong Buffers) ---
        # States remain float64 (X, Y coordinates)
        buf_A_states = np.zeros((max_states, s_dim), dtype=np.float64)
        buf_B_states = np.zeros((max_states, s_dim), dtype=np.float64)

        # Fields MUST be complex128 to hold wave amplitudes (a + bj)
        buf_A_fields = np.zeros((max_states, f_dim), dtype=np.complex128)
        buf_B_fields = np.zeros((max_states, f_dim), dtype=np.complex128)

        # --- 2. INJECTED MEMORY PLACEHOLDERS ---
        # These will be replaced with real pointers during Environment Injection
        empty_states = np.empty((0, s_dim), dtype=np.float64)
        empty_global = np.empty((0, g_dim), dtype=np.complex128)
        empty_indices = np.empty(0, dtype=np.int64)

        # --- 3. CONSTRUCT FASTREF PAYLOAD ---
        # This object is what Numba actually sees. By typing it here,
        # we lock the memory width for the JIT compiler.
        self._fast_refs = GeneratorKernelFastRef(
            buffer_A_states=buf_A_states,
            buffer_A_fields=buf_A_fields,
            buffer_B_states=buf_B_states,
            buffer_B_fields=buf_B_fields,
            active_count_A=0,
            active_count_B=0,
            state_coordinates=empty_states,
            edge_offsets=empty_indices,
            edge_targets=empty_indices,
            global_states=empty_states,
            global_fields=empty_global,
            global_normalized_fields=empty_global
        )

    # ==========================================
    # IGENERATORKERNELSTORAGE FULFILLMENT
    # ==========================================
    @property
    def fast_refs(self) -> GeneratorKernelFastRef:
        return self._fast_refs

    def get_fast_ref(self) -> GeneratorKernelFastRef:
        return self._fast_refs

    @property
    def buffer_A_states(self) -> np.ndarray:
        return self._fast_refs.buffer_A_states

    @property
    def buffer_A_fields(self) -> np.ndarray:
        return self._fast_refs.buffer_A_fields

    @property
    def buffer_B_states(self) -> np.ndarray:
        return self._fast_refs.buffer_B_states

    @property
    def buffer_B_fields(self) -> np.ndarray:
        return self._fast_refs.buffer_B_fields

    def clear(self) -> None:
        """Zeros out the counters and the memory buffers."""
        self._fast_refs.active_count_A = 0
        self._fast_refs.active_count_B = 0

        # Using explicit complex literals (0j) ensures the underlying
        # C-memory is cleared with 128-bit precision.
        self._fast_refs.buffer_A_states.fill(0.0)
        self._fast_refs.buffer_A_fields.fill(0.0 + 0.0j)
        self._fast_refs.buffer_B_states.fill(0.0)
        self._fast_refs.buffer_B_fields.fill(0.0 + 0.0j)

    # ==========================================
    # ICSRGENERATORSTORAGE FULFILLMENT (Injected Memory)
    # ==========================================
    @property
    def state_coordinates(self) -> np.ndarray:
        return self._fast_refs.state_coordinates

    @property
    def edge_offsets(self) -> np.ndarray:
        return self._fast_refs.edge_offsets

    @property
    def edge_targets(self) -> np.ndarray:
        return self._fast_refs.edge_targets

    @property
    def global_states(self) -> np.ndarray:
        return self._fast_refs.global_states

    @property
    def global_fields(self) -> np.ndarray:
        return self._fast_refs.global_fields

    @property
    def global_normalized_fields(self) -> np.ndarray:
        return self._fast_refs.global_normalized_fields