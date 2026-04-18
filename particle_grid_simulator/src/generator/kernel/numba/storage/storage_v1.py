import numpy as np

from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef, ICSRGeneratorStorage, \
    GeneratorKernelDataContract


# Assuming imports from your architecture:
# from particle_grid_simulator.src.generator.interfaces.storage import ICSRGeneratorStorage
# from particle_grid_simulator.src.generator.contracts import GeneratorKernelDataContract, GeneratorKernelFastRef

class NumbaCSRGeneratorStorage(ICSRGeneratorStorage):
    """
    CONCRETE NUMBA STORAGE: Physically allocates the Ping-Pong C-arrays
    and constructs the FastRef pointer payload.
    """

    def __init__(self, contract: 'GeneratorKernelDataContract') -> None:
        self._contract = contract

        # ==========================================
        # 1. ALLOCATE OWNED MEMORY (The Ping-Pong Buffers)
        # We use np.zeros to ensure clean memory blocks for the C-compiler.
        # ==========================================
        max_states = contract.max_active_states
        s_dim = contract.state_dimensions
        f_dim = contract.input_field_size

        buf_A_states = np.zeros((max_states, s_dim), dtype=np.float64)
        buf_A_fields = np.zeros((max_states, f_dim), dtype=np.float64)

        buf_B_states = np.zeros((max_states, s_dim), dtype=np.float64)
        buf_B_fields = np.zeros((max_states, f_dim), dtype=np.float64)

        # ==========================================
        # 2. PLACEHOLDER INJECTED MEMORY
        # These cost practically zero bytes. They just hold the shape/type
        # contract until the Translator runs `bake_topology_field` and
        # swaps these pointers for the real ones.
        # ==========================================
        g_dim = contract.global_field_size
        empty_states = np.empty((0, s_dim), dtype=np.float64)
        empty_global = np.empty((0, g_dim), dtype=np.float64)
        empty_indices = np.empty(0, dtype=np.int32)

        # ==========================================
        # 3. CONSTRUCT THE FAST-REF PAYLOAD
        # ==========================================
        self._fast_refs = GeneratorKernelFastRef(
            # Owned arrays
            buffer_A_states=buf_A_states,
            buffer_A_fields=buf_A_fields,
            buffer_B_states=buf_B_states,
            buffer_B_fields=buf_B_fields,
            active_count_A=0,
            active_count_B=0,

            # Injected CSR arrays
            state_coordinates=empty_states,
            edge_offsets=empty_indices,
            edge_targets=empty_indices,

            # Injected Global Field arrays
            global_states=empty_states,
            global_fields=empty_global,
            global_normalized_fields=empty_global
        )

    # ==========================================
    # INTERFACE FULFILLMENT: SINGLE SOURCE OF TRUTH
    # We strictly route all property requests through the FastRef
    # to guarantee we are always looking at the exact same memory Numba sees.
    # ==========================================

    @property
    def fast_refs(self) -> 'GeneratorKernelFastRef':
        return self._fast_refs

    # --- OWNED MEMORY ---
    @property
    def buffer_A_states(self) -> np.ndarray: return self._fast_refs.buffer_A_states

    @property
    def buffer_A_fields(self) -> np.ndarray: return self._fast_refs.buffer_A_fields

    @property
    def buffer_B_states(self) -> np.ndarray: return self._fast_refs.buffer_B_states

    @property
    def buffer_B_fields(self) -> np.ndarray: return self._fast_refs.buffer_B_fields

    # --- INJECTED CSR MEMORY ---
    @property
    def state_coordinates(self) -> np.ndarray: return self._fast_refs.state_coordinates

    @property
    def edge_offsets(self) -> np.ndarray: return self._fast_refs.edge_offsets

    @property
    def edge_targets(self) -> np.ndarray: return self._fast_refs.edge_targets

    # --- INJECTED GLOBAL FIELD MEMORY ---
    @property
    def global_states(self) -> np.ndarray: return self._fast_refs.global_states

    @property
    def global_fields(self) -> np.ndarray: return self._fast_refs.global_fields

    @property
    def global_normalized_fields(self) -> np.ndarray: return self._fast_refs.global_normalized_fields

    def clear(self) -> None:
        """Zeros out the real-valued ping-pong memory."""
        self._fast_refs.active_count_A = 0
        self._fast_refs.active_count_B = 0
        self._fast_refs.buffer_A_states.fill(0.0)
        self._fast_refs.buffer_A_fields.fill(0.0)
        self._fast_refs.buffer_B_states.fill(0.0)
        self._fast_refs.buffer_B_fields.fill(0.0)