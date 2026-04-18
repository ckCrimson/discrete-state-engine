import numpy as np

from particle_grid_simulator.src.generator.iterfaces.storage import ParallelGeneratorFastRef, \
    IParallelGeneratorKernelStorage, ParallelGeneratorDataContract


class NumbaParallelArenaStorage(IParallelGeneratorKernelStorage):
    """
    CONCRETE NUMBA STORAGE: Physically allocates the N-Lane Ping-Pong C-arrays
    and the crucial thread-safe scratchpads.
    """

    def __init__(self, contract: 'ParallelGeneratorDataContract') -> None:
        self._contract = contract

        # ==========================================
        # 1. EXTRACT DIMENSIONS
        # ==========================================
        num_p = contract.max_particles
        max_s = contract.max_active_states
        s_dim = contract.state_dimensions
        f_dim = contract.input_field_size

        # Max nodes in the graph (used for pre-allocating scratchpads)
        # Numba CSR arrays will never exceed this size.
        max_csr_nodes = contract.global_field_size

        # ==========================================
        # 2. ALLOCATE N-LANE OWNED MEMORY (Ping-Pong)
        # ==========================================
        buf_A_states = np.zeros((num_p, max_s, s_dim), dtype=np.float64)
        buf_A_fields = np.zeros((num_p, max_s, f_dim), dtype=np.float64)

        buf_B_states = np.zeros((num_p, max_s, s_dim), dtype=np.float64)
        buf_B_fields = np.zeros((num_p, max_s, f_dim), dtype=np.float64)

        counts_A = np.zeros(num_p, dtype=np.int32)
        counts_B = np.zeros(num_p, dtype=np.int32)

        # ==========================================
        # 3. ALLOCATE THREAD-SAFE SCRATCHPADS
        # Every particle thread gets its own dedicated workspace.
        # ==========================================
        scratch_fields = np.zeros((num_p, max_csr_nodes, f_dim), dtype=np.float64)
        scratch_seen = np.zeros((num_p, max_csr_nodes), dtype=np.bool_)

        # ==========================================
        # 4. PLACEHOLDER INJECTED MEMORY
        # ==========================================
        empty_states = np.empty((0, s_dim), dtype=np.float64)
        empty_global = np.empty((0, f_dim), dtype=np.float64)
        empty_indices = np.empty(0, dtype=np.int32)

        # ==========================================
        # 5. CONSTRUCT THE FAST-REF PAYLOAD
        # ==========================================
        self._parallel_refs = ParallelGeneratorFastRef(
            buffer_A_states=buf_A_states,
            buffer_A_fields=buf_A_fields,
            buffer_B_states=buf_B_states,
            buffer_B_fields=buf_B_fields,
            active_counts_A=counts_A,
            active_counts_B=counts_B,

            scratchpad_acc_fields=scratch_fields,
            scratchpad_seen_nodes=scratch_seen,

            state_coordinates=empty_states,
            edge_offsets=empty_indices,
            edge_targets=empty_indices,
            global_states=empty_states,
            global_fields=empty_global,
            global_normalized_fields=empty_global
        )

    # ==========================================
    # INTERFACE FULFILLMENT
    # ==========================================
    @property
    def parallel_fast_refs(self) -> 'ParallelGeneratorFastRef':
        return self._parallel_refs

    @property
    def fast_refs(self):
        # Fallback to prevent base class crashes, though Parallel Utility won't use it
        return self._parallel_refs

    # --- INJECTED CSR MEMORY (Shared Read-Only) ---
    @property
    def state_coordinates(self) -> np.ndarray: return self._parallel_refs.state_coordinates

    @property
    def edge_offsets(self) -> np.ndarray: return self._parallel_refs.edge_offsets

    @property
    def edge_targets(self) -> np.ndarray: return self._parallel_refs.edge_targets

    # --- INJECTED GLOBAL FIELD MEMORY (Shared Read-Only) ---
    @property
    def global_states(self) -> np.ndarray: return self._parallel_refs.global_states

    @property
    def global_fields(self) -> np.ndarray: return self._parallel_refs.global_fields

    @property
    def global_normalized_fields(self) -> np.ndarray: return self._parallel_refs.global_normalized_fields

    # --- BASE CLASS OVERRIDES (To satisfy ICSRGeneratorStorage) ---
    # We raise NotImplementedError here to ensure no one accidentally tries
    # to run the sequential engine on the parallel 3D arrays.
    @property
    def buffer_A_states(self) -> np.ndarray: raise NotImplementedError("Use parallel_fast_refs")

    @property
    def buffer_A_fields(self) -> np.ndarray: raise NotImplementedError("Use parallel_fast_refs")

    @property
    def buffer_B_states(self) -> np.ndarray: raise NotImplementedError("Use parallel_fast_refs")

    @property
    def buffer_B_fields(self) -> np.ndarray: raise NotImplementedError("Use parallel_fast_refs")

    def clear(self) -> None:
        """Zeros out the real-valued ping-pong memory for all N lanes."""
        self._parallel_refs.active_counts_A.fill(0)
        self._parallel_refs.active_counts_B.fill(0)
        self._parallel_refs.buffer_A_states.fill(0.0)
        self._parallel_refs.buffer_A_fields.fill(0.0)
        self._parallel_refs.buffer_B_states.fill(0.0)
        self._parallel_refs.buffer_B_fields.fill(0.0)