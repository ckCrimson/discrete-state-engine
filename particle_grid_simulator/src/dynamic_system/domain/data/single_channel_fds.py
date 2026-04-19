from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import numpy as np

from particle_grid_simulator.src.dynamic_system.domain.interfaces.single_channel_fds import ISingleChannelFDSData, T, \
    ISingleChannelFDSRunner
from particle_grid_simulator.src.dynamic_system.domain.utility.single_channel_fds import ISingleChannelFDSUtility


@dataclass(frozen=True)
class SingleChannelFDSData(ISingleChannelFDSData[T]):
    """
    The immutable blueprint. The user provides pre-baked Component Managers
    (Topology and Field can be deep-baked before passing them here).
    """
    _initial_states: np.ndarray
    _initial_fields: np.ndarray
    _topology_cm: Any
    _field_cm: Any
    _generator_cm: Any
    _operator_cm: Any
    _history_window_size: int
    _save_directory: Path
    _is_independent: bool = False

    @property
    def initial_states(self) -> np.ndarray: return self._initial_states
    @property
    def initial_fields(self) -> np.ndarray: return self._initial_fields
    @property
    def topology_cm(self) -> Any: return self._topology_cm
    @property
    def field_cm(self) -> Any: return self._field_cm
    @property
    def generator_cm(self) -> Any: return self._generator_cm
    @property
    def operator_cm(self) -> Any: return self._operator_cm
    @property
    def history_window_size(self) -> int: return self._history_window_size
    @property
    def save_directory(self) -> Path: return self._save_directory
    @property
    def is_independent(self) -> bool: return self._is_independent


class SingleChannelFDSRunner(ISingleChannelFDSRunner):
    """
    The Master Clock. Handles Memory I/O, JIT Compilation Poison Prevention,
    and automatic Channel shape routing (Overlapping vs Independent).
    """

    def __init__(self, system_data: ISingleChannelFDSData[Any], utility: Type[ISingleChannelFDSUtility]):
        self.data = system_data
        self.utility = utility

        self.data.save_directory.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.data.save_directory / "temp_chunks"
        self.temp_dir.mkdir(exist_ok=True)

        self.N_entities = self.data.initial_states.shape[0]
        self.state_dim = self.data.initial_states.shape[1]
        self.field_dim = self.data.initial_fields.shape[1]

        # Buffer initialization
        self.history_buffer = np.zeros(
            (self.data.history_window_size, self.N_entities, self.state_dim),
            dtype=self.data.initial_states.dtype
        )

        # Internal State tracking
        self.current_states = self.data.initial_states.copy()
        self.current_fields = self.data.initial_fields.copy()
        self._last_gen_states = None
        self._last_gen_fields = None

        self.tick_count = 0
        self.buffer_index = 0
        self.chunk_count = 0

        # === INITIALIZATION PIPELINE ===
        self._wire_and_warmup()

        # Record t=0
        self._record_frame()

    def _wire_and_warmup(self) -> None:
        """
        Solves the Wiring and 'Dummy Poisoning' hurdles automatically.
        """
        print("   [FDS Runner] Wiring Environment...")
        self.data.generator_cm.inject_environment(self.data.topology_cm, self.data.field_cm)

        print("   [FDS Runner] Executing Throwaway JIT Compilation...")
        # 1. Use a coordinate far outside normal bounds to prevent caching origin
        # FIX: Inherit dtypes dynamically from the user's initial data!
        # Set dummy_s to 0.0 to safely clear bounded topologies.
        dummy_s = np.full((self.N_entities, self.state_dim), 0.0, dtype=self.current_states.dtype)
        dummy_f = np.ones((self.N_entities, self.field_dim), dtype=self.current_fields.dtype)


        # 2. Trigger Generator C-Kernels
        self.data.generator_cm.load_initial_state(dummy_s, dummy_f)
        g_s, g_f = self.data.generator_cm.generate_steps(steps=1)

        # 3. Handle Broadcasting for Dummy Operator
        M = len(g_s)
        b_s = np.ascontiguousarray(np.broadcast_to(g_s, (self.N_entities, M, self.state_dim)))
        if self.data.is_independent:
            b_f = g_f  # Native N shape
        else:
            b_f = np.ascontiguousarray(np.broadcast_to(g_f, (self.N_entities, M, self.field_dim)))

        # 4. Trigger Operator C-Kernels
        self.utility.evolve(self.data.operator_cm, dummy_s, b_s, b_f)

        # 5. Clean up the dummy poison and load reality
        self.data.generator_cm.clear()
        print("   [FDS Runner] System Ready.")

    def next(self, apply_generator: bool = True, steps: int = 1) -> None:
        if apply_generator:
            # PHASE 1: Wave Expansion
            self.data.generator_cm.clear()
            self.data.generator_cm.load_initial_state(self.current_states, self.current_fields)

            # Store the resulting wave for the operator phase
            self._last_gen_states, self._last_gen_fields = self.data.generator_cm.generate_steps(steps=steps)

        else:
            # PHASE 2: Observation & Collapse
            if self._last_gen_states is None:
                raise RuntimeError("Must call next(apply_generator=True) before collapsing.")

            M = len(self._last_gen_states)

            # Broadcast generated states to match Operator contract
            batch_gen_states = np.ascontiguousarray(
                np.broadcast_to(self._last_gen_states, (self.N_entities, M, self.state_dim))
            )

            # Route Field arrays based on Channel type (Overlapping vs Independent)
            if self.data.is_independent:
                # Shape is already (N, M, Field_Dim)
                batch_gen_fields = np.empty((self.N_entities, M, self.field_dim), dtype=np.float64)
                for p in range(self.N_entities):
                    # Note: Handles the slicing mapping you had in your independent script
                    batch_gen_fields[p, :, 0] = self._last_gen_fields[:, p]
            else:
                # Overlapping returns (1, M, Field_Dim), broadcast to all N entities
                batch_gen_fields = np.ascontiguousarray(
                    np.broadcast_to(self._last_gen_fields, (self.N_entities, M, self.field_dim))
                )

            # Evolve current states in-place via utility
            self.utility.evolve(self.data.operator_cm, self.current_states, batch_gen_states, batch_gen_fields)

            # Log the new state
            self.tick_count += 1
            self._record_frame()

            # Clear buffers to prevent stale data
            self._last_gen_states = None
            self._last_gen_fields = None

    # Memory handling remains identical to the classical system
    def _record_frame(self) -> None:
        self.history_buffer[self.buffer_index] = self.current_states
        self.buffer_index += 1
        if self.buffer_index >= self.data.history_window_size:
            self._flush_buffer()

    def _flush_buffer(self, partial_size: int = None) -> None:
        size_to_save = partial_size if partial_size is not None else self.data.history_window_size
        if size_to_save == 0: return
        chunk_path = self.temp_dir / f"chunk_{self.chunk_count:05d}.npy"
        np.save(chunk_path, self.history_buffer[:size_to_save])
        self.chunk_count += 1
        self.buffer_index = 0

    def end(self, compile_csv: bool = True) -> None:
        print(f"   [FDS Runner] Simulation complete at tick {self.tick_count}.")
        self._flush_buffer(partial_size=self.buffer_index)
        if compile_csv:
            self._compile_to_csv()

    def _compile_to_csv(self) -> None:
        print("   [FDS Runner] Compiling high-speed binary chunks to CSV...")
        chunk_files = sorted(self.temp_dir.glob("chunk_*.npy"))
        if not chunk_files: return
        master_array = np.vstack([np.load(f) for f in chunk_files])
        flat_array = master_array.reshape(master_array.shape[0], -1)
        csv_path = self.data.save_directory / "compiled_telemetry.csv"

        headers = []
        for i in range(self.N_entities):
            for d in range(self.state_dim):
                dim_label = chr(88 + d) if d < 3 else f"D{d}"
                headers.append(f"P{i}_{dim_label}")

        np.savetxt(csv_path, flat_array, delimiter=",", header=",".join(headers), comments="")
        for f in chunk_files: f.unlink()
        self.temp_dir.rmdir()