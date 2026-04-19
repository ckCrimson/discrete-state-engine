from typing import Type, Any

import numpy as np

from particle_grid_simulator.src.dynamic_system.domain.interfaces.dynamic_systems import IDynamicSystemData, \
    IDynamicSystemRunner
from particle_grid_simulator.src.operator.domain.interfaces.operator import IOperatorUtility


class DynamicSystemRunner(IDynamicSystemRunner):
    """
    The concrete Clock engine.
    Manages the DOD pre-allocated rolling buffer and fast binary I/O.
    """

    def __init__(self, system_data: IDynamicSystemData[Any], utility: Type['IOperatorUtility']):
        self.data = system_data
        self.utility = utility

        # Ensure directories exist
        self.data.save_directory.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.data.save_directory / "temp_chunks"
        self.temp_dir.mkdir(exist_ok=True)

        # Batch Constraints
        self.N_entities = self.data.initial_states.shape[0]
        self.state_dim = self.data.initial_states.shape[1]

        # DOD Memory Allocation: The Rolling Buffer
        self.history_buffer = np.zeros(
            (self.data.history_window_size, self.N_entities, self.state_dim),
            dtype=self.data.initial_states.dtype
        )

        # Internal Clock State
        self.current_states = self.data.initial_states.copy()
        self.tick_count = 0
        self.buffer_index = 0
        self.chunk_count = 0

        # Record t=0
        self._record_frame()

    def next(self) -> None:
        # 1. Apply the Operator via the injected Utility
        # For a classical DS, the context is just the current states.
        new_state_obj = self.utility.evolve(self.data.operator, self.current_states)

        # Extract raw numpy array from the returned State object
        self.current_states = new_state_obj.vector

        # 2. Record to buffer
        self.tick_count += 1
        self._record_frame()

    def _record_frame(self) -> None:
        """Internal DOD memory router."""
        self.history_buffer[self.buffer_index] = self.current_states
        self.buffer_index += 1

        # Flush to binary disk format if buffer is full
        if self.buffer_index >= self.data.history_window_size:
            self._flush_buffer()

    def _flush_buffer(self, partial_size: int = None) -> None:
        """Saves the rolling buffer to a high-speed .npy binary chunk."""
        size_to_save = partial_size if partial_size is not None else self.data.history_window_size
        if size_to_save == 0:
            return

        chunk_path = self.temp_dir / f"chunk_{self.chunk_count:05d}.npy"
        np.save(chunk_path, self.history_buffer[:size_to_save])

        self.chunk_count += 1
        self.buffer_index = 0

    def end(self, compile_csv: bool = True) -> None:
        """
        The Lifecycle terminator.
        Flushes the remaining RAM and compiles binary chunks into the final format.
        """
        print(f"Ending simulation at tick {self.tick_count}...")

        # 1. Flush whatever is left in the partial buffer
        self._flush_buffer(partial_size=self.buffer_index)

        # 2. Compile Data
        if compile_csv:
            self._compile_to_csv()

        print(f"Simulation telemetry saved to {self.data.save_directory}")

    def _compile_to_csv(self) -> None:
        """Reads all fast-binary chunks and compiles them into a human-readable CSV."""
        print("Compiling binary chunks to CSV...")
        chunk_files = sorted(self.temp_dir.glob("chunk_*.npy"))

        if not chunk_files:
            return

        # Stack all chunks vertically into one master timeline
        all_data = [np.load(f) for f in chunk_files]
        master_array = np.vstack(all_data)

        # Reshape for CSV (Flatten the Entities and Dimensions)
        # From (Total_Ticks, N, Dim) -> (Total_Ticks, N * Dim)
        flat_array = master_array.reshape(master_array.shape[0], -1)

        csv_path = self.data.save_directory / "compiled_telemetry.csv"

        # Generate clean column headers (e.g., P0_X, P0_Y, P1_X, P1_Y...)
        headers = []
        for i in range(self.N_entities):
            for d in range(self.state_dim):
                dim_label = chr(88 + d) if d < 3 else f"D{d}"  # X, Y, Z fallback
                headers.append(f"P{i}_{dim_label}")

        np.savetxt(csv_path, flat_array, delimiter=",", header=",".join(headers), comments="")

        # Cleanup temporary binary files
        for f in chunk_files:
            f.unlink()
        self.temp_dir.rmdir()