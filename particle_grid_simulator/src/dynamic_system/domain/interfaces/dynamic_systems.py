
from pathlib import Path
from typing import Protocol, TypeVar
import numpy as np

from particle_grid_simulator.src.operator.domain.interfaces.operator import IOperatorData

T = TypeVar('T')

# Assuming IOperatorData and IOperatorUtility are imported here...

class IDynamicSystemData(Protocol[T]):
    """
    The immutable mathematical blueprint of the dynamic system.
    """
    @property
    def initial_states(self) -> np.ndarray:
        """The starting N-dimensional batch array (s0). Shape: (N, State_Dim)"""
        ...

    @property
    def operator(self) -> 'IOperatorData[T]':
        """The mathematical rule (Theta) governing the evolution."""
        ...

    @property
    def history_window_size(self) -> int:
        """How many ticks to store in RAM before flushing to disk."""
        ...

    @property
    def save_directory(self) -> Path:
        """The root directory where the telemetry will be compiled."""
        ...


class IDynamicSystemRunner(Protocol):
    """
    The Clock and Memory Manager.
    Executes the ticks and handles the I/O lifecycle without exposing internal buffers.
    """
    def next(self) -> None:
        """Advances the system by one discrete tick (t -> t+1) and records telemetry."""
        ...

    def end(self, compile_csv: bool = True) -> None:
        """Flushes remaining memory and cleanly terminates the simulation."""
        ...