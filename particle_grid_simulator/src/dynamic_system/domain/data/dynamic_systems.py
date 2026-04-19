from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particle_grid_simulator.src.dynamic_system.domain.interfaces.dynamic_systems import IDynamicSystemData, T
from particle_grid_simulator.src.operator.domain.interfaces.operator import IOperatorData


@dataclass(frozen=True)
class DynamicSystemData(IDynamicSystemData[T]):
    """
    Concrete implementation of the System Blueprint.
    Frozen to guarantee the mathematical constraints cannot be mutated mid-simulation.
    """
    _initial_states: np.ndarray
    _operator: 'IOperatorData[T]'
    _history_window_size: int
    _save_directory: Path

    @property
    def initial_states(self) -> np.ndarray:
        return self._initial_states

    @property
    def operator(self) -> 'IOperatorData[T]':
        return self._operator

    @property
    def history_window_size(self) -> int:
        return self._history_window_size

    @property
    def save_directory(self) -> Path:
        return self._save_directory


