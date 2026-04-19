from typing import Protocol, Any, TypeVar
import numpy as np
from pathlib import Path

from particle_grid_simulator.src.dynamic_system.domain.interfaces.dynamic_systems import IDynamicSystemRunner, \
    IDynamicSystemData

# Assume IDynamicSystemData and IDynamicSystemRunner are imported
T = TypeVar('T')


# ==========================================
# 1. THE DATA BLUEPRINT (Extension)
# ==========================================
class ISingleChannelFDSData(IDynamicSystemData[T], Protocol[T]):
    """
    Extends the classical dynamic system blueprint with spatial/field mechanics.
    Inherits: initial_states, operator, history_window_size, save_directory.
    """

    @property
    def initial_fields(self) -> np.ndarray:
        """The initial field values injected by the entities. Shape: (N, Field_Dim)"""
        ...

    @property
    def topology_cm(self) -> Any:
        """Pre-built Topology Component Manager."""
        ...

    @property
    def generator_cm(self) -> Any:
        """Pre-built Generator Component Manager."""
        ...

    @property
    def max_steps(self) -> int:
        """Maximum micro-ticks the wave is allowed to expand."""
        ...


# ==========================================
# 2. THE RUNNER (Extension)
# ==========================================
class ISingleChannelFDSRunner(IDynamicSystemRunner, Protocol):
    """
    Extends the classical Clock to handle two-phase wave mechanics.
    Inherits: end(compile_csv: bool = True) -> None
    """

    # We safely override next() by adding optional parameters.
    # Because they have defaults, it still satisfies the classical `next() -> None` contract!
    def next(self, apply_generator: bool = True, steps: int = 1) -> None:
        """
        Advances the simulation.
        - If apply_generator=True: Entities stay still, their fields expand by 'steps'.
        - If apply_generator=False: Entities look at the global field and move (collapse).
        """
        ...