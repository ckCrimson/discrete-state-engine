from dataclasses import dataclass
from typing import Set

import numpy as np


@dataclass(frozen=True)
class State:
    vector: np.ndarray

    def __repr__(self) -> str:
        return f"State(vector={self.vector})"

    def __str__(self) -> str:
        return f"State: {self.vector}"

    def __hash__(self) -> int:
        # Converts the raw C-contiguous memory block to an immutable byte string for hashing
        return hash(self.vector.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        # Safely compares the actual numeric values of the arrays
        return np.array_equal(self.vector, other.vector)

@dataclass(frozen=True)
class StateSpace:
    states: Set[State]

    def __repr__(self) -> str:
        return f"StateSpace(states={self.states})"

    def __str__(self) -> str:
        return f"StateSpace with {len(self.states)} states"