from typing import Iterable, Callable

import numpy as np

from ..state_domain import State, StateSpace


class StateSpaceUtility:
    """
    Pure Python domain operations for the Field Dynamic System.
    These methods mutate the high-level Python objects and serve as the
    blueprint for the high-speed hardware kernels.
    """

    @staticmethod
    def union_inplace(space: StateSpace, new_states: Iterable[State]) -> None:
        """
        Merges new states into the existing state space in-place.
        Under the hood, this utilizes Python's highly optimized set.update().
        """
        space.states.update(new_states)

    @staticmethod
    def intersection_inplace(space: StateSpace, valid_states: Iterable[State]) -> None:
        """
        Filters the state space in-place, keeping only states that are also
        present in the provided iterable.
        """
        space.states.intersection_update(valid_states)

    @staticmethod
    def map_inplace(space: StateSpace, transform_func: Callable[[State], State]) -> None:
        """
        Applies a transformation function to every state in the space.

        Because individual State vectors are strictly frozen, this computes
        the new states and replaces the contents of the set in-place,
        preserving the original StateSpace memory reference.
        """
        # 1. Compute the new transformed states
        mapped_states = {transform_func(state) for state in space.states}

        # 2. Mutate the existing set in-place
        space.states.clear()
        space.states.update(mapped_states)

    @staticmethod
    def add_state(state: State, state_space: StateSpace):
        """
        Adds a new state in the state space set
        """
        state_space.states.add(state)

    @staticmethod
    def remove_state(state: State, state_space: StateSpace):
        """
        Removes a new state in the state space set
        """
        state_space.states.remove(state)

    @staticmethod
    def to_vector_matrix(space: StateSpace) -> np.ndarray:
        """ Compiles the set of OOP objects into a single 2D C-array. """
        if not space.states:
            return np.empty((0,))
        return np.vstack([s.vector for s in space.states])


