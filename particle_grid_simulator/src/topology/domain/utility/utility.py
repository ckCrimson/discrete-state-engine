from typing import Tuple

from particle_grid_simulator.src.state.domain import State, StateSpace
from particle_grid_simulator.src.topology.domain.topology_domain import Topology


class TopologyUtility:
    """
    Stateless system that handles graph traversal and cache management.
    """

    @staticmethod
    def _get_cached_reachable(topology: Topology, state: State) -> Tuple[State, ...]:
        """ Internal utility logic to handle the Dynamic Programming cache. """
        if not topology.use_cache:
            return tuple(topology.reachable_func(state))

        if state not in topology.adjacency_cache:
            topology.adjacency_cache[state] = tuple(topology.reachable_func(state))

        return topology.adjacency_cache[state]

    @staticmethod
    def clear_cache(topology: Topology) -> None:
        """ Flushes the data container's memory. """
        topology.adjacency_cache.clear()

    @staticmethod
    def get_reachable(topology: Topology, s_0: State) -> StateSpace:
        # Now routing through the stateless cache handler
        neighbors = TopologyUtility._get_cached_reachable(topology, s_0)
        return StateSpace(states=set(neighbors))

    @staticmethod
    def get_multi_step_reachable_frontier(topology: Topology, s_0: State, l: int) -> StateSpace:
        if l == 0:
            return StateSpace(states={s_0})

        current_frontier = {s_0}
        for _ in range(l):
            current_frontier = {
                next_state
                for current_state in current_frontier
                for next_state in TopologyUtility._get_cached_reachable(topology, current_state)
            }
            if not current_frontier:
                break
        return StateSpace(states=current_frontier)

    @staticmethod
    def get_multi_step_reachable_basin(topology: Topology, s_0: State, l: int) -> StateSpace:
        if l == 0:
            return StateSpace(states=set())

        current_frontier = {s_0}
        visited_basin = {s_0}
        seen = {s_0}

        for _ in range(l):
            next_frontier = set()
            for current_state in current_frontier:
                for next_state in TopologyUtility._get_cached_reachable(topology, current_state):
                    if next_state not in seen:
                        next_frontier.add(next_state)
                        seen.add(next_state)

            visited_basin.update(next_frontier)
            current_frontier = next_frontier

            if not current_frontier:
                break

        return StateSpace(states=visited_basin)