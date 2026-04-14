# High-level domain structures for Topology
from typing import Callable, Iterable, Type, Optional, Dict, Tuple

from particle_grid_simulator.src.state.domain import State


class Topology:
    """
    Pure Data Container for the rules of the universe and the dynamic cache.
    """

    def __init__(
            self,
            reachable_func: Callable[[State], Iterable[State]],
            state_class: Type,
            reaching_func: Optional[Callable[[State], Iterable[State]]] = None,
            use_cache: bool = True
    ):
        self.reachable_func = reachable_func
        self.raw_reaching = reaching_func
        self.state_class = state_class

        # The Data (Utility will manage this)
        self.use_cache = use_cache
        self.adjacency_cache: Dict[State, Tuple[State, ...]] = {}