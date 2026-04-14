from abc import abstractmethod
from typing import Iterable, Any, Callable
import numpy as np

from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility

class ITopologyUtility(IKernelUtility):
    """
    CONTRACT: Stateless hardware execution for Topology traversals.
    Must operate entirely on the injected fast_ref and hardware functions.
    Zero state is permitted.
    """

    @staticmethod
    @abstractmethod
    def get_reachable_multi_step_frontier(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray,
        steps: int
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def get_reachable_multi_step_basin(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray,
        steps: int
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def get_reachable(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def get_reaching(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def get_reaching_multi_step_frontier(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray,
        steps: int
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def get_reaching_multi_step_basin(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray,
        steps: int
    ) -> Iterable[np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def warmup(
        fast_ref: Any,
        neighbour_func: Callable[[np.ndarray], np.ndarray],
        state_vector_in: np.ndarray,
        steps: int
    ) -> None:
        """Forces the graph to pre-build up to the specified depth."""
        pass