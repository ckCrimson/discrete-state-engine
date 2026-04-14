from abc import abstractmethod
from typing import Any, Callable
import numpy as np

# Assuming base import
from hpc_ecs_core.src.hpc_ecs_core.interfaces import IKernelUtility
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelFastRef


# Type checking imports
# from particle_grid_simulator.src.generator.contracts import GeneratorKernelFastRef

class IGeneratorKernelUtility(IKernelUtility):
    """
    CONTRACT: Hardware-accelerated execution of the Markovian Field Generator.
    Strictly stateless. Operates on FastRefs and raw C-pointers.
    Zero OOP objects are permitted to cross this boundary.
    """

    @staticmethod
    @abstractmethod
    def execute_multi_step(
            fast_refs: 'GeneratorKernelFastRef',
            steps: int,
            transition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            math_utility: Any,
            do_implicit_norm: bool,
            do_explicit_norm: bool,
            **kwargs: Any
    ) -> str:
        """
        Executes the L-step Ping-Pong generation loop entirely in C-memory.

        Args:
            fast_refs: The structural payload containing active buffers,
                       CSR topology arrays, and Global field arrays.
            steps: Number of forward iterations to simulate.
            transition_func: The raw, JIT-compiled C-function pointer for T(s_j -> s_i).
            math_utility: The injected Field Utility class (e.g., NumbaKernelFieldUtility)
                          containing the @staticmethod vector math pointers.
            do_implicit_norm: Flag to enable magnitude normalization.
            do_explicit_norm: Flag to enable explicit frontier conservation.

        Returns:
            str: 'A' or 'B', indicating which Ping-Pong buffer holds the final computed state.
        """
        pass