# ==========================================
# 4. THE V2 UTILITY (Parallel Bare-Metal execution)
# ==========================================
from typing import Callable, Any

import numpy as np

from particle_grid_simulator.src.generator.iterfaces.utility import IGeneratorKernelUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.dense_storage import DenseGeneratorFastRef


@nb.njit(parallel=True, fastmath=True)
def _dense_generate_step(grid_in, grid_out, global_field_grid, deltas, grid_size):
    # 1. Clear the output grid (Parallelized)
    for q in nb.prange(grid_size):
        for r in range(grid_size):
            for d in range(8):
                grid_out[q, r, d] = 0.0 + 0.0j

    # 2. Execute Quantum Propagation (Parallelized across X-axis)
    for q in nb.prange(grid_size):
        for r in range(grid_size):
            for d_in in range(8):
                amp = grid_in[q, r, d_in]

                # Skip empty space (Massive speedup)
                if amp.real == 0.0 and amp.imag == 0.0:
                    continue

                # Apply 8-way Grover Transition and Shift
                for d_out in range(8):
                    # Grover Coin Math
                    val = (2.0 / 8.0) - 1.0 if d_in == d_out else (2.0 / 8.0)
                    new_amp = amp * val

                    # Shift Operator
                    nq = q + int(deltas[d_out, 0])
                    nr = r + int(deltas[d_out, 1])

                    # Boundary Check & Apply Global Potential
                    if 0 <= nq < grid_size and 0 <= nr < grid_size:
                        g_amp = global_field_grid[nq, nr]
                        if g_amp.real != 0.0 or g_amp.imag != 0.0:
                            grid_out[nq, nr, d_out] += new_amp * g_amp




class DenseGeneratorKernelUtility(IGeneratorKernelUtility):
    @staticmethod
    def execute_multi_step(
            fast_refs: 'DenseGeneratorFastRef',
            steps: int,
            transition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
            math_utility: Any,
            do_implicit_norm: bool,
            do_explicit_norm: bool,
            **kwargs: Any
    ) -> str:
        """
        Executes the L-step Ping-Pong generation loop entirely in dense contiguous memory.
        Satisfies the generic IGeneratorKernelUtility contract while bypassing
        dynamic function injection for maximum cache-locality speed.
        """
        grid_size = fast_refs.grid_A.shape[0]

        for _ in range(steps):
            if fast_refs.active_buffer == 0:
                _dense_generate_step(
                    fast_refs.grid_A,
                    fast_refs.grid_B,
                    fast_refs.global_field_grid,
                    fast_refs.deltas,
                    grid_size
                )
                fast_refs.active_buffer = 1
            else:
                _dense_generate_step(
                    fast_refs.grid_B,
                    fast_refs.grid_A,
                    fast_refs.global_field_grid,
                    fast_refs.deltas,
                    grid_size
                )
                fast_refs.active_buffer = 0

        # Sync interface expects 'A' or 'B' to know which buffer won the ping-pong
        return 'B' if fast_refs.active_buffer == 1 else 'A'