import numpy as np
from numba import njit, prange
from typing import Any

from particle_grid_simulator.src.operator.interfaces.utility import IOperatorCMUtility, OperatorContext


# Assuming IOperatorUtility and OperatorContext are imported from the interface file

class NumbaOperatorUtility(IOperatorCMUtility):
    """
    The concrete Numba implementation.
    Handles the hardware-specific JIT templating and array manipulations.
    """
    _kernel_cache = {}

    @staticmethod
    def _get_compiled_kernel(num_context_arrays: int, inplace: bool):
        """The Dynamic JIT Template Factory (Variadic Template Equivalent)."""
        cache_key = (num_context_arrays, inplace)
        if cache_key in NumbaOperatorUtility._kernel_cache:
            return NumbaOperatorUtility._kernel_cache[cache_key]

        ctx_args = ", ".join([f"c_{j}" for j in range(num_context_arrays)])
        ctx_vals = ", ".join([f"c_{j}[i]" for j in range(num_context_arrays)])
        comma = ", " if num_context_arrays > 0 else ""

        if inplace:
            code = f"""
def _dynamic_kernel(func, primary, {ctx_args}):
    for i in prange(len(primary)):
        primary[i] = func(primary[i]{comma}{ctx_vals})
"""
        else:
            code = f"""
def _dynamic_kernel(func, out, primary, {ctx_args}):
    for i in prange(len(primary)):
        out[i] = func(primary[i]{comma}{ctx_vals})
"""
        local_scope = {}
        exec(code, {'prange': prange}, local_scope)
        compiled_kernel = njit(parallel=True, fastmath=True)(local_scope['_dynamic_kernel'])
        NumbaOperatorUtility._kernel_cache[cache_key] = compiled_kernel

        return compiled_kernel

    # --- Interface Fulfillments ---

    def evolve(self, func: Any, context: OperatorContext) -> Any:
        return func(context.primary_data, *context.context_data)

    def evolve_inplace(self, func: Any, context: OperatorContext) -> None:
        new_val = func(context.primary_data, *context.context_data)
        context.primary_data.update(new_val)

    def evolve_batch(self, func: Any, context: OperatorContext) -> np.ndarray:
        # We safely cast here because we are in the Numba-specific implementation
        primary_arr = context.primary_data
        out_states = np.empty_like(primary_arr)

        kernel = self._get_compiled_kernel(len(context.context_data), inplace=False)
        kernel(func, out_states, primary_arr, *context.context_data)

        return out_states

    def evolve_batch_inplace(self, func: Any, context: OperatorContext) -> None:
        kernel = self._get_compiled_kernel(len(context.context_data), inplace=True)
        kernel(func, context.primary_data, *context.context_data)