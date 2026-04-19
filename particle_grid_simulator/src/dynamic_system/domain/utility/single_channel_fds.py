from typing import Any

import numpy as np


class ISingleChannelFDSUtility:

    @staticmethod
    def evolve(operator_cm: Any, current_states: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> None:
        pass

class SingleChannelFDSUtility(ISingleChannelFDSUtility):
    """
    A clean pass-through utility. It satisfies the domain architectural pattern
    while leveraging the DOD Operator's native inplace C-speed execution.
    """
    @staticmethod
    def evolve(operator_cm: Any, current_states: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> None:
        operator_cm.evolve_batch_inplace(current_states, gen_states, gen_fields)