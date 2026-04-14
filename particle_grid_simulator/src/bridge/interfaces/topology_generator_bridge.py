from typing import Protocol, Dict
from enum import Enum
import numpy as np

class TopologyFormat(Enum):
    """
    Standardized handshake formats so the receiving Storage
    knows exactly how to unpack the raw memory pointers.
    """
    CSR_ARRAYS = "csr_arrays"
    DENSE_MATRIX = "dense_matrix"
    SPATIAL_GRID = "spatial_grid"

class ITopologyGeneratorStorageBridge(Protocol):
    """
    CONTRACT: The secure, zero-copy data bus between Topology CM and Generator CM.
    Only passes memory references (pointers), never copies data.
    """
    @property
    def format_type(self) -> TopologyFormat:
        ...

    @property
    def is_static(self) -> bool:
        """
        If True, the orchestrator knows not to request this data again
        unless a major global event forces a re-bake.
        """
        ...

    @property
    def adjacency_data(self) -> Dict[str, np.ndarray]:
        """
        The raw C-array pointers.
        For CSR, expects keys: 'offsets', 'targets', 'state_coordinates'
        """
        ...