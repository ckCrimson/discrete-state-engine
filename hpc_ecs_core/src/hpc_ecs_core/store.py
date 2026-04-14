from typing import List, Any, Optional
from .interfaces import KernelDataContract, IKernelStorage, SyncState


class ComponentStore:
    def __init__(self, contract: KernelDataContract, storage: IKernelStorage,
                 high_level_data: Optional[Any] = None) -> None:
        self.contract = contract
        self.storage = storage

        # Merge the concepts: This IS your high-level data.
        self.domain_data: List[Any] = high_level_data if high_level_data is not None else []
        self.sync_state: SyncState = SyncState.CLEAN