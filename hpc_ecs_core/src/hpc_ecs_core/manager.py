from typing import Optional, Any, Type
from .interfaces import KernelDataContract, IKernelStorage, ITranslator, IKernelUtility, SyncState
from .store import ComponentStore
from .command_buffer import CommandBuffer


class BaseComponentManager:
    def __init__(
            self,
            utility: Any,  # Typed to IKernelUtility or similar
            contract: Optional[KernelDataContract] = None,
            raw_storage: Optional[IKernelStorage] = None,
            translator: Optional[ITranslator] = None,
            initial_data: Optional[Any] = None
    ) -> None:
        self.utility = utility

        # ---------------------------------------------------------
        # PATHWAY 1: STATIC / STATELESS MODE
        # If instantiated with ONLY a utility, bypass all storage overhead.
        # ---------------------------------------------------------
        if contract is None and raw_storage is None and translator is None:
            # FIX: Removed the class-level type(self)._static_utility assignment.
            # We strictly bind to the instance now.
            self.is_static = True
            return

        # ---------------------------------------------------------
        # PATHWAY 2: NORMAL / STATEFUL MODE
        # If the heavy objects are provided, build the full data store.
        # ---------------------------------------------------------
        self.is_static = False
        self.store = ComponentStore(contract, raw_storage, initial_data)
        self.translator = translator
        self.command_buffer = CommandBuffer()
        self.metadata = {}

        self.fast_refs = self.store.storage.fast_refs

        if initial_data:
            self.translator.bake(self.fast_refs, initial_data)
            self.store.sync_state = SyncState.CLEAN

    @classmethod
    def create_utility_cm(cls, utility_class: Type[Any]) -> 'BaseComponentManager':
        """
        THE DEDICATED CONSTRUCTOR.
        Bypasses __init__ completely to create an ultra-lightweight instance
        that only holds a reference to the static utility class.
        """
        # 1. Allocate memory for the object, but DO NOT call __init__
        instance = cls.__new__(cls)

        # 2. Inject only the bare minimum state
        instance.is_static = True
        instance.utility = utility_class  # Holds the CLASS, not an object!

        return instance

    # ==========================================
    # UNIVERSAL SAFETY GUARDS
    # Using fast, direct boolean evaluation
    # ==========================================

    def _ensure_stateful(self) -> None:
        if self.is_static:
            raise RuntimeError(f"Cannot perform stateful operations on a static {self.__class__.__name__} bridge.")

    def _ensure_static(self) -> None:
        if not self.is_static:
            raise RuntimeError(f"Raw bridge methods can only be called on a static {self.__class__.__name__} bridge.")

    # ==========================================
    # LIFECYCLE METHODS
    # ==========================================

    def commit_frame(self) -> None:
        # FIX: Replaced slow getattr() with fast, direct attribute access
        if self.is_static:
            return

        queue = self.command_buffer.queue
        if queue:
            self.translator.bake_incremental(self.fast_refs, queue)
            self.command_buffer.clear()
        self.store.sync_state = SyncState.CLEAN