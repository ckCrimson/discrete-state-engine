import warnings
from typing import Any, Tuple, Callable, Optional, List
import numpy as np

# Assuming base manager and interfaces are imported from your architecture
from hpc_ecs_core.src.hpc_ecs_core.manager import BaseComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract, IGeneratorKernelStorage
from particle_grid_simulator.src.generator.iterfaces.translator import IGeneratorTranslator
from particle_grid_simulator.src.generator.iterfaces.utility import IGeneratorKernelUtility


class GeneratorComponentManager(BaseComponentManager):

    def __init__(
            self,
            contract: 'GeneratorKernelDataContract',
            storage: 'IGeneratorKernelStorage',
            translator: 'IGeneratorTranslator',
            utility: 'IGeneratorKernelUtility',
            transition_func: Callable,
            # --- THE BACKWARD COMPATIBILITY FIX ---
            math_utility_cm: Optional[Any] = None
    ) -> None:

        super().__init__(
            utility=utility,
            contract=contract,
            raw_storage=storage,
            translator=translator
        )

        if not self.is_static:
            self._contract = contract
            self._transition_func = transition_func

            # Save it so legacy V1 utilities don't break
            self._math_utility_cm = math_utility_cm

            # Warn the developer if they are using the legacy pattern
            if math_utility_cm is not None:
                warnings.warn(
                    "math_utility_cm is deprecated in the V2 Generic Architecture. "
                    "The Translator now extracts the math algebra directly from the "
                    "Global Field Manager.",
                    DeprecationWarning,
                    stacklevel=2
                )

            self._environment_ready = False

    # ==========================================
    # PHASE 1: LOAD THE SEED STATE
    # ==========================================

    def load_initial_state(self, states: np.ndarray, fields: np.ndarray) -> None:
        """
        Bakes the starting particles into Ping-Pong Buffer A.
        Must be called before environment injection so the Topology knows the origin points.
        """
        self._ensure_stateful()

        # The Translator handles the capacity checks and high-water marks
        self.translator.bake(self.fast_refs, (states, fields))

    # ==========================================
    # PHASE 2: INJECT AND WARMUP ENVIRONMENT
    # ==========================================

    def inject_environment(self, topology_cm: Any, global_field_cm: Any) -> None:
        """
        Warms up the Topology graph based on the seed state, then maps all
        environment pointers (CSR and Global Fields) directly into the FastRef.
        """
        self._ensure_stateful()

        if self.fast_refs.active_count_A == 0:
            raise RuntimeError(
                "Generator is empty. Must call load_initial_state() before injecting the environment."
            )

        # 1. Lazy Topology Warmup
        # Only expands the graph if another generator hasn't already baked it deep enough
        required_steps = self._contract.maximum_steps
        if getattr(topology_cm.fast_refs, 'steps_prepared', 0) < required_steps:
            seed_states = self.fast_refs.buffer_A_states[:self.fast_refs.active_count_A]
            topology_cm.prepare_graph(seed_states, steps=required_steps)

        # 2. Pointer Wiring (Zero Memory Allocation)
        self.translator.bake_topology_field(self.fast_refs, topology_cm, global_field_cm)
        self._environment_ready = True

    # ==========================================
    # PHASE 3: HARDWARE EXECUTION
    # ==========================================

    def generate_steps(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drives the multi-step C-kernel.
        """
        self._ensure_stateful()

        if not self._environment_ready:
            raise RuntimeError(
                "Environment pointers not wired! Call inject_environment() first."
            )

        # 1. Execute the pure C-speed loop
        winning_buffer = self.utility.execute_multi_step(
            fast_refs=self.fast_refs,
            steps=steps,
            transition_func=self._transition_func,
            math_utility=self._math_utility_cm,

            # --- THE FIX: Pass BOTH naming conventions ---
            # V2 Architecture Names
            do_implicit_norm=self._contract.intrinsic_norm,
            do_explicit_norm=self._contract.extrinsic_norm,

            # V1 Legacy Architecture Names
            # (V2 will harmlessly absorb these into **kwargs)
            intrinsic_norm=self._contract.intrinsic_norm,
            extrinsic_norm=self._contract.extrinsic_norm
        )

        # 2. Sync and return the zero-copy memory views
        return self.translator.sync_to_domain(self.fast_refs, winning_buffer)

    # ==========================================
    # PHASE 4: TRAJECTORY TELEMETRY
    # ==========================================
    # ==========================================
    # PHASE 4: TRAJECTORY TELEMETRY
    # ==========================================
    def generate_trajectory(self, steps: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Executes the simulation statelessly. Bypasses all Numba 1-step memory caching
        bugs by resetting to the initial state and calculating forward from zero for every frame.
        """
        self._ensure_stateful()

        if not self._environment_ready:
            raise RuntimeError("Environment pointers not wired! Call inject_environment() first.")

        history = []

        # 1. Secure the Initial Seed Data
        count = self.fast_refs.active_count_A
        seed_s = self.fast_refs.buffer_A_states[:count].copy()
        seed_f = self.fast_refs.buffer_A_fields[:count].copy()

        # Record Frame 0
        history.append((seed_s.copy(), seed_f.copy()))
        print(f"   [Physics] Frame 0 computed: {len(seed_s)} active particles.")

        # 2. Stateless Execution Loop
        for current_step in range(1, steps + 1):
            # Reset the engine completely to the origin
            self.load_initial_state(seed_s, seed_f)

            # Blast forward from zero to the current step natively in C
            curr_s, curr_f = self.generate_steps(steps=current_step)

            history.append((curr_s.copy(), curr_f.copy()))

            # THE PROOF: This logger will show the wave expanding
            print(f"   [Physics] Frame {current_step} computed: {len(curr_s)} active particles.")

        return history