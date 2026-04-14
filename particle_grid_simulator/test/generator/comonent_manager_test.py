import os
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage

# ==========================================
# DOMAIN IMPORTS
# ==========================================
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData

# ==========================================
# COMPONENT MANAGER & DOD IMPORTS
# ==========================================
# Field
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator

# Topology
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility

# Generator
from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.generator.kernel.numba.storage.storage_v1 import NumbaCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.translator_v1 import NumbaGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.utility_v1 import NumbaGeneratorUtility


# ==========================================
# 1. PURE C-SPEED PHYSICS RULES
# ==========================================
@njit(cache=True, fastmath=True)
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)

# Domain rule (used strictly for the mock blueprint here)
def random_walker_neighbors(state: State):
    x, y = state.vector[0], state.vector[1]
    return [
        State(np.array([x, y + 1.0])),
        State(np.array([x, y - 1.0]))
        # ,State(np.array([x + 1.0, y + 1.0])),
        # State(np.array([x - 1.0, y - 1.0]))
    ]

# Hardware Rule (C-Speed)
@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    """Pure C-speed neighbor generation for the hardware graph builder."""
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0]
        ,[x, y - 1.0]
         ,[x + 1.0, y + 1.0]
        #, [x - 1.0, y - 1.0]
    ], dtype=np.float64)

# ==========================================
# 2. VISUALIZATION
# ==========================================
def plot_and_save_field(states: np.ndarray, fields: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    X, Y = states[:, 0], states[:, 1]
    Z = fields[:, 0]

    # --- ADDED GUARD: Prevent crash if field is totally empty ---
    if Z.max() <= 0:
        print("⚠️ Warning: Field mass is zero. Nothing to plot.")
        return

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        X, Y, c=Z, cmap='magma', marker='s', s=15,
        norm=mcolors.LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max())
    )
    plt.colorbar(sc, label='Field Mass (Log Scale)')
    plt.title("2D Random Walker Field Distribution (50 Steps - DOD Pipeline)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.2)

    file_path = os.path.join(save_dir, "dod_random_walker_50_steps.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to: {file_path}")
# ==========================================
# TEST EXECUTION
# ==========================================
def run_dod_random_walker_test():
    print("1. Configuring OOP Domain Blueprints...")
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)

    initial_states = np.array([[0.0, 0.0]], dtype=np.float64)
    initial_fields = np.array([[1.0]], dtype=np.float64)

    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=uniform_transition,
        maximum_step_baking=50,
        max_size=10000,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
    )

    print("2. Spinning up Hardware Component Managers...")

    # A. Static Math Bridge (Zero-Allocation Utility)
    # FIX: No parenthesis! Pass the class reference.
    field_math_bridge = FieldComponentManager.create_utility_cm(NumbaKernelFieldUtility)

    # B. Global Field (Vacuum)
    global_contract = FieldKernelDataContract(
        max_active_states=1000,
        state_dimensions=2,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State
    )

    global_field_cm = FieldComponentManager.create_from_raw(
        utility=NumbaKernelFieldUtility, # FIX: No parenthesis!
        contract=global_contract,
        storage=NumbaFieldKernelStorage(global_contract),
        translator=NumbaFieldTranslator(),
        states=np.empty((0, 2), dtype=np.float64),
        fields=np.empty((0, 1), dtype=np.float64)
    )
    global_field_cm.fill(1)
    # C. Topology Component Manager
    # FIX: Uses the actual real Topology Data Contract
    topology_contract = TopologyKernelDataContract(
        neighbour_function=hardware_random_walker_neighbors,
        state_class_reference=State,
        initial_capacity=10000,
        dimensions=2,
        vector_dtype=np.float64
    )

    # Use the static factory builder to spin it up natively
    topology_cm = TopologyComponentManager.create_from_raw_data(
        data_contract=topology_contract,
        storage=NumbaTopologyStorage(topology_contract),
        translator=NumbaTopologyTranslator(),
        utility=NumbaTopologyUtility # FIX: No parenthesis!
    )

    # D. Generator Component Manager (The Orchestrator)
    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)

    generator_cm = GeneratorComponentManager(
        contract=generator_contract,
        storage=NumbaCSRGeneratorStorage(generator_contract),
        translator=NumbaGeneratorTranslator(),
        utility=NumbaGeneratorUtility, # FIX: No parenthesis!
        transition_func=uniform_transition,
        math_utility_cm=field_math_bridge
    )

    print("3. Executing DOD Pipeline Sequence...")
    # 1. Seed the initial particle at [0.0, 0.0]
    generator_cm.load_initial_state(initial_states, initial_fields)

    # 2. CRITICAL FIX: Expand the Topology graph first!
    # Without this, the Generator sees 0 neighbors and stops instantly.
    print("   -> Expanding Topology Graph (50 steps)...")
    topology_cm.warmup([State(initial_states[0])], steps=50)

    print("3. Executing DOD Pipeline Sequence...")

    # 1. LOAD THE PARTICLE
    generator_cm.load_initial_state(initial_states, initial_fields)

    # 2. BUILD THE GRAPH (Do this BEFORE injecting)
    print("   -> Expanding Topology Graph (50 steps)...")
    topology_cm.warmup([State(initial_states[0])], steps=50)

    # 3. RE-INJECT (This ensures the Generator grabs the LATEST pointers)
    print("   -> Wiring Hardware Pointers...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    # 4. EXECUTE
    print("   -> Blasting 50 steps via Ping-Pong Buffer...")
    final_states, final_fields = generator_cm.generate_steps(steps=50)

    print("\n--- TEST COMPLETE ---")
    print(f"Total Unique Active States after 50 steps: {len(final_states)}")
    print(f"Total Field Mass: {np.sum(final_fields)}")

    max_idx = np.argmax(final_fields)
    print(f"Peak Field Value: {final_fields[max_idx][0]} at Coordinate: {final_states[max_idx]}")

    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    plot_and_save_field(final_states, final_fields, save_directory)


if __name__ == "__main__":
    run_dod_random_walker_test()