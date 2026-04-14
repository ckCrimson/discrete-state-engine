import os
import time
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
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator

from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility

from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.generator.kernel.numba.storage.storage_v1 import NumbaCSRGeneratorStorage

from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility


# ==========================================
# 1. PURE C-SPEED PHYSICS RULES
# ==========================================
@njit(cache=True, fastmath=True)
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


def random_walker_neighbors(state: State):
    x, y = state.vector[0], state.vector[1]
    return [
        State(np.array([x, y + 1.0])),
        State(np.array([x, y - 1.0]))
    ]


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0],
        [x, y - 1.0],
        [x + 1.0, y + 1.0],
        [x - 1.0, y + 1.0],
        [x - 1.0, y - 1.0],
        [x + 1, y ],
        [x - 1, y],
        [x + 1, y-1]
    ], dtype=np.float64)


def plot_and_save_field(states: np.ndarray, fields: np.ndarray, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    X, Y = states[:, 0], states[:, 1]
    Z = fields[:, 0]

    max_z = Z.max()
    if max_z <= 0:
        print("⚠️ Warning: Field mass is zero. Nothing to plot.")
        return

    positive_z = Z[Z > 0]
    min_z = positive_z.min() if len(positive_z) > 0 else 1e-10
    if min_z >= max_z: min_z = max_z * 0.99

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        X, Y, c=Z, cmap='magma', marker='s', s=15,
        norm=mcolors.LogNorm(vmin=min_z, vmax=max_z)
    )
    plt.colorbar(sc, label='Field Mass (Log Scale)')
    plt.title("2D Random Walker Field (50 Steps - V2 Generic Architecture)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.2)

    file_path = os.path.join(abs_save_dir, "generic_bedod_random_walker_50_steps.png")

    try:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"✅ Plot successfully saved to:\n   --> {file_path}")
    except Exception as e:
        print(f"❌ Failed to save plot. Error: {e}")


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
        max_size=100000,  # Matched to contract
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
    )

    print("2. Spinning up Hardware Component Managers...")
    global_contract = FieldKernelDataContract(
        max_active_states=100000,
        state_dimensions=2,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State,
        mapper_func=None
    )

    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaKernelFieldUtility, global_contract, NumbaFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64), np.empty((0, 1), dtype=np.float64)
    )
    # Ensure vacuum has mass for initial lookup
    global_field_cm.fill(1.0)

    topology_contract = TopologyKernelDataContract(
        hardware_random_walker_neighbors, State, 100000, 2, np.float64
    )
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, uniform_transition
    )

    print("\n===========================================")
    print("      TELEMETRY EXECUTION PIPELINE         ")
    print("===========================================")

    # 0. Initial Load
    generator_cm.load_initial_state(initial_states, initial_fields)

    # PHASE 1: TOPOLOGY WARMUP & REBUILD
    # CRITICAL: We time the rebuild because it translates Python sets to CSR Arrays
    print("   -> Expanding Topology Graph & Rebuilding CSR...")
    t_start_warmup = time.perf_counter()

    topology_cm.warmup([State(initial_states[0])], steps=50)

    t_end_warmup = time.perf_counter()
    warmup_time = (t_end_warmup - t_start_warmup) * 1000
    print(f"   [Phase 1] Topology Warmup/Rebuild: {warmup_time:.4f} ms")

    # PHASE 2: ENVIRONMENT INJECTION
    print("   -> Wiring Hardware Arrays & JIT Math Callables...")
    t_start_inject = time.perf_counter()

    # This calls your GenericGeneratorTranslator.bake_topology_field
    generator_cm.inject_environment(topology_cm, global_field_cm)

    t_end_inject = time.perf_counter()
    injection_time = (t_end_inject - t_start_inject) * 1000
    print(f"   [Phase 2] Environment Injection: {injection_time:.4f} ms")

    # PHASE 3: FIELD GENERATION (The Numba Loop)
    print("   -> Blasting 50 steps via Ping-Pong Buffer...")
    t_start_gen = time.perf_counter()

    final_states, final_fields = generator_cm.generate_steps(steps=50)

    t_end_gen = time.perf_counter()
    generation_time = (t_end_gen - t_start_gen) * 1000
    print(f"   [Phase 3] Field Generation: {generation_time:.4f} ms")

    print("===========================================")
    print(f"TOTAL PIPELINE TIME: {warmup_time + injection_time + generation_time:.4f} ms")
    print("===========================================\n")

    # Slicing the result to ensure zero-mass states aren't plotted
    mask = final_fields[:, 0] > 0
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    plot_and_save_field(final_states[mask], final_fields[mask], save_directory)


if __name__ == "__main__":
    run_dod_random_walker_test()