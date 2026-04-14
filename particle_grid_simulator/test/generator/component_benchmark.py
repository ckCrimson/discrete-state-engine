import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# ==========================================
# 1. ARCHITECTURE IMPORTS
# ==========================================
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData

# Component Managers & DOD
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
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
from particle_grid_simulator.src.generator.kernel.numba.translator.translator_v1 import NumbaGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.utility_v1 import NumbaGeneratorUtility


# ==========================================
# 2. PHYSICS RULES (C-SPEED)
# ==========================================
@njit(cache=True, fastmath=True)
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0], [x, y - 1.0],
        [x + 1.0, y], [x - 1.0, y]
    ], dtype=np.float64)


def random_walker_neighbors(state: State):
    # Dummy for domain-level init
    return []


# ==========================================
# 3. BENCHMARK SUITE
# ==========================================
def run_full_pipeline_benchmark():
    overall_t0 = time.perf_counter()
    print("🚀 Initializing High-Performance DOD Pipeline...")

    # --- A. Setup Contracts ---
    MAX_CAPACITY = 250_000
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)

    # Topology
    top_contract = TopologyKernelDataContract(
        neighbour_function=hardware_random_walker_neighbors,
        state_class_reference=State,
        initial_capacity=MAX_CAPACITY,
        dimensions=2,
        vector_dtype=np.float64
    )

    # Global Field
    field_contract = FieldKernelDataContract(
        max_active_states=MAX_CAPACITY,
        state_dimensions=2,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State
    )

    # --- B. Instantiate Managers ---
    topology_cm = TopologyComponentManager.create_from_raw_data(
        data_contract=top_contract,
        storage=NumbaTopologyStorage(top_contract),
        translator=NumbaTopologyTranslator(),
        utility=NumbaTopologyUtility
    )

    global_field_cm = FieldComponentManager.create_from_raw(
        utility=NumbaKernelFieldUtility,
        contract=field_contract,
        storage=NumbaFieldKernelStorage(field_contract),
        translator=NumbaFieldTranslator(),
        states=np.empty((0, 2), dtype=np.float64),
        fields=np.empty((0, 1), dtype=np.float64)
    )
    global_field_cm.fast_refs.field_array.fill(1.0)
    global_field_cm.fast_refs.normalized_field_array.fill(1.0)

    dummy_topology = Topology(reachable_func=random_walker_neighbors, state_class=State)
    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=dummy_topology,
        transition_function=uniform_transition,
        maximum_step_baking=150,
        max_size=MAX_CAPACITY,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
    )

    gen_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    math_bridge = FieldComponentManager.create_utility_cm(NumbaKernelFieldUtility)

    generator_cm = GeneratorComponentManager(
        contract=gen_contract,
        storage=NumbaCSRGeneratorStorage(gen_contract),
        translator=NumbaGeneratorTranslator(),
        utility=NumbaGeneratorUtility,
        transition_func=uniform_transition,
        math_utility_cm=math_bridge
    )

    # --- C. Injection & Warmup ---
    print("\n🔥 --- COMMENCING HEAVY INITIALIZATION PHASE --- 🔥")

    # 1. Load Initial State
    initial_states = np.array([[0.0, 0.0]], dtype=np.float64)
    initial_fields = np.array([[1.0]], dtype=np.float64)
    generator_cm.load_initial_state(initial_states, initial_fields)

    # 2. Topology Graph Build (CRITICAL FOR O(1) GENERATOR)
    print("   -> 1/3: Expanding Topology Graph (150 steps)...")
    topo_t0 = time.perf_counter()
    topology_cm.warmup([State(initial_states[0])], steps=150)
    topo_t1 = time.perf_counter()
    print(f"      [Done] Graph populated in {(topo_t1 - topo_t0):.4f} seconds.")

    # 3. Inject Environment
    print("   -> 2/3: Wiring Memory Pointers...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    # 4. Forcing Numba LLVM Compilation
    print("   -> 3/3: Forcing Numba LLVM Compilation (CPU will freeze)...")
    jit_t0 = time.perf_counter()
    generator_cm.generate_steps(steps=1)
    jit_t1 = time.perf_counter()
    print(f"      [Done] C-Kernel compiled in {(jit_t1 - jit_t0):.4f} seconds.")

    overall_t1 = time.perf_counter()
    print(f"\n✅ TOTAL PIPELINE INITIALIZATION TIME: {(overall_t1 - overall_t0):.4f} seconds.\n")

    # --- D. Hot Loop Scaling Test ---
    step_values = list(range(1, 150, 10))
    execution_times = []
    frontier_sizes = []

    print(f"⏱️ Starting O(1) Scaling Test (1 to 150 steps)...")

    for l in step_values:
        generator_cm.load_initial_state(initial_states, initial_fields)

        t0 = time.perf_counter()
        final_states, _ = generator_cm.generate_steps(steps=l)
        t1 = time.perf_counter()

        exec_time_ms = (t1 - t0) * 1000
        execution_times.append(exec_time_ms)
        frontier_sizes.append(len(final_states))

        print(f"Steps: {l:<4} | Frontier: {len(final_states):<6} | Time: {exec_time_ms:>8.4f} ms")

    # --- E. Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Steps (l)', fontsize=12)
    ax1.set_ylabel('Time (milliseconds)', color='red', fontsize=12)
    ax1.plot(step_values, execution_times, color='red', marker='s', linewidth=2, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Active State Count', color='blue', fontsize=12)
    ax2.plot(step_values, frontier_sizes, color='blue', linestyle='--', alpha=0.7, label='Frontier Size')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('DOD Engine Performance: O(1) Field Generation (2D Random Walker)', fontsize=14)
    fig.tight_layout()

    # Change this line at the bottom of the benchmark script
    output_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "generator_scaling_150_o1.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    run_full_pipeline_benchmark()