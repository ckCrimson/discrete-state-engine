import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from typing import Iterable

# ==========================================
# IMPORT YOUR DOMAIN & DOD COMPONENTS HERE
# ==========================================
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.generator.domain.utilities.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorUtility

# DOD Imports
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator

from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility

from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility


# ==========================================
# 1. THE PHYSICS RULES
# ==========================================
# 3-Fold Recurrent Branching (Forward, Forward-Left, Forward-Right)
def domain_3_fold_neighbors(state: State) -> Iterable[State]:
    x, y = state.vector[0], state.vector[1]
    return [
        State(np.array([x + 1.0, y + 1.0])),  # Forward-Left
        State(np.array([x + 1.0, y])),  # Straight
        State(np.array([x + 1.0, y - 1.0]))  # Forward-Right
    ]


@njit(cache=True, fastmath=True)
def dod_3_fold_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x + 1.0, y + 1.0],
        [x + 1.0, y],
        [x + 1.0, y - 1.0]
    ], dtype=np.float64)


# Biased Kernel: Favors moving straight (0.6) over turning (0.2)
def domain_biased_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    dy = s_i[1] - s_j[1]
    weight = 0.6 if dy == 0 else 0.2
    return np.array([weight], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def dod_biased_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    dy = s_i[1] - s_j[1]
    weight = 0.6 + 0.0j if dy == 0 else 0.2 + 0.0j
    return np.array([weight], dtype=np.complex128)


# ==========================================
# 2. THE BENCHMARK RUNNER
# ==========================================
def run_scaling_benchmark():
    step_counts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,27, 30,32,38,39 ,40,43,47, 50, 60, 70, 80, 90, 100]
    domain_times = []
    dod_times = []

    print("==================================================")
    print("      FDS SCALING BENCHMARK: OOP vs DOD           ")
    print("==================================================")
    print("Booting JIT Compilers...")

    # Vector Field Setup (2 Dimensions per field node)
    algebra = FieldAlgebra(dimensions=2, dtype=np.complex128)
    initial_states = np.array([[0.0, 0.0]], dtype=np.float64)
    initial_fields = np.array([[1.0 + 0.0j, 0.5 + 0.0j]], dtype=np.complex128)  # <--- Vector Field!

    topology_domain = Topology(reachable_func=domain_3_fold_neighbors, state_class=State, use_cache=True)
    mapper_domain = FieldMapper(algebra=algebra, state_class_ref=State)
    global_mapper = FieldMapper(algebra=algebra, state_class_ref=State)

    # Boot JIT once to ensure fair timing
    _ = dod_3_fold_neighbors(initial_states[0])
    _ = dod_biased_transition(initial_states[0], np.array([1.0, 1.0], dtype=np.float64))

    for steps in step_counts:
        print(f"\n--- Running Step Count: {steps} ---")

        mapper_domain = FieldMapper(algebra=algebra, state_class_ref=State)
        mapper_domain.set_fields_at([initial_states[0]], [initial_fields[0]])

        gen_data = GenericMarkovianFieldGeneratorData(
            mapper=mapper_domain, topology=topology_domain, transition_function=domain_biased_transition,
            maximum_step_baking=steps, max_size=50000, state_shape=(2,), implicit_norm=False, explicit_norm=False
        )

        t0 = time.perf_counter()
        final_states, _ = GenericMarkovianFieldGeneratorUtility.generate_multi_step_field(
            initial_mapper=mapper_domain, global_mapper=global_mapper, generator_data=gen_data, steps=steps
        )
        t_domain = time.perf_counter() - t0
        domain_times.append(t_domain)
        print(f"Domain OOP   | Time: {t_domain:.4f} s | States: {len(final_states)}")

        # ---------------------------------------------------------
        # DOD EXECUTION
        # ---------------------------------------------------------
        global_contract = FieldKernelDataContract(2, 2, algebra, State, None, 50000)
        global_field_cm = FieldComponentManager.create_from_raw(
            NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
            NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.complex128)
        )
        global_field_cm.fill(0.0 + 0.0j)

        topo_contract = TopologyKernelDataContract(dod_3_fold_neighbors, State, 50000, 2, np.float64)
        topology_cm = TopologyComponentManager.create_from_raw_data(
            topo_contract, NumbaTopologyStorage(topo_contract), NumbaTopologyTranslator(), NumbaTopologyUtility
        )

        gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=2)
        generator_cm = GeneratorComponentManager(
            gen_contract, NumbaComplexCSRGeneratorStorage(gen_contract), GenericGeneratorTranslator(),
            GenericGeneratorKernelUtility, dod_biased_transition
        )

        generator_cm.load_initial_state(initial_states, initial_fields)
        topology_cm.warmup([State(initial_states[0])], steps=steps + 2)
        generator_cm.inject_environment(topology_cm, global_field_cm)

        t0 = time.perf_counter()
        dod_states, _ = generator_cm.generate_steps(steps=steps)
        t_dod = time.perf_counter() - t0
        dod_times.append(t_dod)
        print(f"Numba DOD    | Time: {t_dod:.4f} s | States: {len(dod_states)}")

    # ==========================================
    # 3. PLOTTING THE BENCHMARK
    # ==========================================
    print("\nSaving Benchmark Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, domain_times, marker='o', color='red', linewidth=2, label='Domain (Pure OOP)')
    plt.plot(step_counts, dod_times, marker='s', color='blue', linewidth=2, label='DOD Engine (Numba/CSR)')

    plt.title("Performance Scaling: Domain OOP vs DOD Engine (3-Fold Recurrent Topology)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Execution Time (Seconds)")
    plt.yscale('log')  # Log scale is crucial to show the widening gap
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    save_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "scaling_benchmark.png"), dpi=300)
    plt.close()
    print("Benchmark complete! Check the output directory for the graph.")


if __name__ == "__main__":
    run_scaling_benchmark()