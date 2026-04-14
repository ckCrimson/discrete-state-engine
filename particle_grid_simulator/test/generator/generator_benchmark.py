import time
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Iterable

# Your exact imports
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.generator.domain.utilities.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorUtility
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.domain.topology_domain import Topology


# 1. The Topology Rule
def random_walker_neighbors(state: State) -> Iterable[State]:
    x, y = state.vector[0], state.vector[1]
    return [
        State(np.array([x, y + 1.0])),        # <0, 1>
        State(np.array([x, y - 1.0])),        # <0, -1>
        State(np.array([x + 1.0, y + 1.0])),  # <1, 1>
        State(np.array([x - 1.0, y - 1.0]))   # <-1, -1>
    ]


# 2. The Transition Rule
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


# ==========================================
# BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    print("--- INITIALIZING ENVIRONMENT ---")
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)

    mapper = FieldMapper(algebra=algebra, state_class_ref=State)
    mapper.set_fields_at([np.array([0.0, 0.0])], [np.array([1.0])])

    global_mapper = FieldMapper(algebra=algebra, state_class_ref=State)

    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)

    # Capacity set to 25,000 to safely cover the max bounds of 100 steps without reallocating arrays.
    max_capacity = 25000

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=mapper,
        topology=topology,
        transition_function=uniform_transition,
        maximum_step_baking=100,
        max_size=max_capacity,
        state_shape=(2,),
        implicit_norm=True,
        explicit_norm=False
    )

    # ==========================================
    # 1. WARM-UP (Compiling & Baking)
    # ==========================================
    print("\n--- WARM-UP PHASE ---")
    print("Running 100 steps to trigger Numba JIT and populate Topology Cache...")
    t0 = time.perf_counter()
    GenericMarkovianFieldGeneratorUtility.generate_multi_step_field(
        initial_mapper=mapper,
        global_mapper=global_mapper,
        generator_data=generator_data,
        steps=100
    )
    t1 = time.perf_counter()
    print(f"Warm-up complete in: {t1 - t0:.4f} seconds")

    # ==========================================
    # 2. BENCHMARKING 10 to 100 STEPS
    # ==========================================
    print("\n--- BENCHMARK PHASE ---")
    step_counts = list(range(10, 101, 10))
    execution_times_ms = []

    for steps in step_counts:
        t_start = time.perf_counter()

        final_states, final_fields = GenericMarkovianFieldGeneratorUtility.generate_multi_step_field(
            initial_mapper=mapper,
            global_mapper=global_mapper,
            generator_data=generator_data,
            steps=steps
        )

        t_end = time.perf_counter()
        time_taken_ms = (t_end - t_start) * 1000  # Convert to milliseconds
        execution_times_ms.append(time_taken_ms)

        print(f"Steps: {steps:<3} | Active States: {len(final_states):<5} | Time: {time_taken_ms:.3f} ms")

    # ==========================================
    # 3. PLOTTING
    # ==========================================
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plotting Time vs Steps
    plt.plot(step_counts, execution_times_ms, marker='o', linestyle='-', color='#00aaff', linewidth=2, markersize=8)
    plt.fill_between(step_counts, execution_times_ms, color='#00aaff', alpha=0.1)

    plt.title("Field Generator Performance Benchmark (10 to 100 Steps)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Simulation Steps", fontsize=12)
    plt.ylabel("Execution Time (Milliseconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 16.6ms Target Line (60 FPS)
    plt.axhline(y=16.66, color='r', linestyle=':', label='60 FPS Target (16.6ms)')
    plt.legend()

    file_path = os.path.join(save_directory, "performance_benchmark_domain.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nBenchmark complete. Plot saved to: {file_path}")


if __name__ == "__main__":
    run_benchmark()