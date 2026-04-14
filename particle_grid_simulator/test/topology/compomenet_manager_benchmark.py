import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List

from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility


# ==========================================
# 1. THE PHYSICS RULE (JIT COMPILED)
# ==========================================
@njit(cache=True)
def move_1d_jit(state_vec: np.ndarray):
    x = state_vec[0]
    out_list = List()
    # 1D Random Walk Logic
    out_list.append(np.array([x - 1], dtype=np.int32))
    out_list.append(np.array([x + 1], dtype=np.int32))
    return out_list


# ==========================================
# 2. BENCHMARK SUITE
# ==========================================
def run_component_manager_benchmark():
    print("🚀 Initializing High-Performance Topology Engine...")

    # --- Setup Architecture ---
    data_contract = TopologyKernelDataContract(
        neighbour_function=move_1d_jit,
        state_class_reference=State,
        initial_capacity=1_000_000,  # Large capacity to avoid resizing during benchmark
        dimensions=1,
        vector_dtype=np.int32
    )

    # THE FIX: create_from_raw_data strictly takes 4 arguments now.
    # It safely unpacks the class references and physics rules from the contract.
    manager = TopologyComponentManager.create_from_raw_data(
        data_contract=data_contract,
        storage=NumbaTopologyStorage(data_contract),
        translator=NumbaTopologyTranslator(),
        utility=NumbaTopologyUtility  # Stateless class reference passed perfectly
    )

    start_state = State(np.array([0], dtype=np.int32))

    # --- WARMUP (Outside Timer) ---
    # We warmup to the maximum expected step count to pre-bake the graph
    # and absorb the 5-second JIT compilation tax.
    max_steps = 1000
    print(f"🔥 Warming up engine...")

    # 1. Bake the graph
    manager.warmup([start_state], steps=max_steps)

    # 2. TRIGGER THE EXACT KERNEL SIGNATURE
    # We run a 1-step query with return_state_class=False
    # so Numba compiles the 'False' path before the timer starts.
    manager.get_reachable_multi_step_frontier(start_state, steps=1, return_state_class=False)

    # --- Scaling Test Parameters ---
    step_values = [10, 50, 100, 200, 500, 800, 1000]
    execution_times = []
    frontier_sizes = []

    print("\n⏱️ Starting Hot-Loop Benchmark...")

    for l in step_values:
        # Measure only the query time
        t0 = time.perf_counter()

        frontier = manager.get_reachable_multi_step_frontier(
            start_state,
            steps=l,
            return_state_class=False  # Measuring raw kernel + array performance
        )

        t1 = time.perf_counter()

        exec_time_ms = (t1 - t0) * 1000
        execution_times.append(exec_time_ms)
        frontier_sizes.append(len(frontier))

        print(f"Steps: {l:<5} | Frontier: {len(frontier):<6} | Time: {exec_time_ms:.4f} ms")

    # --- 3. PLOTTING ---
    plt.figure(figsize=(12, 7))
    plt.plot(step_values, execution_times, marker='s', color='red', linewidth=2, label='Component Manager (JIT)')

    # Annotations
    for i, txt in enumerate(execution_times):
        plt.annotate(f"{txt:.3f}ms", (step_values[i], execution_times[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.title('Component Manager Scaling: Query Time vs Steps (l)', fontsize=14)
    plt.xlabel('Number of Steps (l)', fontsize=12)
    plt.ylabel('Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save Path
    output_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\topology\plots"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "manager_scaling_jit.png")

    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    print(f"\n✅ Benchmark Plot saved to: {file_path}")
    plt.show()


if __name__ == "__main__":
    run_component_manager_benchmark()