import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List

# --- Domain Imports ---
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility

# --- Component Manager Imports ---
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility


# ==========================================
# 1. THE PHYSICS RULES
# ==========================================
def move_1d_python(s: State):
    x = s.vector[0]
    return [State(np.array([x - 1], dtype=np.int32)), State(np.array([x + 1], dtype=np.int32))]


@njit(cache=True)
def move_1d_jit(state_vec: np.ndarray):
    x = state_vec[0]
    out_list = List()
    out_list.append(np.array([x - 1], dtype=np.int32))
    out_list.append(np.array([x + 1], dtype=np.int32))
    return out_list


# ==========================================
# 2. THE COMPARISON BENCHMARK
# ==========================================
def run_side_by_side_benchmark():
    print("🚀 Initializing Comparison Benchmark...")

    start_vec = np.array([0], dtype=np.int32)
    start_state = State(start_vec)
    step_values = [10, 50, 100, 200, 300, 400, 500]

    # --- SETUP DOMAIN ENGINE ---
    domain_universe = Topology(reachable_func=move_1d_python, state_class=State)

    # --- SETUP COMPONENT MANAGER ---
    # Setup contract explicitly with kwargs for clarity
    contract = TopologyKernelDataContract(
        neighbour_function=move_1d_jit,
        state_class_reference=State,
        initial_capacity=100_000,
        dimensions=1,
        vector_dtype=np.int32
    )

    # THE FIX: 4 strictly-typed arguments, and NumbaTopologyUtility passed as a class ref (no parenthesis)
    manager = TopologyComponentManager.create_from_raw_data(
        data_contract=contract,
        storage=NumbaTopologyStorage(contract),
        translator=NumbaTopologyTranslator(),
        utility=NumbaTopologyUtility
    )

    # --- WARMUP (PRE-BAKE GRAPH & JIT) ---
    print("🔥 Warming up JIT Engine...")
    manager.warmup([start_state], steps=max(step_values))
    # Prime the signature for raw vector return
    manager.get_reachable_multi_step_frontier(start_state, steps=1, return_state_class=False)

    domain_times = []
    manager_times = []

    for l in step_values:
        print(f"📊 Benchmarking l={l}...")

        # A. Domain Performance (Pythonic)
        t0 = time.perf_counter()
        _ = TopologyUtility.get_multi_step_reachable_frontier(domain_universe, start_state, l=l)
        t1 = time.perf_counter()
        domain_times.append((t1 - t0) * 1000)

        # B. Component Manager Performance (Numba)
        t0 = time.perf_counter()
        _ = manager.get_reachable_multi_step_frontier(start_state, steps=l, return_state_class=False)
        t1 = time.perf_counter()
        manager_times.append((t1 - t0) * 1000)

    # ==========================================
    # 3. PLOTTING THE COMPARISON
    # ==========================================
    plt.figure(figsize=(12, 7))
    plt.plot(step_values, domain_times, marker='o', color='blue', label='Domain Engine (Python Objects)', linewidth=2)
    plt.plot(step_values, manager_times, marker='s', color='red', label='Component Manager (Numba CSR)', linewidth=2)

    plt.yscale('log')  # Log scale helps visualize the order of magnitude difference
    plt.title('Performance Comparison: Domain vs. Component Manager', fontsize=14)
    plt.xlabel('Number of Steps ($l$)', fontsize=12)
    plt.ylabel('Time (milliseconds) - Log Scale', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()

    # Annotation
    plt.annotate('Order of Magnitude Speedup', xy=(300, manager_times[-3]), xytext=(300, domain_times[-3]),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    output_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\topology\plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "domain_vs_manager_comparison.png"), dpi=300)
    print(f"\n✅ Comparison Plot saved to: {output_dir}")
    plt.show()


if __name__ == "__main__":
    run_side_by_side_benchmark()