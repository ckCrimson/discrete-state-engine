import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit

# --- Engine Imports ---
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
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
# CONFIGURATION
# ==========================================
NUM_PARTICLES = 15
ITERATIONS = 60
STEPS = 30
PATH = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"


@njit(cache=True, fastmath=True)
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.ones(NUM_PARTICLES, dtype=np.float64)


@njit(cache=True, fastmath=True)
def hardware_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]], dtype=np.float64)


def random_walker_neighbors(state: State):
    return [State(v) for v in hardware_neighbors(state.vector)]


# ==========================================
# THE CLEAN OPERATOR
# ==========================================
@njit(fastmath=True)
def clean_evolution_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
    masses = gen_fields[:, 0]
    total_mass = np.sum(masses)

    if total_mass > 0:
        rand_val = np.random.random() * total_mass
        cumulative = 0.0
        for i in range(len(masses)):
            cumulative += masses[i]
            if rand_val <= cumulative:
                return gen_states[i]
    return state_vec


# ==========================================
# GIF LOGIC
# ==========================================
def save_simulation_gif(trajectories, u_coords, u_weights, filename):
    print(f"\nGenerating GIF: {filename}...")
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- VISUALIZE GLOBAL FIELD ("THE POT") ---
    # We use tricontourf to plot the baked universe and the field gradient
    X = u_coords[:, 0]
    Y = u_coords[:, 1]
    Z = u_weights[:, 0]  # Use channel 0 (since all channels share the same global pull)

    # Plot the background field. Darker color = stronger pull
    ax.tricontourf(X, Y, Z, levels=20, cmap='Purples', alpha=0.3)

    # Draw a faint boundary line around the edge of the baked topology
    ax.tricontour(X, Y, Z, levels=[Z.min(), Z.max()], colors='purple', alpha=0.5, linewidths=1.0)

    # --- PARTICLE PLOTTING ---
    colors = plt.cm.jet(np.linspace(0, 1, NUM_PARTICLES))
    lines = [ax.plot([], [], color=colors[p], alpha=0.8, linewidth=1.5, zorder=3)[0] for p in range(NUM_PARTICLES)]
    scats = [ax.scatter([], [], color=colors[p], s=40, edgecolors='black', zorder=4) for p in range(NUM_PARTICLES)]

    # Frame the camera around the actual trajectory path with some padding
    all_pts = trajectories.reshape(-1, 2)
    ax.set_xlim(all_pts[:, 0].min() - 5, all_pts[:, 0].max() + 5)
    ax.set_ylim(all_pts[:, 1].min() - 5, all_pts[:, 1].max() + 5)

    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    def update(frame):
        for p in range(NUM_PARTICLES):
            lines[p].set_data(trajectories[:frame + 1, p, 0], trajectories[:frame + 1, p, 1])
            scats[p].set_offsets(trajectories[frame: frame + 1, p])
        return lines + scats

    ani = FuncAnimation(fig, update, frames=len(trajectories), blit=True)
    ani.save(os.path.join(PATH, filename), writer=PillowWriter(fps=10))
    plt.close()
    print(f"GIF saved successfully in {(time.perf_counter() - t0):.2f}s.")

# ==========================================
# MAIN TEST SUITE
# ==========================================
def run_full_pipeline_batch_test():
    print("--- Running Multi-Channel Matrix Simulation ---")

    algebra = FieldAlgebra(dimensions=NUM_PARTICLES, dtype=np.float64)
    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)

    g_contract = FieldKernelDataContract(max_active_states=500000, state_dimensions=2, field_dimensions=NUM_PARTICLES,
                                         algebra=algebra, state_class_ref=State, mapper_func=None)
    t_contract = TopologyKernelDataContract(hardware_neighbors, State, 500000, 2, np.float64)

    gen_data = GenericMarkovianFieldGeneratorData(FieldMapper(algebra, State), topology, uniform_transition, STEPS,
                                                  500000, (2,), False, True)
    gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=NUM_PARTICLES)

    topology_cm = TopologyComponentManager.create_from_raw_data(t_contract, NumbaTopologyStorage(t_contract),
                                                                NumbaTopologyTranslator(), NumbaTopologyUtility)
    generator_cm = GeneratorComponentManager(gen_contract, NumbaCSRGeneratorStorage(gen_contract),
                                             GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                             uniform_transition)
    op_cm = OperatorComponentManager.create_raw(clean_evolution_kernel, NumbaOperatorUtility(), State)

    current_states_batch = np.random.randint(-5, 6, size=(NUM_PARTICLES, 2)).astype(np.float64)
    trajectories = np.zeros((ITERATIONS + 1, NUM_PARTICLES, 2))
    trajectories[0] = current_states_batch.copy()

    # --- PROFILING: WARMUP ---
    print("Pre-baking Deep Topology...")
    t_bake_start = time.perf_counter()
    topology_cm.warmup([State(np.array([0.0, 0.0]))], steps=200)

    u_coords = np.array(topology_cm.fast_refs.handle_map)
    base_weights = np.array([[max(0.01, 1.0 - (x / 15.0))] for x, y in u_coords], dtype=np.float64)
    u_weights = np.repeat(base_weights, NUM_PARTICLES, axis=1)

    global_field_cm = FieldComponentManager.create_from_raw(NumbaKernelFieldUtility, g_contract,
                                                            NumbaFieldKernelStorage(g_contract), NumbaFieldTranslator(),
                                                            u_coords, u_weights)
    generator_cm.inject_environment(topology_cm, global_field_cm)
    t_bake_total = time.perf_counter() - t_bake_start

    # --- PROFILING: JIT COMPILE ---
    print("Executing Dummy Compile...")
    t_jit_start = time.perf_counter()
    dummy_s = np.zeros((NUM_PARTICLES, 1, 2), dtype=np.float64)
    dummy_f = np.zeros((NUM_PARTICLES, 1, 1), dtype=np.float64)
    op_cm.evolve_batch_inplace(current_states_batch, dummy_s, dummy_f)
    # Also compile generator
    generator_cm.clear()
    generator_cm.load_initial_state(current_states_batch, np.eye(NUM_PARTICLES, dtype=np.float64))
    generator_cm.generate_steps(steps=1)
    t_jit_total = time.perf_counter() - t_jit_start

    # Reset for the real run
    seed_fields = np.eye(NUM_PARTICLES, dtype=np.float64)

    # --- PROFILING ACCUMULATORS ---
    time_load = 0.0
    time_gen = 0.0
    time_handoff = 0.0
    time_operator = 0.0

    print("Starting Evolution Loop...")
    t_loop_start = time.perf_counter()

    for i in range(ITERATIONS):
        # 1. Load & Clear
        t0 = time.perf_counter()
        generator_cm.clear()
        generator_cm.load_initial_state(current_states_batch, seed_fields)
        time_load += (time.perf_counter() - t0)

        # 2. Generator Math (The Multi-Channel C-Kernel)
        t1 = time.perf_counter()
        final_states, final_fields = generator_cm.generate_steps(steps=STEPS)
        time_gen += (time.perf_counter() - t1)

        # 3. Operator Hand-off (Python Slicing & Broadcasting)
        t2 = time.perf_counter()
        M = len(final_states)
        batch_gen_states = np.ascontiguousarray(np.broadcast_to(final_states, (NUM_PARTICLES, M, 2)))
        batch_gen_fields = np.empty((NUM_PARTICLES, M, 1), dtype=np.float64)
        for p in range(NUM_PARTICLES):
            batch_gen_fields[p, :, 0] = final_fields[:, p]
        time_handoff += (time.perf_counter() - t2)

        # 4. Operator Evolution
        t3 = time.perf_counter()
        op_cm.evolve_batch_inplace(current_states_batch, batch_gen_states, batch_gen_fields)
        trajectories[i + 1] = current_states_batch.copy()
        time_operator += (time.perf_counter() - t3)

    t_loop_total = time.perf_counter() - t_loop_start

    # --- PRINT REPORT ---
    print("\n" + "=" * 45)
    print("   PERFORMANCE PROFILING REPORT")
    print("=" * 45)
    print(f"Topology Setup & Bake  : {t_bake_total:.4f} s")
    print(f"JIT Compilation        : {t_jit_total:.4f} s")
    print("-" * 45)
    print(f"HOT LOOP TOTAL         : {t_loop_total:.4f} s ({ITERATIONS} iterations)")
    print(f"  Load & Clear         : {time_load:.4f} s ({(time_load / ITERATIONS) * 1000:.2f} ms/iter)")
    print(f"  Generator Expansion  : {time_gen:.4f} s ({(time_gen / ITERATIONS) * 1000:.2f} ms/iter)")
    print(f"  Python Array Slicing : {time_handoff:.4f} s ({(time_handoff / ITERATIONS) * 1000:.2f} ms/iter)")
    print(f"  Operator Sampling    : {time_operator:.4f} s ({(time_operator / ITERATIONS) * 1000:.2f} ms/iter)")
    print("=" * 45)

    save_simulation_gif(trajectories, u_coords, u_weights, "independent_particle_demo.gif")

if __name__ == "__main__":
    run_full_pipeline_batch_test()