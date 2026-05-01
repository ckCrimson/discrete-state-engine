import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from numba import njit

# --- Engine Imports ---
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.storage_v1 import NumbaCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility


# ==========================================
# PURE JIT KERNELS
# ==========================================
@njit(fastmath=True)
def hardware_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]], dtype=np.float64)


@njit(fastmath=True)
def compiled_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


@njit(fastmath=True)
def clean_evolution_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
    masses = gen_fields.flatten()
    total_mass = np.sum(masses)
    if total_mass > 1e-9:
        rand_val = np.random.random() * total_mass
        cumulative = 0.0
        for i in range(len(masses)):
            cumulative += masses[i]
            if rand_val <= cumulative:
                return gen_states[i]
    return state_vec


# ==========================================
# THE PIPELINE
# ==========================================
def run_full_pipeline_batch_test():
    BOX_SIZE, NUM_PARTICLES, ITERATIONS, STEPS = 10, 15, 40, 4
    CAP = 200000
    SAVE_DIR = Path(r"particle_grid_simulator\test\operator\plots")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Components
    algebra = FieldAlgebra(1, np.float64)
    topology_obj = Topology(lambda s: [State(v) for v in hardware_neighbors(s.vector)], State, True)

    t_con = TopologyKernelDataContract(hardware_neighbors, State, CAP, 2, np.float64)
    t_cm = TopologyComponentManager.create_from_raw_data(t_con, NumbaTopologyStorage(t_con), NumbaTopologyTranslator(),
                                                         NumbaTopologyUtility)

    # 2. Map Environment properly using Basin
    print("[1/3] Mapping environment...")
    u_coords_list = t_cm.get_reachable_multi_step_basin(state_in=np.array([0.0, 0.0]), steps=BOX_SIZE * 2 + 5)
    u_coords = np.array(u_coords_list)

    u_weights = np.zeros((len(u_coords), 1))
    for idx, c in enumerate(u_coords):
        if abs(c[0]) <= BOX_SIZE and abs(c[1]) <= BOX_SIZE:
            u_weights[idx, 0] = 2.0 if c[0] < 0 else 1.0

    # 3. Setup Decoupled Field and Generator
    g_con = FieldKernelDataContract(max_active_states=CAP, state_dimensions=2, field_dimensions=1, algebra=algebra,
                                    state_class_ref=State)
    f_cm = FieldComponentManager.create_from_raw(NumbaKernelFieldUtility, g_con, NumbaFieldKernelStorage(g_con),
                                                 NumbaFieldTranslator(), u_coords, u_weights)

    gen_data = GenericMarkovianFieldGeneratorData(
        FieldMapper(algebra, State), topology_obj, compiled_transition, STEPS, CAP, (2,), False, False
    )
    g_cm = GeneratorComponentManager(GeneratorKernelDataContract.from_domain(gen_data, 1),
                                     NumbaCSRGeneratorStorage(GeneratorKernelDataContract.from_domain(gen_data, 1)),
                                     GenericGeneratorTranslator(), GenericGeneratorKernelUtility, compiled_transition,)
    g_cm.inject_environment(t_cm, f_cm)
    op_cm = OperatorComponentManager.create_raw(clean_evolution_kernel, NumbaOperatorUtility(), State)

    # 4. Evolution
    states = np.zeros((NUM_PARTICLES, 2), dtype=np.float64)
    trajectories, field_history = [states.copy()], []

    print("[2/3] Starting Evolution Loop...")
    for i in range(ITERATIONS):
        # We work on a fresh copy and manually write back to ensure Numba sees the updates
        current_step_states = states.copy()
        tick_viz = [[] for _ in range(STEPS)]

        for p in range(NUM_PARTICLES):
            # Contiguous buffer for the specific particle
            p_buf = np.ascontiguousarray(current_step_states[p:p + 1])

            g_cm.clear()
            g_cm.load_initial_state(p_buf, np.ones((1, 1)))

            # Step expansion
            for s in range(STEPS):
                gs, gf = g_cm.generate_steps(1)
                tick_viz[s].append(gs.copy())

            # Jump
            if len(gs) > 0:
                op_cm.evolve_batch_inplace(p_buf, gs.reshape(1, len(gs), 2), gf.reshape(1, len(gs), 1))

            current_step_states[p] = p_buf[0]

        states = current_step_states
        field_history.append([np.vstack(v) for v in tick_viz])
        trajectories.append(states.copy())

        if (i + 1) % 10 == 0:
            print(f"      -> Iteration {i + 1} | Mean X: {np.mean(states[:, 0]):.2f}")

    print("[3/3] Saving Visualization...")
    render_gif(np.array(trajectories), u_coords, u_weights, field_history, BOX_SIZE, NUM_PARTICLES, STEPS, ITERATIONS,
               SAVE_DIR)


def render_gif(trajectories, u_coords, u_weights, field_history, BOX_SIZE, NUM_PARTICLES, STEPS, ITERATIONS, SAVE_DIR):
    fig, ax = plt.subplots(figsize=(8, 8))
    X, Y, Z = u_coords[:, 0], u_coords[:, 1], u_weights[:, 0]
    ax.scatter(X[Z > 0], Y[Z > 0], c=Z[Z > 0], cmap='Purples', marker='s', s=130, alpha=0.15)
    ax.add_patch(patches.Rectangle((-BOX_SIZE - 0.5, -BOX_SIZE - 0.5), BOX_SIZE * 2 + 1, BOX_SIZE * 2 + 1, linewidth=2,
                                   edgecolor='purple', fill=False))

    gen_field_scat = ax.scatter([], [], c='orange', marker='s', s=130, alpha=0.4, zorder=3)
    colors = plt.cm.jet(np.linspace(0, 1, NUM_PARTICLES))
    lines = [ax.plot([], [], color=colors[p], alpha=0.6, lw=2)[0] for p in range(NUM_PARTICLES)]
    scats = [ax.scatter([], [], color=colors[p], s=50, edgecolors='black', zorder=5) for p in range(NUM_PARTICLES)]

    ax.set_xlim(-BOX_SIZE - 2, BOX_SIZE + 2);
    ax.set_ylim(-BOX_SIZE - 2, BOX_SIZE + 2)

    def update(frame):
        t, sub = frame // (STEPS + 1), frame % (STEPS + 1)
        if t >= len(field_history): return lines + scats + [gen_field_scat]
        tail = max(0, t - 2)
        if sub < STEPS:
            gen_field_scat.set_offsets(field_history[t][sub])
            for p in range(NUM_PARTICLES):
                scats[p].set_offsets(trajectories[t, p])
                lines[p].set_data(trajectories[tail:t + 1, p, 0], trajectories[tail:t + 1, p, 1])
        else:
            gen_field_scat.set_offsets(np.empty((0, 2)))
            for p in range(NUM_PARTICLES):
                scats[p].set_offsets(trajectories[t + 1, p])
                lines[p].set_data(trajectories[tail:t + 2, p, 0], trajectories[tail:t + 2, p, 1])
        return lines + scats + [gen_field_scat]

    ani = FuncAnimation(fig, update, frames=ITERATIONS * (STEPS + 1), blit=True)
    ani.save(SAVE_DIR / "stochastic_overlapping_fields.gif", writer=PillowWriter(fps=15))
    plt.close()


if __name__ == "__main__":
    run_full_pipeline_batch_test()