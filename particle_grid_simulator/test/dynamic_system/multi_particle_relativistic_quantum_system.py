import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.colors as mcolors
import numba as nb

# ==========================================
# FDS DOMAIN & MANAGER IMPORTS
# ==========================================
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility
from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract

# ==========================================
# 1. SYSTEM CONFIGURATION (Fast Demo Baseline)
# ==========================================
NUM_PARTICLES = 5
STEPS = 7
ITERATIONS = 10
BOX_RADIUS = 20
PATH = "particle_grid_simulator/test/dynamic_system/plots"

DELTA_POS = np.array([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0]
], dtype=np.float64)


# ==========================================
# 2. THE KERNEL FACTORY
# ==========================================
def build_relativistic_kernels(deltas: np.ndarray):
    n_neighbors = len(deltas)
    norm_scalar = 1.0 / np.sqrt(n_neighbors)

    @nb.njit(fastmath=True)
    def dynamic_topology(state_vec: np.ndarray) -> np.ndarray:
        q, r = state_vec[0], state_vec[1]
        out = np.empty((n_neighbors, 3), dtype=np.float64)
        for d_out in range(n_neighbors):
            out[d_out, 0] = q + deltas[d_out, 0]
            out[d_out, 1] = r + deltas[d_out, 1]
            out[d_out, 2] = float(d_out)
        return out

    @nb.njit(fastmath=True)
    def dynamic_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
        d_in = s_j[2]
        d_out = s_i[2]

        if d_in == d_out:
            val = (2.0 / n_neighbors) - 1.0
        else:
            val = (2.0 / n_neighbors)

        return np.array([val + 0j], dtype=np.complex128)

    return dynamic_topology, dynamic_transition


@nb.njit(fastmath=True)
def sequential_relativistic_collapse(
        p_state: np.ndarray, global_states: np.ndarray, global_fields: np.ndarray,
        claimed_states: np.ndarray, num_claimed: int, deltas: np.ndarray, steps: int
) -> np.ndarray:
    n_neighbors = len(deltas)
    max_bfs_nodes = ((2 * steps + 1) ** 2) * n_neighbors + 100

    frontier = np.empty((max_bfs_nodes, 3), dtype=np.float64)
    frontier[0] = p_state
    read_idx = 0
    write_idx = 1

    for _ in range(steps):
        layer_end = write_idx
        while read_idx < layer_end:
            curr = frontier[read_idx]
            read_idx += 1
            for d in range(n_neighbors):
                nq, nr = curr[0] + deltas[d, 0], curr[1] + deltas[d, 1]
                nd = float(d)

                is_duplicate = False
                for i in range(write_idx):
                    if frontier[i, 0] == nq and frontier[i, 1] == nr and frontier[i, 2] == nd:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    frontier[write_idx, 0] = nq
                    frontier[write_idx, 1] = nr
                    frontier[write_idx, 2] = nd
                    write_idx += 1

    valid_probs = np.zeros(len(global_states), dtype=np.float64)
    total_prob = 0.0

    for g_idx in range(len(global_states)):
        g_state = global_states[g_idx]

        in_frontier = False
        for f_idx in range(write_idx):
            if frontier[f_idx, 0] == g_state[0] and frontier[f_idx, 1] == g_state[1] and frontier[f_idx, 2] == g_state[
                2]:
                in_frontier = True
                break

        is_claimed = False
        if in_frontier:
            for c_idx in range(num_claimed):
                if claimed_states[c_idx, 0] == g_state[0] and claimed_states[c_idx, 1] == g_state[1]:
                    is_claimed = True
                    break

        if in_frontier and not is_claimed:
            amp = global_fields[g_idx, 0]
            prob = amp.real ** 2 + amp.imag ** 2
            valid_probs[g_idx] = prob
            total_prob += prob

    if total_prob > 1e-15:
        r = np.random.random() * total_prob
        acc = 0.0
        for g_idx in range(len(global_states)):
            acc += valid_probs[g_idx]
            if r <= acc:
                return global_states[g_idx]

    return p_state


# ==========================================
# 3. INTEGRATION AND SYSTEM FLOW
# ==========================================
def run_relativistic_multi_particle():
    total_start_time = time.time()

    # --- PHASE 1: BAKING ---
    baking_start = time.time()
    topo_fn, trans_fn = build_relativistic_kernels(DELTA_POS)

    n_neighbors = len(DELTA_POS)
    MAX_CAPACITY = int(((2 * BOX_RADIUS + 1) ** 2) * n_neighbors * 1.5)

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State)

    gen_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=trans_fn,
        maximum_step_baking=STEPS,
        max_size=MAX_CAPACITY,
        state_shape=(3,),
        implicit_norm=False,
        explicit_norm=False
    )

    topo_contract = TopologyKernelDataContract(topo_fn, State, MAX_CAPACITY, 3, np.float64)
    topo_cm = TopologyComponentManager.create_from_raw_data(topo_contract, NumbaTopologyStorage(topo_contract),
                                                            NumbaTopologyTranslator(), NumbaTopologyUtility)

    gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=1)
    gen_cm = GeneratorComponentManager(gen_contract, NumbaComplexCSRGeneratorStorage(gen_contract),
                                       GenericGeneratorTranslator(), GenericGeneratorKernelUtility, trans_fn)

    particle_states = np.zeros((NUM_PARTICLES, 3), dtype=np.float64)
    for i in range(NUM_PARTICLES):
        particle_states[i, 0] = float((i - NUM_PARTICLES / 2) * 2)
        particle_states[i, 1] = 0.0
        particle_states[i, 2] = 0.0

    print(f"[*] Baking Bounded Environment (Box Radius {BOX_RADIUS}, Max Capacity {MAX_CAPACITY})...")
    topo_cm.warmup([State(np.array([0.0, 0.0, 0.0]))], steps=BOX_RADIUS)

    u_coords = np.array(topo_cm.fast_refs.handle_map)
    u_weights = np.ones((len(u_coords), 1), dtype=np.complex128)

    for i in range(len(u_coords)):
        x, y, _ = u_coords[i]
        if abs(x) >= BOX_RADIUS - 2 or abs(y) >= BOX_RADIUS - 2:
            u_weights[i, 0] = 0.0 + 0.0j

    field_contract = FieldKernelDataContract(
        max_active_states=MAX_CAPACITY,
        state_dimensions=3,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State,
        initial_capacity=MAX_CAPACITY
    )

    global_field_cm = FieldComponentManager.create_from_raw(NumbaComplexUtility, field_contract,
                                                            NumbaComplexFieldKernelStorage(field_contract),
                                                            NumbaFieldTranslator(), u_coords, u_weights)

    gen_cm.inject_environment(topo_cm, global_field_cm)
    baking_time = time.time() - baking_start
    print(f"    [+] Baking completed in {baking_time:.3f} seconds.\n")

    # --- PHASE 2: EXECUTION ---
    print(f"[*] Starting Relativistic Execution Loop ({ITERATIONS} iterations)...")
    exec_start = time.time()

    history_wave = []
    history_particles = []

    for iteration in range(ITERATIONS):
        gen_cm.clear()
        initial_phases = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)
        gen_cm.load_initial_state(particle_states, initial_phases)

        for s in range(STEPS):
            g_states, g_fields = gen_cm.generate_steps(1)
            history_wave.append((g_states.copy(), np.abs(g_fields[:, 0]) ** 2))
            history_particles.append(particle_states.copy())

        claimed_registry = np.zeros((NUM_PARTICLES, 2), dtype=np.float64)
        num_claimed = 0

        for k in range(NUM_PARTICLES):
            chosen_state = sequential_relativistic_collapse(
                particle_states[k], g_states, g_fields, claimed_registry, num_claimed, DELTA_POS, STEPS
            )
            claimed_registry[num_claimed, 0] = chosen_state[0]
            claimed_registry[num_claimed, 1] = chosen_state[1]
            num_claimed += 1
            particle_states[k] = chosen_state

        history_wave.append((np.empty((0, 3)), np.empty(0)))
        history_particles.append(particle_states.copy())

    exec_time = time.time() - exec_start
    print(f"    [+] Physics execution completed in {exec_time:.3f} seconds.\n")

    # --- PHASE 3: RENDERING ---
    print("[*] Dispatching to Animator...")
    anim_start = time.time()
    animate_relativistic_walk(history_wave, history_particles, DELTA_POS, STEPS, ITERATIONS, PATH)
    anim_time = time.time() - anim_start

    total_time = time.time() - total_start_time
    print(f"\n======================================")
    print(f"⏱️  PERFORMANCE PROFILE")
    print(f"======================================")
    print(f"Baking Setup : {baking_time:.3f} s")
    print(f"Physics Math : {exec_time:.3f} s")
    print(f"Video Render : {anim_time:.3f} s")
    print(f"--------------------------------------")
    print(f"Total Time   : {total_time:.3f} s")
    print(f"======================================")


def animate_relativistic_walk(history_wave, history_particles, deltas, steps, iterations, path):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate global bounds for consistent axis scaling
    all_states = []
    for w_states, _ in history_wave:
        if len(w_states) > 0:
            all_states.append(w_states)

    all_coords = np.concatenate(all_states)
    x_min, x_max = all_coords[:, 0].min() - 2, all_coords[:, 0].max() + 2
    y_min, y_max = all_coords[:, 1].min() - 2, all_coords[:, 1].max() + 2

    def update(frame):
        ax.clear()
        ax.set_facecolor('#050510')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        iter_idx = frame // (steps + 1)
        step_idx = frame % (steps + 1)

        # 1. Plot the Wave as a Vector Field (Momentum Currents)
        if step_idx < steps:
            w_states, w_masses = history_wave[frame]
            if len(w_states) > 0:
                q = w_states[:, 0]
                r = w_states[:, 1]
                # Extract the momentum integer and map it to spatial vectors
                d_in = w_states[:, 2].astype(int)
                U = deltas[d_in, 0]
                V = deltas[d_in, 1]

                # Quiver draws the arrows.
                # Color maps to probability, pointing in the direction of momentum.
                ax.quiver(q, r, U, V, w_masses,
                          cmap='inferno', pivot='mid',
                          norm=mcolors.LogNorm(vmin=1e-5, vmax=1.0),
                          scale=35, width=0.003, headwidth=4, alpha=0.9)

            ax.set_title(f"Quantum Momentum Currents | Iteration {iter_idx + 1} | Step {step_idx + 1}", color='white')
        else:
            ax.set_title(f"Relativistic Collapse | Iteration {iter_idx + 1}", color='cyan', fontweight='bold')

        # 2. Plot the Particles and their Trails
        p_states = history_particles[frame]
        p_states_history = np.array(history_particles[:frame + 1])

        colors = ['#00F0FF', '#FF007F', '#7FFF00', '#FFD700', '#FF4500',
                  '#9400D3', '#00FF7F', '#DC143C', '#1E90FF', '#FF1493']

        for i in range(len(p_states)):
            c = colors[i % len(colors)]
            ax.plot(p_states_history[:, i, 0], p_states_history[:, i, 1],
                    color=c, alpha=0.5, linewidth=2.5, linestyle=':', zorder=9)
            ax.scatter(p_states[i, 0], p_states[i, 1], color=c,
                       s=120, edgecolors='white', linewidth=1.5, zorder=10)

    total_frames = iterations * (steps + 1)
    anim = FuncAnimation(fig, update, frames=total_frames, blit=False)

    os.makedirs(path, exist_ok=True)
    mp4_path = os.path.join(path, "relativistic_field_analysis.mp4")
    gif_path = os.path.join(path, "relativistic_field_analysis.gif")

    print(f"    🎬 Rendering {total_frames} frames...")
    try:
        writer = FFMpegWriter(fps=10)
        anim.save(mp4_path, writer=writer)
        print(f"    ✅ Analysis video ready: {mp4_path}")
    except Exception as e:
        print(f"    FFMpeg failed, falling back to GIF... (Error: {e})")
        anim.save(gif_path, writer=PillowWriter(fps=10))
        print(f"    ✅ Analysis GIF ready: {gif_path}")

if __name__ == "__main__":
    run_relativistic_multi_particle()