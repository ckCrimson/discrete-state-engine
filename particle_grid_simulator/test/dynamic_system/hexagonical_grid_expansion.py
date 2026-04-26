import os
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors

# ==========================================
# COMPLEX DOD STORAGE & UTILITY
# ==========================================
from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility

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
# COMPONENT MANAGERS
# ==========================================
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator

from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility

from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator

from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility

# ==========================================
# CONFIGURATION
# ==========================================
NUM_PARTICLES = 1
STEPS = 12
ITERATIONS = 5
PATH = r"./plots"


# ==========================================
# HEXAGONAL PHYSICS RULES
# ==========================================
@njit(cache=True, fastmath=True)
def hex_scattering_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    d_in = int(s_j[2])
    d_out = int(s_i[2])
    turn_idx = (d_out - d_in) % 6

    alpha = 0.5
    beta = np.sqrt((1.0 - 2.25 * (alpha ** 2)) / 3.0)
    sq3_2 = (np.sqrt(3.0) / 2.0) * beta

    F_real = np.array([alpha, 0.75 * alpha, 0.25 * alpha, 0.0, 0.25 * alpha, 0.75 * alpha], dtype=np.float64)
    F_imag = np.array([0.0, sq3_2, sq3_2, 0.0, -sq3_2, -sq3_2], dtype=np.float64)

    c_weight = F_real[turn_idx] + 1j * F_imag[turn_idx]
    return np.array([c_weight], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def hardware_hex_neighbors(state_vec: np.ndarray) -> np.ndarray:
    q, r = state_vec[0], state_vec[1]
    dq = np.array([1.0, 0.0, -1.0, -1.0, 0.0, 1.0], dtype=np.float64)
    dr = np.array([0.0, -1.0, -1.0, 0.0, 1.0, 1.0], dtype=np.float64)

    out = np.empty((6, 3), dtype=np.float64)
    for d_out in range(6):
        out[d_out, 0] = q + dq[d_out]
        out[d_out, 1] = r + dr[d_out]
        out[d_out, 2] = float(d_out)
    return out


@njit(fastmath=True)
def quantum_collapse_batch_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
    amplitudes = gen_fields[:, 0]
    masses = amplitudes.real ** 2 + amplitudes.imag ** 2
    total_mass = np.sum(masses)

    if total_mass > 1e-12:
        rand_val = np.random.random() * total_mass
        cumulative = 0.0
        for i in range(len(masses)):
            cumulative += masses[i]
            if rand_val <= cumulative:
                return gen_states[i]
    return state_vec


# ==========================================
# 2D HEX PROJECTION & VISUALIZATION
# ==========================================
def animate_hex_trajectory(history_particles: np.ndarray, history_wave: list, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)

    num_frames = len(history_particles)
    num_particles = history_particles.shape[1]

    fig, ax = plt.subplots(figsize=(10, 10))

    all_q = history_particles[:, :, 0]
    all_r = history_particles[:, :, 1]
    all_x = all_q + 0.5 * all_r
    all_y = (np.sqrt(3.0) / 2.0) * all_r

    padding = STEPS + 2
    x_min, x_max = np.min(all_x) - padding, np.max(all_x) + padding
    y_min, y_max = np.min(all_y) - padding, np.max(all_y) + padding

    colors = plt.cm.hsv(np.linspace(0, 1, num_particles))

    def update(frame):
        print(f"   -> Rendering Frame {frame + 1}/{num_frames}...", end='\r')
        ax.clear()

        ax.set_facecolor('#050510')
        fig.patch.set_facecolor('#050510')
        ax.grid(True, linestyle='--', color='white', alpha=0.1)

        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(colors='white')
        ax.set_title(f"Hexagonal Quantum Walk Interference\n{ITERATIONS} Iterations, {STEPS} Steps/Iter", color='white')

        # 1. Update Wave
        active_states, active_masses = history_wave[frame]
        if len(active_states) > 0:
            coords_qr = active_states[:, 0:2]
            unique_qr, inverse_idx = np.unique(coords_qr, axis=0, return_inverse=True)
            agg_masses = np.bincount(inverse_idx, weights=active_masses)

            valid_mask = agg_masses > 1e-10
            if np.any(valid_mask):
                q = unique_qr[valid_mask, 0]
                r = unique_qr[valid_mask, 1]
                x_cart = q + 0.5 * r
                y_cart = (np.sqrt(3.0) / 2.0) * r

                # Using Inferno and a Linear scale makes the interference fringes pop
                ax.scatter(x_cart, y_cart, c=agg_masses[valid_mask], cmap='inferno', s=120, alpha=0.9,
                           norm=mcolors.Normalize(vmin=0.0, vmax=0.15), marker='h', edgecolors='none', zorder=2)

        # 2. Update Particles
        for p in range(num_particles):
            trail_q = history_particles[:frame + 1, p, 0]
            trail_r = history_particles[:frame + 1, p, 1]
            trail_x = trail_q + 0.5 * trail_r
            trail_y = (np.sqrt(3.0) / 2.0) * trail_r

            ax.plot(trail_x, trail_y, marker='', linestyle='-', linewidth=2.0, color=colors[p], alpha=0.6, zorder=5)
            ax.scatter([trail_x[-1]], [trail_y[-1]], color='#FFFFFF', edgecolor=colors[p], s=80, marker='o', zorder=6)

    print(f"\n🔄 Starting GIF Generation ({num_frames} frames)...")
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
    file_path = os.path.join(abs_save_dir, "hexagonal_quantum_walk.gif")
    ani.save(file_path, writer=PillowWriter(fps=12))
    print(f"\n✅ Hexagonal Quantum GIF rendered successfully! Saved to: {file_path}")


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_hexagonal_walk_test():
    print(f"1. Configuring Hexagonal DOD Blueprint (Steps: {STEPS}, Iterations: {ITERATIONS})...")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State), topology=topology, transition_function=hex_scattering_transition,
        maximum_step_baking=STEPS, max_size=500_000, state_shape=(3,), implicit_norm=False, explicit_norm=True
    )

    global_contract = FieldKernelDataContract(max_active_states=500_000, state_dimensions=3, field_dimensions=1,
                                              algebra=algebra, state_class_ref=State, mapper_func=None)
    topology_contract = TopologyKernelDataContract(hardware_hex_neighbors, State, 500_000, 3, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(topology_contract,
                                                                NumbaTopologyStorage(topology_contract),
                                                                NumbaTopologyTranslator(), NumbaTopologyUtility)

    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
                                             GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                             hex_scattering_transition)

    op_cm = OperatorComponentManager.create_raw(evolution_func=quantum_collapse_batch_kernel,
                                                utility=NumbaOperatorUtility(), state_class_ref=State)

    print("\n===========================================")
    print("      HEXAGONAL TELEMETRY PIPELINE         ")
    print("===========================================")

    current_states = np.zeros((NUM_PARTICLES, 3), dtype=np.float64)

    # 1. PRE-BAKE THE ENTIRE UNIVERSE ONCE
    print("   -> Baking Static Topology Graph (Radius: 65)...")
    total_radius = (STEPS * ITERATIONS) + 5
    topology_cm.warmup([State(s) for s in current_states], steps=total_radius)

    # 2. POPULATE THE GLOBAL FIELD ONCE
    u_coords = np.array(topology_cm.fast_refs.handle_map)
    u_weights = np.ones((len(u_coords), 1), dtype=np.complex128)

    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), u_coords, u_weights
    )

    # 3. INJECT THE ENVIRONMENT ONCE (Locks the CSR matrix mapping)
    generator_cm.inject_environment(topology_cm, global_field_cm)

    # 4. START THE LOOP
    total_frames = 1 + (ITERATIONS * STEPS) + ITERATIONS
    history_particles = np.zeros((total_frames, NUM_PARTICLES, 3), dtype=np.float64)
    history_wave = []

    frame_idx = 0
    history_particles[frame_idx] = current_states
    history_wave.append((np.empty((0, 3)), np.empty(0)))
    frame_idx += 1

    print("   -> Starting Continuous Observer Loop...")

    for i in range(ITERATIONS):
        generator_cm.clear()

        # Load the states mapped to the pre-baked environment
        particle_phases = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)
        generator_cm.load_initial_state(current_states, particle_phases)

        for step in range(STEPS):
            final_states, final_fields = generator_cm.generate_steps(steps=1)
            print(f"      Step {step + 1}/{STEPS} | Active States Expanded: {len(final_states)}", end='\r')

            history_particles[frame_idx] = current_states
            history_wave.append((final_states.copy(), np.abs(final_fields[:, 0]) ** 2))
            frame_idx += 1

        print("")

        # Collapse Wave
        M = len(final_states)
        if M > 0:
            b_s = np.ascontiguousarray(np.broadcast_to(final_states, (NUM_PARTICLES, M, 3)))
            b_f = np.ascontiguousarray(np.broadcast_to(final_fields, (NUM_PARTICLES, M, 1)))
            op_cm.evolve_batch_inplace(current_states, b_s, b_f)

        history_particles[frame_idx] = current_states
        history_wave.append((np.empty((0, 3)), np.empty(0)))
        frame_idx += 1

        print(f"   ✅ Iteration {i + 1}/{ITERATIONS} complete...")

    animate_hex_trajectory(history_particles, history_wave, PATH)


if __name__ == "__main__":
    run_hexagonal_walk_test()