import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
NUM_PARTICLES = 10
STEPS = 5
ITERATIONS = 6
PATH = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"


# ==========================================
# 4D PHYSICS RULES (x, y, px, py)
# ==========================================
@njit(cache=True, fastmath=True)
def momentum_phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """
    s_j = [x_old, y_old, px_old, py_old]
    s_i = [x_new, y_new, px_new, py_new]
    """
    px_old, py_old = s_j[2], s_j[3]
    px_new, py_new = s_i[2], s_i[3]

    # Deviation from previous momentum induces the phase
    dp_x = px_new - px_old
    dp_y = py_new - py_old

    theta = np.arctan2(dp_y, dp_x)

    # Unitary scaling: 8 neighbors = scale by 1/sqrt(8) to conserve probability
    amplitude = 1.0 / np.sqrt(8.0)
    c_weight = amplitude * (np.cos(theta) + 1j * np.sin(theta))

    return np.array([c_weight], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def hardware_momentum_neighbors(state_vec: np.ndarray) -> np.ndarray:
    """
    Takes a 4D state and returns the 8 possible new 4D states.
    The new momentum is simply the spatial step taken.
    """
    x, y, px, py = state_vec[0], state_vec[1], state_vec[2], state_vec[3]

    # The 8 spatial directions
    moves = np.array([
        [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0],
        [1.0, 0.0], [-1.0, 0.0], [1.0, -1.0], [-1.0, 1.0]
    ], dtype=np.float64)

    out = np.empty((8, 4), dtype=np.float64)
    for i in range(8):
        dx, dy = moves[i, 0], moves[i, 1]
        out[i, 0] = x + dx  # New X
        out[i, 1] = y + dy  # New Y
        out[i, 2] = dx  # New PX
        out[i, 3] = dy  # New PY
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
# 2D PROJECTION & VISUALIZATION
# ==========================================
def animate_momentum_trajectory(history_particles: np.ndarray, history_wave: list, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    num_frames = len(history_particles)
    num_particles = history_particles.shape[1] # Dynamically get particle count

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#050510')
    ax.grid(True, linestyle='--', color='white', alpha=0.1)
    ax.set_title(f"4D Phase Space Collapse (Projected to 2D)\n{ITERATIONS} Iterations, {STEPS} Steps/Iter | {num_particles} Particles")
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_aspect('equal', adjustable='datalim')

    # Dynamic Camera Bounds
    x_min, x_max = np.min(history_particles[:, :, 0]), np.max(history_particles[:, :, 0])
    y_min, y_max = np.min(history_particles[:, :, 1]), np.max(history_particles[:, :, 1])
    padding = STEPS + 2
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Wave visualizer (Aggregates all particles automatically)
    wave_scat = ax.scatter([], [], c=[], cmap='magma', s=35, alpha=0.9, edgecolors='none', zorder=2,
                           norm=mcolors.LogNorm(vmin=1e-5, vmax=1.0))

    # THE FIX: Create unique colors and plot objects for EVERY particle
    colors = plt.cm.hsv(np.linspace(0, 1, num_particles))
    particle_lines = [
        ax.plot([], [], marker='', linestyle='-', linewidth=1.5, color=colors[p], alpha=0.6, zorder=5)[0]
        for p in range(num_particles)
    ]
    particle_dots = [
        ax.scatter([], [], color='#FFFFFF', edgecolor=colors[p], s=80, marker='o', zorder=6)
        for p in range(num_particles)
    ]

    def update(frame):
        # 1. Update Wave
        active_states, active_masses = history_wave[frame]
        if len(active_states) > 0:
            spatial_coords = active_states[:, 0:2]
            wave_scat.set_offsets(spatial_coords)
            wave_scat.set_array(active_masses)
        else:
            wave_scat.set_offsets(np.empty((0, 2)))

        # 2. Update All Particles
        for p in range(num_particles):
            trail_data = history_particles[:frame + 1, p, 0:2]
            current_pos = history_particles[frame, p, 0:2]

            particle_lines[p].set_data(trail_data[:, 0], trail_data[:, 1])
            particle_dots[p].set_offsets([[current_pos[0], current_pos[1]]])

        return [wave_scat] + particle_lines + particle_dots

    anim = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
    file_path = os.path.join(abs_save_dir, "4d_momentum_collapse.gif")

    print(f"🔄 Rendering {num_frames} frames to GIF at 10 FPS...\n   --> {file_path}")
    anim.save(file_path, writer='pillow', fps=10)
    print(f"✅ 4D Quantum GIF rendered successfully!")
# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_momentum_operator_test(steps: int = STEPS, iterations: int = ITERATIONS):
    print(f"1. Configuring 4D DOD Blueprint (Steps: {steps}, Iterations: {iterations})...")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State), topology=topology, transition_function=momentum_phase_shift_transition,
        maximum_step_baking=steps, max_size=500_000, state_shape=(4,), implicit_norm=False, explicit_norm=True
    )

    print("2. Spinning up Hardware Component Managers (4D State Arrays)...")

    # Contract is 4D State, 1D Complex Field
    global_contract = FieldKernelDataContract(max_active_states=500_000, state_dimensions=4, field_dimensions=1,
                                              algebra=algebra, state_class_ref=State, mapper_func=None)
    global_field_cm = FieldComponentManager.create_from_raw(NumbaComplexUtility, global_contract,
                                                            NumbaComplexFieldKernelStorage(global_contract),
                                                            NumbaFieldTranslator(), np.empty((0, 4), dtype=np.float64),
                                                            np.empty((0, 1), dtype=np.complex128))
    global_field_cm.fill(1.0 + 0.0j)

    topology_contract = TopologyKernelDataContract(hardware_momentum_neighbors, State, 500_000, 4, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(topology_contract,
                                                                NumbaTopologyStorage(topology_contract),
                                                                NumbaTopologyTranslator(), NumbaTopologyUtility)

    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
                                             GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                             momentum_phase_shift_transition)

    op_cm = OperatorComponentManager.create_raw(evolution_func=quantum_collapse_batch_kernel,
                                                utility=NumbaOperatorUtility(), state_class_ref=State)

    print("\n===========================================")
    print("      4D MOMENTUM TELEMETRY PIPELINE       ")
    print("===========================================")

    # Initialize particle at [x=0, y=0, px=0, py=0]
    current_states = np.zeros((NUM_PARTICLES, 4), dtype=np.float64)

    total_frames = 1 + (iterations * steps) + iterations
    history_particles = np.zeros((total_frames, NUM_PARTICLES, 4), dtype=np.float64)
    history_wave = []

    frame_idx = 0
    history_particles[frame_idx] = current_states
    history_wave.append((np.empty((0, 4)), np.empty(0)))
    frame_idx += 1

    safe_warmup = 50
    print(f"   -> Expanding Topology Graph (Warmup depth: {safe_warmup})...")
    warmup_states = [State(s) for s in current_states]
    topology_cm.warmup(warmup_states, steps=safe_warmup + 2)

    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    print("   -> Executing JIT Warmup (Compiling Numba kernels)...")
    dummy_phases = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)
    generator_cm.load_initial_state(current_states, dummy_phases)
    d_states, d_fields = generator_cm.generate_steps(steps=1)
    M_dummy = len(d_states)
    b_s_dummy = np.ascontiguousarray(np.broadcast_to(d_states, (NUM_PARTICLES, M_dummy, 4)))
    b_f_dummy = np.ascontiguousarray(np.broadcast_to(d_fields, (NUM_PARTICLES, M_dummy, 1)))
    _ = op_cm.evolve_batch_inplace(current_states, b_s_dummy, b_f_dummy)
    generator_cm.clear()

    print(f"   -> Starting Frame-by-Frame Continuous Observer Loop...")

    for i in range(iterations):
        generator_cm.clear()

        # Reset Phase to pure real 1.0 + 0.0j at start of each collapse
        particle_phases = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)
        generator_cm.load_initial_state(current_states, particle_phases)

        # --- 1. CONTINUOUS EVOLUTION ---
        for step in range(steps):
            final_states, final_fields = generator_cm.generate_steps(steps=1)

            history_particles[frame_idx] = current_states
            history_wave.append((final_states.copy(), np.abs(final_fields[:, 0]) ** 2))
            frame_idx += 1

        # --- 2. DISCRETE OBSERVATION ---
        M = len(final_states)
        b_s = np.ascontiguousarray(np.broadcast_to(final_states, (NUM_PARTICLES, M, 4)))
        b_f = np.ascontiguousarray(np.broadcast_to(final_fields, (NUM_PARTICLES, M, 1)))

        # Collapse
        op_cm.evolve_batch_inplace(current_states, b_s, b_f)

        history_particles[frame_idx] = current_states
        history_wave.append((np.empty((0, 4)), np.empty(0)))
        frame_idx += 1

        print(f"      Iteration {i + 1}/{iterations} complete...")

    animate_momentum_trajectory(history_particles, history_wave, PATH)


if __name__ == "__main__":
    run_momentum_operator_test()