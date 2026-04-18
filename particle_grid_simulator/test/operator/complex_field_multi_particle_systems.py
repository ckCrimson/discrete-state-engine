import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
NUM_PARTICLES = 15
STEPS = 4
ITERATIONS = 60  # Set slightly higher for a more expressive GIF
PATH = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"


# ==========================================
# 1. PURE C-SPEED PHYSICS RULES (COMPLEX)
# ==========================================
def plot_pre_collapse_superposition(history_particles, history_wave, iteration_index: int, save_dir: str):
    """
    Plots the superimposed complex wave field right before the operator collapses it.
    """
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    # 1. Extract the data for the specific iteration
    # particle_positions are where the particles are BEFORE the jump
    particle_positions = history_particles[iteration_index]

    # wave_states and wave_masses are the expanded field
    wave_states, wave_masses = history_wave[iteration_index]

    if len(wave_states) == 0:
        print("⚠️ Warning: Wave field is empty for this iteration.")
        return

    # 2. Set up the plot
    plt.figure(figsize=(12, 10))
    plt.gca().set_facecolor('#050510')  # Dark theme for the wave

    # 3. Plot the Probability Field (The Wave)
    X, Y = wave_states[:, 0], wave_states[:, 1]

    # Use log normalization if the wave is highly localized, otherwise linear
    # We add a tiny epsilon to prevent log(0)
    scat_wave = plt.scatter(X, Y, c=wave_masses, cmap='magma', s=25, alpha=0.8,
                            norm=plt.matplotlib.colors.LogNorm(vmin=max(1e-6, np.min(wave_masses)),
                                                               vmax=np.max(wave_masses)),
                            edgecolors='none', zorder=2)

    plt.colorbar(scat_wave, label='Probability Mass $|Z|^2$ (Log Scale)')

    # 4. Plot the Particles (The Sources)
    colors = plt.cm.hsv(np.linspace(0, 1, len(particle_positions)))
    for p in range(len(particle_positions)):
        px, py = particle_positions[p, 0], particle_positions[p, 1]
        plt.scatter([px], [py], color=colors[p], edgecolor='white', s=100, marker='o', zorder=5,
                    label=f'P{p}' if p < 5 else "")

    plt.title(f"Pre-Collapse Superposition Field (Iteration {iteration_index})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', color='white', alpha=0.1)

    # 5. Save
    file_path = os.path.join(abs_save_dir, f"pre_collapse_field_iter_{iteration_index}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    plt.close('all')
    print(f"✅ Pre-Collapse Snapshot saved to:\n   --> {file_path}")

@njit(cache=True, fastmath=True)
def phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """Simulates quantum-like wave interference."""
    dx = s_i[0] - s_j[0]
    dy = s_i[1] - s_j[1]
    theta = np.arctan2(dy, dx)
    c_weight = np.cos(theta) + 1j * np.sin(theta)
    return np.array([c_weight], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0], [x, y - 1.0], [x + 1.0, y + 1.0],
        [x - 1.0, y - 1.0], [x + 1.0, y], [x - 1.0, y], [x + 1.0, y - 1], [x - 1.0, y + 1]
    ], dtype=np.float64)


# ==========================================
# 2. BATCH OPERATOR RULE (OBSERVER)
# ==========================================
@njit(fastmath=True)
def quantum_collapse_batch_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
    """Collapses Complex Amplitudes using the Born Rule (|z|^2) for a single particle in a batch."""
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

    return state_vec  # Fallback if wave is zero


# ==========================================
# 3. MULTI-PARTICLE VISUALIZATION (GIF Version)
# ==========================================
def animate_multi_trajectory(history: list, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    # History is a list of (NUM_PARTICLES, 2) arrays. Shape: (Iterations+1, Particles, 2)
    history_arr = np.array(history)
    num_frames = len(history_arr)

    # Setup Figure and Axes once
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, NUM_PARTICLES))
    ax.set_facecolor('#050510')  # Dark theme
    ax.grid(True, linestyle='--', color='white', alpha=0.1)
    ax.set_title(f"Quantum Multi-Particle Evolution ({NUM_PARTICLES} Particles, {num_frames - 1} Jumps)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='datalim')

    # Establish visual limits based on the complete history data
    x_min, x_max = np.min(history_arr[:, :, 0]), np.max(history_arr[:, :, 0])
    y_min, y_max = np.min(history_arr[:, :, 1]), np.max(history_arr[:, :, 1])
    buffer_x = (x_max - x_min) * 0.1
    buffer_y = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
    ax.set_ylim(y_min - buffer_y, y_max + buffer_y)

    # Pre-create visual objects (empty for initialization)
    # 1. Start points (static)
    start_pts = history_arr[0]
    for p in range(NUM_PARTICLES):
        ax.scatter(start_pts[p, 0], start_pts[p, 1], color='white', edgecolor='black', s=80, zorder=5)

    # 2. Particle trails (lines)
    particle_lines = [
        ax.plot([], [], marker='o', markersize=3, linestyle='-', linewidth=1, color=colors[p], alpha=0.3)[0] for p in
        range(NUM_PARTICLES)]

    # 3. Particle heads (scatters)
    particle_scatters = [ax.scatter([], [], color=colors[p], edgecolor='black', s=120, marker='*', zorder=6) for p in
                         range(NUM_PARTICLES)]

    def update(frame):
        """Animation update function."""
        # 1. Update trails (lines up to current frame)
        trail_data = history_arr[:frame + 1]
        for p in range(NUM_PARTICLES):
            particle_lines[p].set_data(trail_data[:, p, 0], trail_data[:, p, 1])

        # 2. Update particle positions (head scatters) for current frame
        current_pos = history_arr[frame]
        # set_offsets needs an Nx2 array of coordinates
        pos_array = current_pos
        # To avoid recreating scatters (slow), update their properties. But `scatter.set_offsets` handles multiple points.
        # It's more efficient for tutor tutorial style to pre-create and set offsets.
        for p in range(NUM_PARTICLES):
            # Visual Tutoring tip: for speed when animating many frames/objects, use set_data or set_offsets.
            particle_scatters[p].set_offsets([[pos_array[p, 0], pos_array[p, 1]]])

        # Combine all dynamic objects to return for efficient blitting
        return particle_lines + particle_scatters

    # Create Animation object
    # frames defines the number of unique updates. blit=True re-draws only changed visual elements for performance.
    # interval is delay between frames in ms (100ms = 10 FPS). fps below should match.
    anim = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)

    file_path = os.path.join(abs_save_dir, "complex_multi_particle_evolution.gif")

    # Save the animation as a GIF using the 'pillow' writer for broad compatibility
    # Ensure Pillow is installed: `pip install pillow`
    # Standard Tutor instruction using high-level FuncAnimation.pillow
    print(f"🔄 Compiling and saving GIF to:\n   --> {file_path} (This might take a moment)...")
    anim.save(file_path, writer='pillow', fps=10)  # Pillow is usually installed or available.

    print(f"✅ Multi-Particle Evolution GIF saved successfully!")


# ==========================================
# 4. EXECUTION PIPELINE (GIF version)
# ==========================================
def run_complex_multi_operator_test(steps: int = STEPS, iterations: int = ITERATIONS):
    print(f"1. Configuring Complex DOD Blueprint (Batch: {NUM_PARTICLES}, Steps: {steps}, Iterations: {iterations})...")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=phase_shift_transition,
        maximum_step_baking=steps,
        max_size=250000,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
    )

    print("2. Spinning up Hardware Component Managers (Capacity: 250,000)...")

    # Field Manager
    global_contract = FieldKernelDataContract(
        state_dimensions=2, field_dimensions=1, algebra=algebra,
        state_class_ref=State, mapper_func=None, initial_capacity=250000
    )
    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64), np.empty((0, 1), dtype=np.complex128)
    )
    global_field_cm.fill(1.0 + 0.0j)

    # Topology Manager
    topology_contract = TopologyKernelDataContract(hardware_random_walker_neighbors, State, 250000, 2, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    # Generator Manager
    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, phase_shift_transition
    )

    # Operator Manager (Using batch kernel)
    op_cm = OperatorComponentManager.create_raw(
        evolution_func=quantum_collapse_batch_kernel,
        utility=NumbaOperatorUtility(),
        state_class_ref=State
    )

    print("\n===========================================")
    print("      COMPLEX TELEMETRY PIPELINE           ")
    print("===========================================")

    current_states = np.random.randint(-2, 3, size=(NUM_PARTICLES, 2)).astype(np.float64)
    history = [current_states.copy()]
    history_wave = []  # <--- 1. ADD THIS LIST

    # Warmup topology
    t_start = time.perf_counter()
    safe_warmup = iterations + 10
    print(f"   -> Expanding Topology Graph (Warmup depth: {safe_warmup})...")
    # Feed one state to seed the graph expansion
    topology_cm.warmup([State(np.array([0.0, 0.0]))], steps=safe_warmup)

    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    print(f"   -> Starting Batch Observer Loop...")
    for i in range(iterations):
        generator_cm.clear()

        # Assign random complex phases to particles for this tick
        particle_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, (NUM_PARTICLES, 1))).astype(np.complex128)

        generator_cm.load_initial_state(current_states, particle_phases)

        # 1. Expand Wave (All particles simultaneously write to the same complex CSR memory)
        final_states, final_fields = generator_cm.generate_steps(steps=steps)

        history_wave.append((final_states.copy(), np.abs(final_fields[:, 0]) ** 2))

        # 2. Prepare Batch Layout for the Operator
        M = len(final_states)
        b_s = np.ascontiguousarray(np.broadcast_to(final_states, (NUM_PARTICLES, M, 2)))
        b_f = np.ascontiguousarray(np.broadcast_to(final_fields, (NUM_PARTICLES, M, 1)))

        # 3. Collapse Wave (In-place C-speed update)
        op_cm.evolve_batch_inplace(current_states, b_s, b_f)

        history.append(current_states.copy())

        if (i + 1) % 10 == 0:
            print(f"      Tick {i + 1}/{iterations} complete...")

    print(f"   [Phase 3] Total Loop Execution: {(time.perf_counter() - t_start):.4f} s")

    # Generate the GIF
    animate_multi_trajectory(history, PATH)

    # Take a snapshot of the complex field just before the 5th jump
    # Ensure we don't go out of bounds if iterations < 5
    snapshot_iteration = min(4, len(history_wave) - 1)

    # <--- 3. PASS history_wave INSTEAD OF history TWICE
    plot_pre_collapse_superposition(history, history_wave, snapshot_iteration, PATH)

if __name__ == "__main__":
    # Let's increase iterations slightly for a longer GIF and more expressible evolution.
    # steps=4, iterations=60 are good for capturing several collapse events visually.
    run_complex_multi_operator_test(steps=30, iterations=60)