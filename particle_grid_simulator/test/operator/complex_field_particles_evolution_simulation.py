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
ITERATIONS = 40
PATH = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"


# ==========================================
# PHYSICS & OPERATOR
# ==========================================
@njit(cache=True, fastmath=True)
def phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
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
# FRAME-BY-FRAME VISUALIZATION
# ==========================================
def animate_multi_trajectory(history_particles: list, history_wave: list, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    history_arr = np.array(history_particles)
    num_frames = len(history_arr)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, NUM_PARTICLES))
    ax.set_facecolor('#050510')
    ax.grid(True, linestyle='--', color='white', alpha=0.1)
    ax.set_title(f"Quantum Frame-by-Frame Evolution\n({NUM_PARTICLES} Particles, {ITERATIONS} Collapses)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='datalim')

    # Establish visual limits safely
    x_min, x_max = np.min(history_arr[:, :, 0]), np.max(history_arr[:, :, 0])
    y_min, y_max = np.min(history_arr[:, :, 1]), np.max(history_arr[:, :, 1])
    buffer = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5)
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)

    wave_scat = ax.scatter([], [], c=[], cmap='magma', s=35, alpha=0.8, edgecolors='none', zorder=2)
    particle_lines = [
        ax.plot([], [], marker='o', markersize=2, linestyle='-', linewidth=0.5, color=colors[p], alpha=0.3)[0] for p in
        range(NUM_PARTICLES)]
    particle_scatters = [ax.scatter([], [], color=colors[p], edgecolor='white', s=100, marker='o', zorder=6) for p in
                         range(NUM_PARTICLES)]

    def update(frame):
        active_states, active_masses = history_wave[frame]
        if len(active_states) > 0:
            wave_scat.set_offsets(active_states)
            wave_scat.set_array(active_masses / (np.max(active_masses) + 1e-12))
        else:
            wave_scat.set_offsets(np.empty((0, 2)))

        trail_data = history_arr[:frame + 1]
        current_pos = history_arr[frame]

        for p in range(NUM_PARTICLES):
            particle_lines[p].set_data(trail_data[:, p, 0], trail_data[:, p, 1])
            particle_scatters[p].set_offsets([[current_pos[p, 0], current_pos[p, 1]]])

        return [wave_scat] + particle_lines + particle_scatters

    anim = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=150)
    file_path = os.path.join(abs_save_dir, "complex_frame_by_frame.gif")

    print(f"🔄 Rendering {num_frames} frames to GIF...\n   --> {file_path}")
    anim.save(file_path, writer='pillow', fps=8)
    print(f"✅ Quantum GIF rendered successfully!")


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_complex_multi_operator_test(steps: int = STEPS, iterations: int = ITERATIONS):
    print(f"1. Configuring Complex DOD Blueprint (Batch: {NUM_PARTICLES}, Steps: {steps}, Iterations: {iterations})...")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State), topology=topology, transition_function=phase_shift_transition,
        maximum_step_baking=steps, max_size=250000, state_shape=(2,), implicit_norm=False, explicit_norm=True
    )

    print("2. Spinning up Hardware Component Managers (Capacity: 250,000)...")
    global_contract = FieldKernelDataContract(max_active_states=250000, state_dimensions=2, field_dimensions=1,
                                              algebra=algebra, state_class_ref=State, mapper_func=None)
    global_field_cm = FieldComponentManager.create_from_raw(NumbaComplexUtility, global_contract,
                                                            NumbaComplexFieldKernelStorage(global_contract),
                                                            NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64),
                                                            np.empty((0, 1), dtype=np.complex128))
    global_field_cm.fill(1.0 + 0.0j)

    topology_contract = TopologyKernelDataContract(hardware_random_walker_neighbors, State, 250000, 2, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(topology_contract,
                                                                NumbaTopologyStorage(topology_contract),
                                                                NumbaTopologyTranslator(), NumbaTopologyUtility)

    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
                                             GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                             phase_shift_transition)

    op_cm = OperatorComponentManager.create_raw(evolution_func=quantum_collapse_batch_kernel,
                                                utility=NumbaOperatorUtility(), state_class_ref=State)

    print("\n===========================================")
    print("      COMPLEX TELEMETRY PIPELINE           ")
    print("===========================================")

    current_states = np.random.randint(-2, 3, size=(NUM_PARTICLES, 2)).astype(np.float64)
    history_particles = [current_states.copy()]
    history_wave = [(np.empty((0, 2)), np.empty(0))]

    safe_warmup = (steps + iterations)
    print(f"   -> Expanding Topology Graph (Warmup depth: {safe_warmup})...")
    topology_cm.warmup([State(np.array([0.0, 0.0]))], steps=safe_warmup)

    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    # ---------------------------------------------------------
    # NEW: JIT WARMUP PHASE (Compiles kernels before the timer)
    # ---------------------------------------------------------
    print("   -> Executing JIT Warmup (Compiling Numba kernels)...")
    dummy_states = np.zeros((NUM_PARTICLES, 2), dtype=np.float64)
    dummy_phases = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)

    generator_cm.load_initial_state(dummy_states, dummy_phases)
    d_states, d_fields = generator_cm.generate_steps(steps=1)

    M_dummy = len(d_states)
    b_s_dummy = np.ascontiguousarray(np.broadcast_to(d_states, (NUM_PARTICLES, M_dummy, 2)))
    b_f_dummy = np.ascontiguousarray(np.broadcast_to(d_fields, (NUM_PARTICLES, M_dummy, 1)))

    _ = op_cm.evolve_batch_inplace(dummy_states, b_s_dummy, b_f_dummy)
    generator_cm.clear()
    print("   -> JIT Warmup Complete.")
    # ---------------------------------------------------------

    print(f"   -> Starting Frame-by-Frame Continuous Observer Loop...")
    t_start = time.perf_counter()  # Timer starts NOW, post-compilation

    for i in range(iterations):
        generator_cm.clear()

        particle_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, (NUM_PARTICLES, 1))).astype(np.complex128)
        generator_cm.load_initial_state(current_states, particle_phases)

        # --- 1. CONTINUOUS EVOLUTION (The Wave Grows) ---
        for step in range(steps):
            final_states, final_fields = generator_cm.generate_steps(steps=1)

            history_particles.append(current_states.copy())
            history_wave.append((final_states.copy(), np.abs(final_fields[:, 0]) ** 2))

        # --- 2. DISCRETE OBSERVATION (The Wave Collapses) ---
        M = len(final_states)
        b_s = np.ascontiguousarray(np.broadcast_to(final_states, (NUM_PARTICLES, M, 2)))
        b_f = np.ascontiguousarray(np.broadcast_to(final_fields, (NUM_PARTICLES, M, 1)))

        op_cm.evolve_batch_inplace(current_states, b_s, b_f)

        history_particles.append(current_states.copy())
        history_wave.append((np.empty((0, 2)), np.empty(0)))

        if (i + 1) % 10 == 0:
            print(f"      Collapse Event {i + 1}/{iterations} complete...")

    print(f"   [Phase 3] Total Runtime (Post-JIT): {(time.perf_counter() - t_start):.4f} s")

    animate_multi_trajectory(history_particles, history_wave, PATH)


if __name__ == "__main__":
    run_complex_multi_operator_test(steps=30, iterations=8)