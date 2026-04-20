import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit

from particle_grid_simulator.src.dynamic_system.domain.data.single_channel_fds import SingleChannelFDSData, \
    SingleChannelFDSRunner
from particle_grid_simulator.src.dynamic_system.domain.utility.single_channel_fds import SingleChannelFDSUtility

# --- Engine Imports ---
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData

from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import TopologyKernelDataContract, \
    NumbaTopologyStorage
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract

from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager

from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility

from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage


# ==========================================
# 1. KERNELS & PHYSICS
# ==========================================
def save_quantum_propagation_gif(csv_path: Path, save_path: Path, n_particles: int, steps: int):
    print("   -> Rendering True Complex Wave Propagation GIF...")

    flat_data = np.loadtxt(csv_path, delimiter=',', skiprows=1, ndmin=2)
    if flat_data.shape[0] <= 1:
        print("\n❌ ERROR: The telemetry CSV only has 1 frame (t=0).")
        return

    history = flat_data.reshape(flat_data.shape[0], n_particles, 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#050510')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_title("True Complex Wave Interference (|Σψ|²)")

    box = plt.Rectangle((-10, -10), 20, 20, fill=False, color='cyan', linestyle='--', linewidth=2)
    ax.add_patch(box)

    colors = plt.cm.spring(np.linspace(0, 1, n_particles))
    scats = [ax.scatter([], [], color=colors[p], s=120, edgecolors='white', zorder=4) for p in range(n_particles)]

    grid_res = 200  # Higher resolution to see the interference fringes!
    X, Y = np.meshgrid(np.linspace(-11, 11, grid_res), np.linspace(-11, 11, grid_res))

    # We use a tighter colormap 'inferno' to make the destructive (black) areas pop
    im = ax.imshow(np.zeros_like(X), extent=[-11, 11, -11, 11], origin='lower', cmap='inferno', alpha=0.9, vmin=0,
                   vmax=2.0)

    frames_per_tick = steps + 2
    total_frames = (len(history) - 1) * frames_per_tick

    def update(frame):
        iteration = frame // frames_per_tick
        sub_frame = frame % frames_per_tick

        if iteration >= len(history) - 1:
            iteration = len(history) - 2
            sub_frame = frames_per_tick - 1

        current_pos = history[iteration]
        next_pos = history[iteration + 1]

        if sub_frame < steps:
            spread = max(0.2, (sub_frame + 1) * 0.6)

            # --- THE FIX: We must add COMPLEX waves, not real ones! ---
            Z_complex = np.zeros_like(X, dtype=np.complex128)

            for p in range(n_particles):
                px, py = current_pos[p]
                scats[p].set_offsets([px, py])

                dx = X - px
                dy = Y - py
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # 1. Amplitude (Gaussian dropoff)
                amplitude = np.exp(-(dist ** 2) / (2 * spread ** 2))

                # 2. Phase (Directional + Radial Ripples to simulate the grid steps)
                # The dist * 3.0 creates the physical wave frequency ripples
                theta = np.arctan2(dy, dx) - (dist * 3.0)

                # 3. Superposition (Adding complex amplitudes)
                Z_complex += amplitude * (np.cos(theta) + 1j * np.sin(theta))

            # The visible heatmap is the Absolute Square of the combined complex wave
            Z = np.abs(Z_complex) ** 2

            mask = (X < -10) | (X > 10) | (Y < -10) | (Y > 10)
            Z[mask] = 0
            im.set_array(Z)

        else:
            # The Collapse
            Z_complex = np.zeros_like(X, dtype=np.complex128)
            for p in range(n_particles):
                px, py = next_pos[p]
                scats[p].set_offsets([px, py])

                dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
                amplitude = np.exp(-(dist ** 2) / (2 * 0.15 ** 2))
                Z_complex += amplitude * (1.0 + 0j)

            Z = np.abs(Z_complex) ** 2
            mask = (X < -10) | (X > 10) | (Y < -10) | (Y > 10)
            Z[mask] = 0
            im.set_array(Z)

        return scats + [im]

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)
    ani.save(save_path / "quantum_propagation_collapse.gif", writer=PillowWriter(fps=12))
    print("✅ Done!")


def make_resolved_quantum_operator(u_coords: np.ndarray, u_weights: np.ndarray, master_states_ptr: np.ndarray):
    """
    Solves the vanishing particle bug.
    Implements: Filter -> Re-normalize -> Pick -> Reserve.
    """
    # Temporary buffer to store newly claimed positions during the batch
    # Initialized with a 'null' coordinate outside the box
    reserved_states = np.full_like(master_states_ptr, -99.0)

    @njit(fastmath=True)
    def operator_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
        M = len(gen_states)
        probs = np.zeros(M, dtype=np.float64)

        # 1. EXCLUSION & RE-NORMALIZATION PHASE
        for i in range(M):
            cand = gen_states[i]

            # Check current start positions
            is_occupied = False
            for p in range(len(master_states_ptr)):
                if cand[0] == master_states_ptr[p, 0] and cand[1] == master_states_ptr[p, 1]:
                    if not (cand[0] == state_vec[0] and cand[1] == state_vec[1]):
                        is_occupied = True;
                        break

            # Check NEWLY CLAIMED positions in this tick
            if not is_occupied:
                for p in range(len(reserved_states)):
                    if cand[0] == reserved_states[p, 0] and cand[1] == reserved_states[p, 1]:
                        is_occupied = True;
                        break

            if is_occupied:
                continue  # Probability remains 0.0

            # 2. BORN RULE EVALUATION (Within the Global Field Well)
            for j in range(len(u_coords)):
                if u_coords[j, 0] == cand[0] and u_coords[j, 1] == cand[1]:
                    c_val = gen_fields[i, 0] * u_weights[j, 0]
                    probs[i] = c_val.real ** 2 + c_val.imag ** 2
                    break

        # 3. PICK & RESERVE
        total_prob = np.sum(probs)
        if total_prob > 1e-12:
            rand_val = np.random.random() * total_prob
            cumulative = 0.0
            for i in range(M):
                cumulative += probs[i]
                if rand_val <= cumulative:
                    chosen = gen_states[i]
                    # This is the key: we 'reserve' this spot for the next particle in the batch
                    # In a vectorized/batch call, we find the first empty slot in reserved_states
                    for p in range(len(reserved_states)):
                        if reserved_states[p, 0] == -99.0:
                            reserved_states[p, 0] = chosen[0]
                            reserved_states[p, 1] = chosen[1]
                            break
                    return chosen

        return state_vec  # No valid move found, stay put

    return operator_kernel

@njit(cache=True, fastmath=True)
def uniform_complex_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """Emits a pure complex wave amplitude (1.0 + 0.0j)."""
    return np.array([1.0 + 0.0j], dtype=np.complex128)

@njit(cache=True, fastmath=True)
def phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """Emits a complex wave with phase shifts for true quantum interference."""
    dx = s_i[0] - s_j[0]
    dy = s_i[1] - s_j[1]
    theta = np.arctan2(dy, dx)
    # The complex weight: cos(theta) + i sin(theta)
    return np.array([np.cos(theta) + 1j * np.sin(theta)], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    """Standard Infinite Grid Topology. The Box is defined by the field, not the topology!"""
    x, y = state_vec[0], state_vec[1]
    return np.array([[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]], dtype=np.float64)


def random_walker_reachable(state: State):
    return [State(v) for v in hardware_random_walker_neighbors(state.vector)]

#
# def make_exclusive_born_operator(u_coords: np.ndarray, u_weights: np.ndarray, master_states_ptr: np.ndarray):
#     """
#     Implements: Exclude -> Re-normalize -> Pick.
#     Ensures no two particles can ever occupy the same state in the same tick.
#     """
#     # Shared across all particles in the batch during this tick
#     # In a real batch, you'd pass this in or clear it per macro-tick
#     reserved_states = np.full_like(master_states_ptr, -999.0)
#
#     @njit(fastmath=True)
#     def operator_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
#         M = len(gen_states)
#         valid_indices = []
#         raw_probs = np.zeros(M, dtype=np.float64)
#
#         # --- STEP 1: EXCLUSION & RAW PROBABILITY ---
#         for i in range(M):
#             cand = gen_states[i]
#
#             # Check against CURRENT positions (master_states_ptr)
#             occupied = False
#             for p in range(len(master_states_ptr)):
#                 if cand[0] == master_states_ptr[p, 0] and cand[1] == master_states_ptr[p, 1]:
#                     if not (cand[0] == state_vec[0] and cand[1] == state_vec[1]):
#                         occupied = True;
#                         break
#
#             # Check against RESERVED positions (reserved_states)
#             for p in range(len(reserved_states)):
#                 if cand[0] == reserved_states[p, 0] and cand[1] == reserved_states[p, 1]:
#                     occupied = True;
#                     break
#
#             if occupied: continue
#
#             # --- STEP 2: FIELD WEIGHTING ---
#             for j in range(len(u_coords)):
#                 if u_coords[j, 0] == cand[0] and u_coords[j, 1] == cand[1]:
#                     c_val = gen_fields[i, 0] * u_weights[j, 0]
#                     p_val = c_val.real ** 2 + c_val.imag ** 2
#                     if p_val > 1e-15:
#                         raw_probs[i] = p_val
#                         valid_indices.append(i)
#                     break
#
#         # --- STEP 3: RE-NORMALIZE & PICK ---
#         total_p = np.sum(raw_probs)
#         if total_p > 1e-15:
#             # Re-normalization happens here implicitly by using total_p as the scale
#             rand_val = np.random.random() * total_p
#             cumulative = 0.0
#             for idx in valid_indices:
#                 cumulative += raw_probs[idx]
#                 if rand_val <= cumulative:
#                     new_state = gen_states[idx]
#                     # Update reserved_states (must be handled carefully in parallel/batch)
#                     return new_state
#
#         return state_vec
#
#     return operator_kernel


def make_exclusive_born_operator(u_coords: np.ndarray, u_weights: np.ndarray, master_states_ptr: np.ndarray):
    """
    Pure Born Rule with Sequential Exclusion.
    Logic: Exclude current/claimed positions -> Re-normalize -> Pick.
    """

    @njit(fastmath=True)
    def operator_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
        M = len(gen_states)
        probs = np.zeros(M, dtype=np.float64)

        for i in range(M):
            cand = gen_states[i]

            # --- THE EXCLUSION CHECK ---
            # We check directly against master_states_ptr.
            # Because the Runner (Phase 2) updates this pointer sequentially,
            # this check automatically covers both current and newly claimed spots.
            is_occupied = False
            for p in range(len(master_states_ptr)):
                if cand[0] == master_states_ptr[p, 0] and cand[1] == master_states_ptr[p, 1]:
                    # Allow the particle to 'stay put' on its own start position
                    if not (cand[0] == state_vec[0] and cand[1] == state_vec[1]):
                        is_occupied = True
                        break

            if is_occupied:
                continue

                # --- FIELD EVALUATION ---
            for j in range(len(u_coords)):
                if u_coords[j, 0] == cand[0] and u_coords[j, 1] == cand[1]:
                    # Combine local wave with global boundary
                    c_val = gen_fields[i, 0] * u_weights[j, 0]
                    probs[i] = c_val.real ** 2 + c_val.imag ** 2
                    break

        # --- RE-NORMALIZE AND PICK ---
        total_prob = np.sum(probs)
        if total_prob > 1e-12:
            rand_val = np.random.random() * total_prob
            cumulative = 0.0
            for i in range(M):
                cumulative += probs[i]
                if rand_val <= cumulative:
                    return gen_states[i]

        return state_vec

    return operator_kernel


#==========================================
# 2. VISUALIZATION
# ==========================================
def save_complex_gif(csv_path: Path, save_path: Path, n_particles: int):
    print("   -> Rendering Quantum Box GIF...")
    flat_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    history = flat_data.reshape(flat_data.shape[0], n_particles, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#050510')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    box = plt.Rectangle((-10, -10), 20, 20, fill=False, color='cyan', linestyle='--', linewidth=2)
    ax.add_patch(box)
    ax.set_title("Complex Overlapping Field (Infinite Potential Well)")

    colors = plt.cm.spring(np.linspace(0, 1, n_particles))
    scats = [ax.scatter([], [], color=colors[p], s=120, edgecolors='white', zorder=4) for p in range(n_particles)]
    lines = [ax.plot([], [], color=colors[p], alpha=0.4, linewidth=2, zorder=3)[0] for p in range(n_particles)]

    def update(frame):
        for p in range(n_particles):
            start_idx = max(0, frame - 15)
            lines[p].set_data(history[start_idx:frame + 1, p, 0], history[start_idx:frame + 1, p, 1])
            scats[p].set_offsets(history[frame: frame + 1, p])
        return lines + scats

    ani = FuncAnimation(fig, update, frames=len(history), blit=True)
    ani.save(save_path / "quantum_box_probabilistic.gif", writer=PillowWriter(fps=12))
    print("✅ Done!")


# ==========================================
# 3. MASTER PIPELINE
# ==========================================
def run_complex_box_test():
    NUM_PARTICLES = 50
    STEPS = 5
    ITERATIONS = 60
    SAVE_DIR = Path(r"./plots")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=random_walker_reachable, state_class=State, use_cache=True)

    g_contract = FieldKernelDataContract(
        max_active_states=100000,
        state_dimensions=2,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State,
        mapper_func=None
    )

    t_contract = TopologyKernelDataContract(
        hardware_random_walker_neighbors,
        State,
        100000,
        2,
        np.float64
    )

    gen_data = GenericMarkovianFieldGeneratorData(FieldMapper(algebra, State), topology, uniform_complex_transition,
                                                  STEPS, 100000, (2,), False, True)
    gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=1)

    topology_cm = TopologyComponentManager.create_from_raw_data(
        t_contract, NumbaTopologyStorage(t_contract), NumbaTopologyTranslator(), NumbaTopologyUtility)

    generator_cm = GeneratorComponentManager(
        gen_contract, NumbaComplexCSRGeneratorStorage(gen_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, uniform_complex_transition)

    print("   -> Baking Infinite Topology...")
    # Warm up large enough to cover the [-10, 10] box plus the dead-zone boundary
    topology_cm.warmup([State(np.array([0.0, 0.0]))], steps=25)
    u_coords = np.array(topology_cm.fast_refs.handle_map)
    u_weights = np.zeros((len(u_coords), 1), dtype=np.complex128)

    # === DEFINING THE BOX VIA THE GLOBAL FIELD ===
    for i in range(len(u_coords)):
        x, y = u_coords[i]
        if -10.0 <= x <= 10.0 and -10.0 <= y <= 10.0:
            u_weights[i, 0] = 1.0 + 0.0j  # Valid Space
        else:
            u_weights[i, 0] = 0.0 + 0.0j  # Infinite Potential Wall (0 probability)

    field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, g_contract, NumbaComplexFieldKernelStorage(g_contract),
        NumbaFieldTranslator(), u_coords, u_weights)

    s0 = np.zeros((NUM_PARTICLES, 2), dtype=np.float64)
    s0[:, 0] = np.linspace(-8.0, 8.0, NUM_PARTICLES)
    f0 = np.ones((NUM_PARTICLES, 1), dtype=np.complex128)

    smart_kernel = make_exclusive_born_operator(u_coords, u_weights, s0)
    op_cm = OperatorComponentManager.create_raw(smart_kernel, NumbaOperatorUtility(), State)

    system_data = SingleChannelFDSData(
        _initial_states=s0, _initial_fields=f0,
        _topology_cm=topology_cm, _field_cm=field_cm, _generator_cm=generator_cm, _operator_cm=op_cm,
        _history_window_size=20, _save_directory=SAVE_DIR, _is_independent=False
    )

    runner = SingleChannelFDSRunner(system_data, SingleChannelFDSUtility)

    print("   -> Running Quantum Loop...")
    for _ in range(ITERATIONS):
        # 1. The wave expands over time and interferes

        runner.next(apply_generator=True, steps=STEPS)
        runner.next(apply_generator=False)

        # 2. The observer looks, collapsing the wave instantly
    runner.end(compile_csv=True)

        # Change the visualization call to the new function and pass STEPS
    save_quantum_propagation_gif(SAVE_DIR / "compiled_telemetry.csv", SAVE_DIR, NUM_PARTICLES, STEPS)


if __name__ == "__main__":
    run_complex_box_test()