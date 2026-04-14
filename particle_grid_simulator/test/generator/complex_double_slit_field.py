import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from PIL import Image

from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage

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
# COMPONENT MANAGER & STANDARD DOD IMPORTS
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
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility


# ==========================================
# 1. 4D QUANTUM WALK PHYSICS RULES
# ==========================================
@njit(cache=True, fastmath=True)
def hardware_quantum_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]

    # State permanently records the direction it took [x, y, dx, dy]
    return np.array([
        [x, y + 1.0, 0.0, 1.0],  # Stepped North
        [x, y - 1.0, 0.0, -1.0],  # Stepped South
        [x + 1.0, y, 1.0, 0.0],  # Stepped East
        [x - 1.0, y, -1.0, 0.0]  # Stepped West
    ], dtype=np.float64)


@njit(cache=True, fastmath=True)
def grover_coin_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    # s_j: [old_x, old_y, incoming_dx, incoming_dy]
    # s_i: [new_x, new_y, outgoing_dx, outgoing_dy]

    in_dx, in_dy = s_j[2], s_j[3]
    out_dx, out_dy = s_i[2], s_i[3]

    # 1. The Grover Scattering Matrix (Topological splitting)
    dot = in_dx * out_dx + in_dy * out_dy
    if dot == 1.0:
        coin_weight = -0.5
    else:
        coin_weight = 0.5

        # 2. Continuous Spatial Phase (Complex wave rotation)
    # k controls how fast the colors shift through the phase plot
    k = 0.8
    prop_phase = np.cos(k) + 1j * np.sin(k)

    # Combine the topological split with the continuous rotation
    final_weight = coin_weight * prop_phase

    return np.array([final_weight], dtype=np.complex128)


from numba import njit
import numpy as np

from numba import njit
import numpy as np


@njit(cache=True, fastmath=True)
def natural_path_integral_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    x0, y0 = s_j[0], s_j[1]
    xn, yn = s_i[0], s_i[1]

    in_dx, in_dy = s_j[2], s_j[3]
    out_dx, out_dy = s_i[2], s_i[3]

    # 1. Kinematics (Delta Theta)
    dot = (in_dx * out_dx) + (in_dy * out_dy)
    cross = (in_dx * out_dy) - (in_dy * out_dx)
    delta_theta = np.arctan2(cross, dot)

    # 2. Inertial Action (Kinetic Energy)
    # Particles naturally resist turning
    mass = 0.5
    W_kinetic = np.exp(-mass * (delta_theta ** 2))

    # ==========================================
    # THE DAM FIX: RESTORE DESTRUCTIVE INTERFERENCE
    # ==========================================
    # We force the forward-moving amplitude to be negative.
    # When waves reflect off the wall and travel backward, they will
    # finally be able to subtract from the incoming positive waves!
    if dot == 1.0:
        W_kinetic = -W_kinetic

    # 3. The Global Field (The Attractor)
    field_j = 0.1 * x0
    field_n = 0.1 * xn

    delta_field = field_n - field_j

    # 4. Boltzmann Amplitude (Thermodynamics)
    coupling_strength = 1.0
    field_amplitude = np.exp(coupling_strength * delta_field)

    # 5. Total Local Probability
    # (Because of the fix above, straight paths now correctly pass a negative amplitude here)
    amplitude = W_kinetic * field_amplitude

    # 6. The Quantum Phase (Feynman)
    base_frequency = 0.8
    phase_shift = base_frequency - (coupling_strength * field_n)

    # Apply Euler's Formula
    weight = amplitude * (np.cos(phase_shift) + 1j * np.sin(phase_shift))

    return np.array([weight], dtype=np.complex128)

@njit(cache=True, fastmath=True)
def theoretical_path_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    # 1. Extract physical positions and momentum
    x0, y0 = s_j[0], s_j[1]
    xn, yn = s_i[0], s_i[1]

    in_dx, in_dy = s_j[2], s_j[3]
    out_dx, out_dy = s_i[2], s_i[3]

    # 2. Calculate Angle of Deviation (Delta Theta)
    dot = (in_dx * out_dx) + (in_dy * out_dy)
    cross = (in_dx * out_dy) - (in_dy * out_dx)
    delta_theta = np.arctan2(cross, dot)

    # ==========================================
    # TERM 1: W(x) - The Inertial Function (Gaussian)
    # ==========================================
    beta = 0.5
    W_theta = np.exp(-beta * (delta_theta ** 2))

    # ==========================================
    # TERM 2: The Global Field Attenuation
    # ==========================================
    field_j = 0.1 * x0
    field_n = 0.1 * xn
    delta_field = field_n - field_j

    interaction_strength = 1.0
    field_attenuation = np.exp(-interaction_strength * delta_field)

    # Total physical amplitude
    amplitude = W_theta * field_attenuation

    # ==========================================
    # TERM 3: The Phase Shift (UPDATED)
    # ==========================================
    k_forward = 0.8  # Base forward wavelength (Ticks even when going straight)
    alpha = 1.0  # The chiral coupling constant

    # Total phase is base forward phase PLUS the turn modifier
    phase_shift = -(k_forward + (alpha * delta_theta))

    # Apply Euler's Formula
    weight = amplitude * (np.cos(phase_shift) + 1j * np.sin(phase_shift))

    return np.array([weight], dtype=np.complex128)

@njit(cache=True, fastmath=True)
def angular_coin_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    in_dx, in_dy = s_j[2], s_j[3]
    out_dx, out_dy = s_i[2], s_i[3]

    dot = in_dx * out_dx + in_dy * out_dy
    cross = in_dx * out_dy - in_dy * out_dx
    turn_angle = np.arctan2(cross, dot)

    # 1. THE UNITARITY FIX: We MUST keep the negative sign for straight paths!
    # This creates the destructive interference (the gaps between the strips)
    if dot == 1.0:
        amplitude = -0.5  # Kept going straight
    else:
        amplitude = 0.5  # Turned or Reflected

    # 2. The Phases
    k_prop = 0.8  # Forward propagation phase
    k_turn = 0.5  # Chiral/Turning phase

    total_phase = k_prop + (k_turn * turn_angle)

    # 3. Apply Euler's Formula
    weight = amplitude * (np.cos(total_phase) + 1j * np.sin(total_phase))

    return np.array([weight], dtype=np.complex128)
# 2. VISUALIZATION HELPERS (Spatial Flattener)
# ==========================================
def flatten_quantum_state(states: np.ndarray, fields: np.ndarray):
    """Sums the probabilities of overlapping momentum vectors at the same physical (x, y) coordinate."""
    Z = fields[:, 0]
    magnitude = np.abs(Z)
    phase = np.angle(Z)

    spatial_mag2 = {}
    spatial_phase = {}

    for i in range(len(states)):
        mag = magnitude[i]
        if mag > 1e-10:
            x, y = states[i, 0], states[i, 1]
            coord = (x, y)

            if coord not in spatial_mag2:
                spatial_mag2[coord] = 0.0
                spatial_phase[coord] = phase[i]

            # Sum the probabilities (magnitude squared) of orthogonal states
            spatial_mag2[coord] += (mag ** 2)

    X, Y, mag_masked, phase_masked = [], [], [], []
    for (x, y), mag2 in spatial_mag2.items():
        if mag2 > 1e-10:
            X.append(x)
            Y.append(y)
            mag_masked.append(np.sqrt(mag2))  # Convert probability back to amplitude for color scaling
            phase_masked.append(spatial_phase[(x, y)])

    return np.array(X), np.array(Y), np.array(mag_masked), np.array(phase_masked)


def plot_and_save_complex_field(states: np.ndarray, fields: np.ndarray, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)

    X, Y, mag_masked, phase_masked = flatten_quantum_state(states, fields)

    if len(mag_masked) == 0:
        print("⚠️ Warning: Field mass is zero. Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    sc1 = ax1.scatter(X, Y, c=mag_masked, cmap='magma', marker='s', s=25, norm=mcolors.LogNorm())
    fig.colorbar(sc1, ax=ax1, label='Magnitude $|Z|$ (Log Scale)')
    ax1.set_title("Complex Field: Magnitude (Double Slit)")
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.2)

    sc2 = ax2.scatter(X, Y, c=phase_masked, cmap='twilight', marker='s', s=25, vmin=-np.pi, vmax=np.pi)
    fig.colorbar(sc2, ax=ax2, label='Phase Angle $\\theta$ (Radians)')
    ax2.set_title("Complex Field: Phase (Double Slit)")
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.2)

    plt.suptitle("4D Quantum Grover Walk - Double Slit Experiment (Static)", fontsize=16)

    file_path = os.path.join(abs_save_dir, "quantum_walker_final_frame.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"✅ Static Plot successfully saved to:\n   --> {file_path}")


# ==========================================
# 2.5 DYNAMIC GIF GENERATION (Frame Stitching)
def animate_double_slit_evolution(
        generator_cm, topology_cm, global_field_cm,
        initial_states, initial_fields, total_steps, save_dir
):
    import os
    import time
    import matplotlib.colors as mcolors
    from PIL import Image

    print("\n===========================================")
    print("      🎥 INITIALIZING GIF DIRECTOR        ")
    print("===========================================")

    print("   -> Expanding Grid & Sculpting Slits...")
    generator_cm.load_initial_state(initial_states, initial_fields)

    # 1. Warmup expands the universe mapping to ~8000 nodes
    topology_cm.warmup([State(initial_states[0])], steps=total_steps + 5)

    # =======================================================
    # 2. THE BUG FIX: Fill the newly expanded universe with 1.0!
    # Without this, the new nodes default to 0.0 and kill the wave.
    # =======================================================
    global_field_cm.fill(1.0 + 0.0j)

    coords = topology_cm.fast_refs.handle_map
    for i in range(len(coords)):
        x, y = coords[i][0], coords[i][1]
        if 10.0 <= x <= 11.0:
            is_slit_1 = (1.0 <= y <= 3.0)
            is_slit_2 = (-3.0 <= y <= -1.0)
            if not (is_slit_1 or is_slit_2):
                global_field_cm.fast_refs.field_array[i] = 0.0 + 0.0j

    generator_cm.inject_environment(topology_cm, global_field_cm)

    print(f"   -> Engine Ready. Recording {total_steps} frames of Physics History...")
    t_start = time.perf_counter()
    history = generator_cm.generate_trajectory(steps=total_steps)
    print(f"   -> Physics Recorded in {(time.perf_counter() - t_start) * 1000:.2f} ms.")

    print("   -> Rendering individual frames...")
    abs_save_dir = os.path.normpath(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)

    frame_files = []
    cam_bound = total_steps + 2

    for frame, (states, fields) in enumerate(history):
        # Apply the Spatial Flattener to the current frame
        X, Y, mag_masked, phase_masked = flatten_quantum_state(states, fields)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # =======================================================
        # THE CAMERA FIX: Center the camera to see all 4 directions
        # =======================================================
        ax1.set_xlim(-cam_bound, cam_bound)
        ax1.set_ylim(-cam_bound, cam_bound)
        ax2.set_xlim(-cam_bound, cam_bound)
        ax2.set_ylim(-cam_bound, cam_bound)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.grid(True, linestyle='--', alpha=0.2)
        ax2.grid(True, linestyle='--', alpha=0.2)

        if len(mag_masked) > 0:
            sc1 = ax1.scatter(X, Y, c=mag_masked, cmap='magma', marker='s', s=30,
                              norm=mcolors.LogNorm(vmin=1e-5, vmax=max(1e5, np.max(mag_masked))))
            sc2 = ax2.scatter(X, Y, c=phase_masked, cmap='twilight', marker='s', s=30, vmin=-np.pi, vmax=np.pi)
        else:
            sc1 = ax1.scatter([0], [0], c=[1e-5], cmap='magma', marker='s', s=0,
                              norm=mcolors.LogNorm(vmin=1e-5, vmax=1e5))
            sc2 = ax2.scatter([0], [0], c=[0.0], cmap='twilight', marker='s', s=0, vmin=-np.pi, vmax=np.pi)

        fig.colorbar(sc1, ax=ax1, label='Magnitude $|Z|$ (Log Scale)')
        fig.colorbar(sc2, ax=ax2, label='Phase Angle $\\theta$ (Radians)')

        ax1.set_title(f"Magnitude (Step {frame}/{total_steps})")
        ax2.set_title(f"Phase (Step {frame}/{total_steps})")

        frame_path = os.path.join(abs_save_dir, f"temp_frame_{frame:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        frame_files.append(frame_path)
        if frame % 5 == 0 and frame > 0:
            print(f"   [Rendering] Frame {frame}/{total_steps} saved.")

    if not frame_files:
        print("❌ Error: No frames were generated.")
        return history

    print("   -> Stitching PNGs into final GIF...")
    gif_path = os.path.join(abs_save_dir, "quantum_evolution.gif")
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=150, loop=0
    )

    for f in frame_files:
        os.remove(f)

    print(f"✅ Perfect GIF successfully saved to:\n   --> {gif_path}\n")
    return history

# ==========================================
# 3. EXECUTION PIPELINE WRAPPER
# ==========================================
def execute_pipeline(
        generator_cm, topology_cm, global_field_cm,
        initial_states, initial_fields, warmup_steps, gen_steps, is_dry_run=False
):
    generator_cm.load_initial_state(initial_states, initial_fields)
    topology_cm.warmup([State(initial_states[0])], steps=warmup_steps)
    generator_cm.inject_environment(topology_cm, global_field_cm)
    final_states, final_fields = generator_cm.generate_steps(steps=gen_steps)
    return final_states, final_fields


def run_complex_field_test():
    print("1. Configuring 4D Domain Blueprints...")
    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)

    # Start with 1 particle firing North [x, y, dx, dy]
    initial_states = np.array([
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float64)

    # The single particle holds 100% of the initial probability wave
    initial_fields = np.array([
        [1.0 + 0.0j]
    ], dtype=np.complex128)

    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=angular_coin_transition,
        maximum_step_baking=100,
        max_size=100000,
        state_shape=(4,),  # <--- Upgraded to 4D
        implicit_norm=False,
        explicit_norm=True
    )

    print("2. Spinning up Component Managers...")
    # <--- Dimensions updated from 2 to 4
    global_contract = FieldKernelDataContract(4, 1, algebra, State, None, 100_000)
    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), np.empty((0, 4), dtype=np.float64), np.empty((0, 1), dtype=np.complex128)
    )
    global_field_cm.fill(1.0 + 0.0j)

    # <--- Dimensions updated from 2 to 4
    topology_contract = TopologyKernelDataContract(hardware_quantum_walker_neighbors, State, 100000, 4, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, natural_path_integral_transition
    )

    print("\n[SYSTEM] Bootstrapping JIT Compilers...")
    execute_pipeline(
        generator_cm, topology_cm, global_field_cm,
        initial_states, initial_fields, warmup_steps=1, gen_steps=1, is_dry_run=True
    )
    print("[SYSTEM] Compilers Hot. Engines Ready.\n")

    global_field_cm.fill(1.0 + 0.0j)
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"

    # Run the GIF Renderer and capture the history returned
    history = animate_double_slit_evolution(
        generator_cm, topology_cm, global_field_cm,
        initial_states, initial_fields, total_steps=30, save_dir=save_directory
    )

    # Save the Final Static Plot from the last frame of history
    print("\n   -> Saving final high-res static plot...")
    final_states, final_fields = history[-1]
    plot_and_save_complex_field(final_states, final_fields, save_directory)


if __name__ == "__main__":
    run_complex_field_test()