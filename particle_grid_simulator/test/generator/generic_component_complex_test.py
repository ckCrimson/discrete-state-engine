import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

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
# 1. PURE C-SPEED PHYSICS RULES (COMPLEX)
# ==========================================
@njit(cache=True, fastmath=True)
def phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """
    Simulates quantum-like wave interference with conserved probability.
    """
    dx = s_i[0] - s_j[0]
    dy = s_i[1] - s_j[1]
    theta = np.arctan2(dy, dx)

    # Scale amplitude by 1 / sqrt(8) to conserve probability across 8 neighbors
    amplitude = 1.0 / np.sqrt(8.0)

    # Complex weight: Magnitude 0.353, Phase theta
    c_weight = amplitude * (np.cos(theta) + 1j * np.sin(theta))

    return np.array([c_weight], dtype=np.complex128)

@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0], [x, y - 1.0], [x + 1.0, y + 1.0],
        [x - 1.0, y - 1.0], [x + 1.0, y], [x - 1.0, y], [x + 1.0, y - 1], [x - 1.0, y + 1]
    ], dtype=np.float64)


# ==========================================
# 2. GIF ANIMATION GENERATOR
# ==========================================
def animate_and_save_complex_field(history: list, save_dir: str, max_steps: int):
    print("\n🎬 Rendering Complex Field GIF...")
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#050510')

    # Setup Axes
    for ax, title in zip([ax1, ax2], ["Magnitude $|Z|$ (Log)", "Phase Angle $\\theta$ (Radians)"]):
        ax.set_facecolor('#050510')
        ax.set_xlim(-max_steps - 2, max_steps + 2)
        ax.set_ylim(-max_steps - 2, max_steps + 2)
        ax.set_title(f"Complex Field: {title}", color='white', pad=15)
        ax.set_aspect('equal')
        ax.grid(True, color='#202030', linestyle='--', alpha=0.5)
        ax.tick_params(colors='white')

    plt.suptitle("2D Complex Field Generation", fontsize=16, color='white')

    # Initialize empty scatters (Explicitly adding c=[])
    sc1 = ax1.scatter([], [], c=[], cmap='magma', marker='s', s=25, norm=mcolors.LogNorm(vmin=1e-6, vmax=1e3))
    sc2 = ax2.scatter([], [], c=[], cmap='twilight', marker='s', s=25, vmin=-np.pi, vmax=np.pi)
    # Add Colorbars
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color='white')
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color='white')

    def update(frame):
        states, fields = history[frame]
        Z = fields[:, 0]
        magnitude = np.abs(Z)
        phase = np.angle(Z)

        mask = magnitude > 1e-10
        X = states[mask, 0]
        Y = states[mask, 1]
        mag_masked = magnitude[mask]
        phase_masked = phase[mask]

        if len(X) > 0:
            sc1.set_offsets(np.c_[X, Y])
            sc1.set_array(mag_masked)

            sc2.set_offsets(np.c_[X, Y])
            sc2.set_array(phase_masked)

        return sc1, sc2

    # Animate and Save
    ani = FuncAnimation(fig, update, frames=len(history), blit=False)
    file_path = os.path.join(abs_save_dir, "complex_field_generation.gif")

    try:
        ani.save(file_path, writer=PillowWriter(fps=6))
        plt.close('all')
        print(f"✅ Complex GIF successfully saved to:\n   --> {file_path}")
    except Exception as e:
        print(f"❌ Failed to save GIF. Error: {e}")


# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
def run_complex_field_test():
    print("1. Configuring Complex OOP Domain Blueprints...")
    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    initial_states = np.array([[0.0, 0.0]], dtype=np.float64)
    initial_fields = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    topology = Topology(reachable_func=None, state_class=State, use_cache=True)
    TOTAL_STEPS = 40

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=phase_shift_transition,
        maximum_step_baking=TOTAL_STEPS + 10,
        max_size=100_000,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=False
    )

    print("2. Spinning up Complex Hardware Component Managers...")

    # ---> THE COMPLEX FIELD MANAGER <---
    global_contract = FieldKernelDataContract(
        state_dimensions=2, field_dimensions=1, algebra=algebra,
        state_class_ref=State, mapper_func=None, initial_capacity=100_000
    )
    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64), np.empty((0, 1), dtype=np.complex128)
    )
    global_field_cm.fill(1.0 + 0.0j)

    # ---> THE TOPOLOGY MANAGER <---
    topology_contract = TopologyKernelDataContract(hardware_random_walker_neighbors, State, 100_000, 2, np.float64)
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    # ---> THE COMPLEX GENERATOR MANAGER <---
    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, phase_shift_transition
    )

    print("\n===========================================")
    print("      COMPLEX TELEMETRY PIPELINE           ")
    print("===========================================")

    generator_cm.load_initial_state(initial_states, initial_fields)

    # PHASE 1
    t_start = time.perf_counter()
    print(f"   -> Expanding Topology Graph ({TOTAL_STEPS} steps)...")
    topology_cm.warmup([State(initial_states[0])], steps=TOTAL_STEPS)
    print(f"   [Phase 1] Topology Warmup: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PHASE 2
    t_start = time.perf_counter()
    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)
    print(f"   [Phase 2] Environment Injection: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PHASE 3: Loop step-by-step to record history for the GIF
    print(f"   -> Executing {TOTAL_STEPS} steps (Recording History)...")
    history = [(initial_states.copy(), initial_fields.copy())]

    t_start = time.perf_counter()
    for _ in range(TOTAL_STEPS):
        states, fields = generator_cm.generate_steps(steps=1)
        history.append((states.copy(), fields.copy()))
    print(f"   [Phase 3] Field Generation (Iterative): {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PLOT & SAVE
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    animate_and_save_complex_field(history, save_directory, max_steps=TOTAL_STEPS)


if __name__ == "__main__":
    run_complex_field_test()