import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# THE NEW COMPLEX DOD IMPORTS
# Make sure these match the files you just created!
# ==========================================

# ==========================================
# 1. PURE C-SPEED PHYSICS RULES (COMPLEX)
# ==========================================
@njit(cache=True, fastmath=True)
def phase_shift_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    """
    Simulates quantum-like wave interference.
    The phase rotates based on the direction of the random walk.
    """
    dx = s_i[0] - s_j[0]
    dy = s_i[1] - s_j[1]
    theta = np.arctan2(dy, dx)

    # Complex weight: Magnitude 1.0, Phase theta
    c_weight = np.cos(theta) + 1j * np.sin(theta)
    return np.array([c_weight], dtype=np.complex128)


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0], [x, y - 1.0], [x + 1.0, y + 1.0],
        [x - 1.0, y - 1.0], [x + 1.0, y], [x - 1.0, y],[x + 1.0, y-1],[x - 1.0, y+1]
    ], dtype=np.float64)


# ==========================================
# 2. DUAL SUBPLOT VISUALIZATION
# ==========================================
def plot_and_save_complex_field(states: np.ndarray, fields: np.ndarray, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    Z = fields[:, 0]
    magnitude = np.abs(Z)
    phase = np.angle(Z)

    # Filter out empty vacuum states so background phase noise doesn't ruin the plot
    mask = magnitude > 1e-10
    X = states[mask, 0]
    Y = states[mask, 1]
    mag_masked = magnitude[mask]
    phase_masked = phase[mask]

    if len(mag_masked) == 0:
        print("⚠️ Warning: Field mass is zero. Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # SUBPLOT 1: MAGNITUDE
    sc1 = ax1.scatter(X, Y, c=mag_masked, cmap='magma', marker='s', s=25, norm=mcolors.LogNorm())
    fig.colorbar(sc1, ax=ax1, label='Magnitude $|Z|$ (Log Scale)')
    ax1.set_title("Complex Field: Magnitude")
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.2)

    # SUBPLOT 2: PHASE
    # 'twilight' is a cyclic colormap, perfect for mapping angles from -pi to pi
    sc2 = ax2.scatter(X, Y, c=phase_masked, cmap='twilight', marker='s', s=25, vmin=-np.pi, vmax=np.pi)
    fig.colorbar(sc2, ax=ax2, label='Phase Angle $\\theta$ (Radians)')
    ax2.set_title("Complex Field: Phase")
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.2)

    plt.suptitle("2D Complex Field Random Walker (50 Steps)", fontsize=16)

    # EXACT FILENAME REQUESTED
    file_path = os.path.join(abs_save_dir, "complex_random_walker_test.png")

    try:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"✅ Complex Plot successfully saved to:\n   --> {file_path}")
    except Exception as e:
        print(f"❌ Failed to save plot. Error: {e}")


# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
def run_complex_field_test():
    print("1. Configuring Complex OOP Domain Blueprints...")
    # NOTE: Algebra explicitly set to complex128
    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    initial_states = np.array([[0.0, 0.0]], dtype=np.float64)
    initial_fields = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    topology = Topology(reachable_func=None, state_class=State, use_cache=True)

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=phase_shift_transition,
        maximum_step_baking=50,
        max_size=100000,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
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

    # ---> THE TOPOLOGY MANAGER (Stays Float64) <---
    topology_contract = TopologyKernelDataContract(hardware_random_walker_neighbors, State, 100000, 2, np.float64)
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
    print("   -> Expanding Topology Graph & Rebuilding CSR...")
    topology_cm.warmup([State(initial_states[0])], steps=50)
    print(f"   [Phase 1] Topology Warmup: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PHASE 2
    t_start = time.perf_counter()
    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)
    print(f"   [Phase 2] Environment Injection: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PHASE 3
    t_start = time.perf_counter()
    print("   -> Blasting 50 steps via Complex Ping-Pong Buffer...")
    final_states, final_fields = generator_cm.generate_steps(steps=40)
    print(f"   [Phase 3] Complex Field Generation: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PLOT & SAVE
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    plot_and_save_complex_field(final_states, final_fields, save_directory)


if __name__ == "__main__":
    run_complex_field_test()