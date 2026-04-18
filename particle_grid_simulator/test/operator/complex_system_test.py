import os
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
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
# 1. PURE C-SPEED PHYSICS RULES (COMPLEX)
# ==========================================
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
# 2. OPERATOR RULE (OBSERVER)
# ==========================================
def complex_field_observer_rule(data_context: tuple) -> State:
    """Collapses Complex Amplitudes using the Born Rule (|z|^2)."""
    states_array, fields_array = data_context
    amplitudes = fields_array[:, 0]
    masses = amplitudes.real ** 2 + amplitudes.imag ** 2

    total_mass = np.sum(masses)
    if total_mass <= 1e-12:
        return State(states_array[0])

    probabilities = masses / total_mass
    chosen_index = np.random.choice(len(states_array), p=probabilities)
    return State(states_array[chosen_index])


# ==========================================
# 3. VISUALIZATION
# ==========================================
def plot_trajectory(history: list, save_dir: str):
    abs_save_dir = os.path.normpath(save_dir)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    history_arr = np.array(history)
    X, Y = history_arr[:, 0], history_arr[:, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(X, Y, marker='o', markersize=4, linestyle='-', linewidth=1, color='purple', alpha=0.5,
             label='Quantum Trajectory')

    plt.scatter([X[0]], [Y[0]], color='green', s=100, label='Start', zorder=5)
    plt.scatter([X[-1]], [Y[-1]], color='red', s=100, label='End', zorder=5)

    plt.title(f"Complex Operator Trajectory ({len(history) - 1} Jumps)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    file_path = os.path.join(abs_save_dir, "complex_operator_trajectory.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"✅ Trajectory Plot saved to:\n   --> {file_path}")


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def run_complex_operator_test(steps: int = 5, iterations: int = 20):
    print(f"1. Configuring Complex DOD Blueprint (Steps: {steps}, Iterations: {iterations})...")

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State, use_cache=True)



    print("2. Spinning up Complex Hardware Component Managers...")

    # Field Manager
    # 1. Field Manager Contract (Bump to 200,000)
    global_contract = FieldKernelDataContract(
        state_dimensions=2, field_dimensions=1, algebra=algebra,
        state_class_ref=State, mapper_func=None, initial_capacity=200000
    )

    # 2. Topology Manager Contract (Bump to 200,000)
    topology_contract = TopologyKernelDataContract(
        hardware_random_walker_neighbors, State, 200000, 2, np.float64
    )

    # 3. Generator Manager Contract
    # To fix this cleanly, ensure max_size inside your Domain Data is updated
    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=phase_shift_transition,
        maximum_step_baking=steps,
        max_size=200000,  # Bump to 200,000
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=True
    )
    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)
    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaComplexUtility, global_contract, NumbaComplexFieldKernelStorage(global_contract),
        NumbaFieldTranslator(), np.empty((0, 2), dtype=np.float64), np.empty((0, 1), dtype=np.complex128)
    )
    global_field_cm.fill(1.0 + 0.0j)

    # Topology Manager
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    # Generator Manager
    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaComplexCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility, phase_shift_transition
    )

    # Operator Manager
    op_cm = OperatorComponentManager.create_raw(
        evolution_func=complex_field_observer_rule,
        utility=NumbaOperatorUtility(),
        state_class_ref=State
    )

    print("\n===========================================")
    print("      COMPLEX TELEMETRY PIPELINE           ")
    print("===========================================")

    current_state = State(np.array([0.0, 0.0], dtype=np.float64))
    history = [current_state.vector]

    # Warmup topology enough to cover max possible distance (steps * iterations)
    t_start = time.perf_counter()
    safe_warmup = iterations + 10
    print(f"   -> Expanding Topology Graph (Warmup depth: {safe_warmup})...")
    topology_cm.warmup([current_state], steps=safe_warmup)

    print("   -> Wiring Hardware Arrays & Complex JIT Math Callables...")
    generator_cm.inject_environment(topology_cm, global_field_cm)

    print(f"   -> Starting Observer Loop...")
    for i in range(iterations):
        generator_cm.clear()

        # Inject the current particle position as a complex seed
        initial_states = np.array([current_state.vector], dtype=np.float64)
        initial_fields = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        generator_cm.load_initial_state(initial_states, initial_fields)

        # 1. Expand Wave
        final_states, final_fields = generator_cm.generate_steps(steps=steps)

        # 2. Collapse Wave (Apply Operator)
        current_state = op_cm.evolve((final_states, final_fields))
        history.append(current_state.vector)

    print(f"   [Phase 3] Total Loop Execution: {(time.perf_counter() - t_start) * 1000:.4f} ms")

    # PLOT & SAVE
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"
    plot_trajectory(history, save_directory)


if __name__ == "__main__":
    # You can change inputs here
    run_complex_operator_test(steps=30, iterations=50)