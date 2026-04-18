import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Callable, Any, Type
from numba import njit

# --- Core Engine Imports ---
from particle_grid_simulator.src.operator.domain.data.operator import GenericOperatorData
from particle_grid_simulator.src.operator.domain.utility.operator import GenericOperatorUtility
from particle_grid_simulator.src.state.domain import State

# --- NEW: Operator CM Imports ---
from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
STEPS = 1  # Number of generator steps for the Field Operator
ITERATIONS = 100  # Number of Operator jumps (trajectory length)
PATH = r"E:\Particle Field Simulation\particle_grid_simulator\test\operator\plots"

T = TypeVar('T')

# ==========================================
# DOD GENERATOR IMPORTS
# ==========================================
from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility
from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract
from particle_grid_simulator.src.generator.kernel.numba.storage.storage_v1 import NumbaCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility


# ==========================================
# PHYSICS & TOPOLOGY RULES
# ==========================================
@njit(cache=True, fastmath=True)
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


def random_walker_neighbors(state: State):
    x, y = state.vector[0], state.vector[1]
    return [State(np.array([x, y + 1.0])), State(np.array([x, y - 1.0]))]


@njit(cache=True, fastmath=True)
def hardware_random_walker_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([
        [x, y + 1.0], [x, y - 1.0],
        [x + 1, y], [x - 1, y]
    ], dtype=np.float64)


# ==========================================
# EVOLUTION FUNCTIONS (THETA)
# ==========================================
def classic_evolution_rule(state: State) -> State:
    """Theta_C: Directly calculates the next state (von neumann 1 dist)"""
    x, y = state.vector[0], state.vector[1]
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dx, dy = random.choice(moves)
    return State(np.array([x + dx, y + dy], dtype=np.float64))


def field_evolution_rule(data_context: tuple) -> State:
    """Theta_F: Probabilistically samples the next state based on the field weights."""
    states_array, fields_array = data_context

    masses = fields_array[:, 0]
    total_mass = np.sum(masses)

    if total_mass == 0:
        raise ValueError("Total field mass is zero. Cannot sample next state.")

    probabilities = masses / total_mass
    chosen_index = np.random.choice(len(states_array), p=probabilities)
    return State(states_array[chosen_index])


# ==========================================
# PLOTTING UTILITY
# ==========================================
def plot_trajectory(history: list, title: str, filename: str):
    abs_save_dir = os.path.normpath(PATH)
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    history_arr = np.array(history)
    X, Y = history_arr[:, 0], history_arr[:, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(X, Y, marker='o', markersize=4, linestyle='-', linewidth=1, color='b', alpha=0.3, label='Trajectory')

    if len(X) > 1:
        u = np.diff(X)
        v = np.diff(Y)
        plt.quiver(X[:-1], Y[:-1], u, v, angles='xy', scale_units='xy', scale=1, color='b', alpha=0.7, width=0.003,
                   zorder=3)

    plt.scatter([X[0]], [Y[0]], color='green', s=100, label='Start', zorder=5)
    plt.scatter([X[-1]], [Y[-1]], color='red', s=100, label='End', zorder=5)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    file_path = os.path.join(abs_save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {file_path}")


# ==========================================
# TEST EXECUTIONS
# ==========================================
def run_classic_test():
    print("\n--- Running Classic Operator Test ---")
    classic_op = GenericOperatorData(classic_evolution_rule, State)
    current_state = State(np.array([0.0, 0.0], dtype=np.float64))

    history = [current_state.vector]

    t_start = time.perf_counter()
    for _ in range(ITERATIONS):
        current_state = GenericOperatorUtility.evolve(classic_op, current_state)
        history.append(current_state.vector)
    t_end = time.perf_counter()

    print(f"Classic execution time for {ITERATIONS} iterations: {(t_end - t_start) * 1000:.2f} ms")
    plot_trajectory(history, f"Classic Random Walker ({ITERATIONS} Iterations)", "classic_walker_trajectory.png")


def run_field_test():
    print("\n--- Running Field Operator Test ---")

    current_state = State(np.array([0.0, 0.0], dtype=np.float64))
    history = [current_state.vector]

    # Initialize Engine Blueprint
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)
    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)
    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=uniform_transition,
        maximum_step_baking=STEPS,
        max_size=100000, state_shape=(2,), implicit_norm=False, explicit_norm=True
    )

    global_contract = FieldKernelDataContract(max_active_states=100000, state_dimensions=2, field_dimensions=1,
                                              algebra=algebra, state_class_ref=State, mapper_func=None)
    topology_contract = TopologyKernelDataContract(hardware_random_walker_neighbors, State, 100000, 2, np.float64)
    generator_contract = GeneratorKernelDataContract.from_domain(generator_data, global_field_dim=1)

    # ==========================================
    # 1. THE BIG BANG (OUTSIDE THE LOOP)
    # ==========================================
    topology_cm = TopologyComponentManager.create_from_raw_data(
        topology_contract, NumbaTopologyStorage(topology_contract),
        NumbaTopologyTranslator(), NumbaTopologyUtility
    )

    print("Baking universe geometry...")
    topology_cm.warmup([current_state], steps=ITERATIONS + 2)

    universe_coords = np.array(topology_cm.fast_refs.handle_map)
    universe_weights = np.ones((len(universe_coords), 1), dtype=np.float64)

    global_field_cm = FieldComponentManager.create_from_raw(
        NumbaKernelFieldUtility, global_contract,
        NumbaFieldKernelStorage(global_contract),
        NumbaFieldTranslator(),
        universe_coords,
        universe_weights
    )

    generator_cm = GeneratorComponentManager(
        generator_contract, NumbaCSRGeneratorStorage(generator_contract),
        GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
        uniform_transition
    )

    generator_cm.inject_environment(topology_cm, global_field_cm)

    # --- THE CHANGE: OPERATOR CM ---
    # Replaced GenericOperatorData with OperatorComponentManager.create_raw
    op_cm = OperatorComponentManager.create_raw(
        evolution_func=field_evolution_rule,
        utility=NumbaOperatorUtility(),
        state_class_ref=State
    )

    t_start = time.perf_counter()

    # ==========================================
    # 2. THE PURE HOT LOOP
    # ==========================================
    for i in range(ITERATIONS):
        generator_cm.clear()

        initial_states_arr = np.array([current_state.vector], dtype=np.float64)
        initial_fields_arr = np.array([[1.0]], dtype=np.float64)
        generator_cm.load_initial_state(initial_states_arr, initial_fields_arr)

        final_states, final_fields = generator_cm.generate_steps(steps=STEPS)

        # --- THE CHANGE: EVOLVE CALL ---
        # Passing the tuple directly to op_cm.evolve
        mask = final_fields[:, 0] > 0
        current_state = op_cm.evolve((final_states[mask], final_fields[mask]))

        history.append(current_state.vector)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{ITERATIONS} field operator jumps...")

    t_end = time.perf_counter()
    print(f"Field execution time for {ITERATIONS} iterations: {(t_end - t_start):.2f} seconds")
    plot_trajectory(history, f"Field Random Walker ({ITERATIONS} Iterations)", "CM_field_walker_trajectory.png")


if __name__ == "__main__":
    run_classic_test()
    run_field_test()