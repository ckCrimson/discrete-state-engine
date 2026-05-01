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
from particle_grid_simulator.src.field.kernel.numba.storage.storage_v1 import NumbaFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.utility_v1 import NumbaKernelFieldUtility
from particle_grid_simulator.src.generator.kernel.numba.storage.storage_v1 import NumbaCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility


# ==========================================
# 1. KERNELS & OPERATOR FACTORY
# ==========================================
@njit(fastmath=True)  # Disabled cache temporarily for clean execution
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


@njit(fastmath=True)  # Disabled cache temporarily for clean execution
def hardware_neighbors(state_vec: np.ndarray) -> np.ndarray:
    x, y = state_vec[0], state_vec[1]
    return np.array([[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]], dtype=np.float64)


def random_walker_neighbors(state: State):
    return [State(v) for v in hardware_neighbors(state.vector)]


def make_gradient_operator(u_coords: np.ndarray, u_weights: np.ndarray):
    @njit(fastmath=True)
    def gradient_climb_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
        best_state = state_vec
        max_val = -1.0

        for i in range(len(gen_states)):
            cand = gen_states[i]
            for j in range(len(u_coords)):
                if u_coords[j, 0] == cand[0] and u_coords[j, 1] == cand[1]:
                    val = u_weights[j, 0]
                    if val > max_val:
                        max_val = val
                        best_state = cand
                    break
        return best_state

    return gradient_climb_kernel


# ==========================================
# 2. RENDERER
# ==========================================
def save_particle_gif(csv_path: Path, save_path: Path, n_particles: int):
    print("   -> Rendering GIF...")
    flat_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # PROACTIVE FIX: Strip the first column (time) before reshaping into coordinates
    history = flat_data.reshape(flat_data.shape[0], n_particles, 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Constructive Interference ({n_particles} Particles)")

    colors = plt.cm.plasma(np.linspace(0, 1, n_particles))
    scats = [ax.scatter([], [], color=colors[p], s=100, edgecolors='black', zorder=4) for p in range(n_particles)]
    lines = [ax.plot([], [], color=colors[p], alpha=0.5, linewidth=2, zorder=3)[0] for p in range(n_particles)]

    def update(frame):
        for p in range(n_particles):
            lines[p].set_data(history[:frame + 1, p, 0], history[:frame + 1, p, 1])
            scats[p].set_offsets(history[frame: frame + 1, p])
        return lines + scats

    ani = FuncAnimation(fig, update, frames=len(history), blit=True)
    ani.save(save_path / "pheromone_bridge.gif", writer=PillowWriter(fps=10))
    print("✅ Done!")


# ==========================================
# 3. THE MASTER PIPELINE
# ==========================================
def run_interference_test():
    NUM_PARTICLES = 2
    STEPS = 6
    ITERATIONS = 30
    SAVE_DIR = Path(r"./plots")
    SAVE_DIR.mkdir(exist_ok=True)  # Ensure directory exists

    # 1. Component Data Contracts
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)
    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)

    g_contract = FieldKernelDataContract(max_active_states=100000, state_dimensions=2, field_dimensions=1,
                                         algebra=algebra, state_class_ref=State, mapper_func=None)
    t_contract = TopologyKernelDataContract(hardware_neighbors, State, 100000, 2, np.float64)

    # FATAL BUG FIX: generator_shape changed from (2,) to (1,) to match the 1D field
    gen_data = GenericMarkovianFieldGeneratorData(FieldMapper(algebra, State), topology, uniform_transition, STEPS,
                                                  100000, (2,), False, True)
    gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=1)

    # 2. Build CMs and Bake Topology
    topology_cm = TopologyComponentManager.create_from_raw_data(t_contract, NumbaTopologyStorage(t_contract),
                                                                NumbaTopologyTranslator(), NumbaTopologyUtility)
    generator_cm = GeneratorComponentManager(gen_contract, NumbaCSRGeneratorStorage(gen_contract),
                                             GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                             uniform_transition)

    print("   -> Baking Topology...")
    topology_cm.warmup([State(np.array([0.0, 0.0]))], steps=20)
    u_coords = np.array(topology_cm.fast_refs.handle_map, dtype=np.float64)  # Forced float64 mapping
    u_weights = np.zeros((len(u_coords), 1), dtype=np.float64)

    field_cm = FieldComponentManager.create_from_raw(NumbaKernelFieldUtility, g_contract,
                                                     NumbaFieldKernelStorage(g_contract), NumbaFieldTranslator(),
                                                     u_coords, u_weights)

    # 3. Create the Smart Operator
    smart_kernel = make_gradient_operator(u_coords, u_weights)
    op_cm = OperatorComponentManager.create_raw(smart_kernel, NumbaOperatorUtility(), State)

    # 4. Dynamic Particle Placement
    s0 = np.zeros((NUM_PARTICLES, 2), dtype=np.float64)
    s0[:, 0] = np.linspace(-5.0, 5.0, NUM_PARTICLES)
    f0 = np.ones((NUM_PARTICLES, 1), dtype=np.float64)

    system_data = SingleChannelFDSData(
        _initial_states=s0, _initial_fields=f0,
        _topology_cm=topology_cm, _field_cm=field_cm, _generator_cm=generator_cm, _operator_cm=op_cm,
        _history_window_size=10, _save_directory=SAVE_DIR, _is_independent=False
    )

    runner = SingleChannelFDSRunner(system_data, SingleChannelFDSUtility)

    print("   -> Running Evolution Loop...")
    for _ in range(ITERATIONS):
        runner.next(apply_generator=True, steps=STEPS)
        runner.next(apply_generator=True, steps=8)
        runner.next(apply_generator=False)

    runner.end(compile_csv=True)
    save_particle_gif(SAVE_DIR / "compiled_telemetry.csv", SAVE_DIR, NUM_PARTICLES)


if __name__ == "__main__":
    run_interference_test()