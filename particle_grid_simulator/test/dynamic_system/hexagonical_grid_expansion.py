import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors

# --- DOD & Domain Imports ---
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData

from particle_grid_simulator.src.field.component_manager.component_manager import FieldComponentManager
from particle_grid_simulator.src.field.interfaces.storage import FieldKernelDataContract
from particle_grid_simulator.src.field.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexFieldKernelStorage
from particle_grid_simulator.src.field.kernel.numba.translator.translator_v1 import NumbaFieldTranslator
from particle_grid_simulator.src.field.kernel.numba.utility.complex_field_utility_v2 import NumbaComplexUtility

from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility

from particle_grid_simulator.src.generator.component_manager.component_manager import GeneratorComponentManager
from particle_grid_simulator.src.generator.kernel.numba.storage.complex_field_storage_v2 import \
    NumbaComplexCSRGeneratorStorage
from particle_grid_simulator.src.generator.kernel.numba.translator.generic_translator_v2 import \
    GenericGeneratorTranslator
from particle_grid_simulator.src.generator.kernel.numba.utility.generic_utility_v2 import GenericGeneratorKernelUtility
from particle_grid_simulator.src.generator.iterfaces.storage import GeneratorKernelDataContract

from particle_grid_simulator.src.operator.component_manager.component_manager import OperatorComponentManager
from particle_grid_simulator.src.operator.kernel.numba.utility.kernel_v1 import NumbaOperatorUtility

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
STEPS = 30
ITERATIONS = 3
N_PHASE = 4  # The phase added per step is pi / N_PHASE
PATH = r"./plots"

# Complex Gaussian Parameters
CENTER_Q, CENTER_R = 5.0, 5.0
SIGMA = 10.0

# Pre-calculate the constant step weight
STEP_WEIGHT = np.exp(1j * np.pi / N_PHASE)


# ==========================================
# 2. SIMPLE KERNELS
# ==========================================

@nb.njit(fastmath=True)
def simple_hex_topology(state_vec: np.ndarray) -> np.ndarray:
    q, r = state_vec[0], state_vec[1]
    dq = np.array([1.0, 0.0, -1.0, -1.0, 0.0, 1.0], dtype=nb.float64)
    dr = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0], dtype=nb.float64)

    out = np.empty((6, 2), dtype=nb.float64)
    for i in range(6):
        out[i, 0] = q + dq[i]
        out[i, 1] = r + dr[i]
    return out


@nb.njit(fastmath=True)
def simple_phase_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    # Every path gets the same constant complex phase shift
    return np.array([STEP_WEIGHT], dtype=nb.complex128)


@nb.njit(fastmath=True)
def born_collapse_kernel(state_vec: np.ndarray, gen_states: np.ndarray, gen_fields: np.ndarray) -> np.ndarray:
    # Standard Born Rule: P = |psi|^2
    amplitudes = gen_fields[:, 0]
    probs = amplitudes.real ** 2 + amplitudes.imag ** 2
    total = np.sum(probs)

    if total > 1e-15:
        r = np.random.random() * total
        acc = 0.0
        for i in range(len(probs)):
            acc += probs[i]
            if r <= acc:
                return gen_states[i]
    return state_vec


# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
def run_gaussian_interference_demo():
    # Dynamic Capacity
    radius = (STEPS * ITERATIONS) + 10
    MAX_CAPACITY = int((3 * radius * (radius + 1) + 1) * 1.5)

    algebra = FieldAlgebra(dimensions=1, dtype=np.complex128)
    topology = Topology(reachable_func=None, state_class=State)

    # Physics Setup
    gen_data = GenericMarkovianFieldGeneratorData(
        mapper=FieldMapper(algebra, State),
        topology=topology,
        transition_function=simple_phase_transition,
        maximum_step_baking=STEPS,
        max_size=MAX_CAPACITY,
        state_shape=(2,),
        implicit_norm=False,
        explicit_norm=False  # CRITICAL: Let interference happen naturally
    )

    # Component Managers
    topo_contract = TopologyKernelDataContract(simple_hex_topology, State, MAX_CAPACITY, 2, np.float64)
    topo_cm = TopologyComponentManager.create_from_raw_data(topo_contract, NumbaTopologyStorage(topo_contract),
                                                            NumbaTopologyTranslator(), NumbaTopologyUtility)

    gen_contract = GeneratorKernelDataContract.from_domain(gen_data, global_field_dim=1)
    gen_cm = GeneratorComponentManager(gen_contract, NumbaComplexCSRGeneratorStorage(gen_contract),
                                       GenericGeneratorTranslator(), GenericGeneratorKernelUtility,
                                       simple_phase_transition)

    op_cm = OperatorComponentManager.create_raw(evolution_func=born_collapse_kernel, utility=NumbaOperatorUtility(),
                                                state_class_ref=State)

    # Initial State at (0,0)
    current_state = np.zeros((1, 2), dtype=np.float64)
    topo_cm.warmup([State(s) for s in current_state], steps=radius)

    # Build Gaussian Global Field
    u_coords = np.array(topo_cm.fast_refs.handle_map)
    u_weights = np.zeros((len(u_coords), 1), dtype=np.complex128)
    for i in range(len(u_coords)):
        q, r = u_coords[i][0], u_coords[i][1]
        dist_sq = (q - CENTER_Q) ** 2 + (r - CENTER_R) ** 2
        # Complex Gaussian: Amplitude fades, and phase rotates with distance
        val = np.exp(-dist_sq / (2 * SIGMA ** 2)) * np.exp(1j * dist_sq / SIGMA)
        u_weights[i, 0] = val

    field_contract = FieldKernelDataContract(
        max_active_states=MAX_CAPACITY,
        state_dimensions=2,
        field_dimensions=1,
        algebra=algebra,
        state_class_ref=State
    )
    global_field_cm = FieldComponentManager.create_from_raw(NumbaComplexUtility, field_contract,
                                                            NumbaComplexFieldKernelStorage(field_contract),
                                                            NumbaFieldTranslator(), u_coords, u_weights)

    gen_cm.inject_environment(topo_cm, global_field_cm)

    # Run Loop
    history_wave = []
    history_particle = [current_state.copy()]

    for i in range(ITERATIONS):
        gen_cm.clear()
        gen_cm.load_initial_state(current_state, np.ones((1, 1), dtype=np.complex128))

        for s in range(STEPS):
            states, fields = gen_cm.generate_steps(1)
            history_wave.append((states.copy(), np.abs(fields[:, 0]) ** 2))
            history_particle.append(current_state.copy())

        # Collapse
        m = len(states)
        if m > 0:
            b_s = np.ascontiguousarray(np.broadcast_to(states, (1, m, 2)))
            b_f = np.ascontiguousarray(np.broadcast_to(fields, (1, m, 1)))
            op_cm.evolve_batch_inplace(current_state, b_s, b_f)

    # Animation (Simplified for standard Cartesian projection)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        ax.set_facecolor('#050510')
        states, masses = history_wave[frame]
        x = states[:, 0] + 0.5 * states[:, 1]
        y = (np.sqrt(3.0) / 2.0) * states[:, 1]
        ax.scatter(x, y, c=masses, cmap='inferno', s=20, norm=mcolors.Normalize(0, 0.1))
        ax.set_title(f"Quantum Gaussian Interference - Step {frame}")

    ani = FuncAnimation(fig, update, frames=len(history_wave))
    ani.save(os.path.join(PATH, "quantum_gaussian_demo.gif"), writer=PillowWriter(fps=15))
    plt.show()


if __name__ == "__main__":
    run_gaussian_interference_demo()