import numpy as np
from typing import Iterable

from particle_grid_simulator.src.field.domain.data.field_algebra import FieldAlgebra
from particle_grid_simulator.src.field.domain.data.field_mapper import FieldMapper
from particle_grid_simulator.src.generator.domain.data.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorData
from particle_grid_simulator.src.generator.domain.utilities.generic_markovian_field_generator import \
    GenericMarkovianFieldGeneratorUtility
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.domain.topology_domain import Topology

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_and_save_field(states: np.ndarray, fields: np.ndarray, save_dir: str):
    """
    Generates a 2D heatmap of the field distribution using a Logarithmic scale
    to handle the exponential growth of the unnormalized field.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract coordinates and scalar values
    X = states[:, 0]
    Y = states[:, 1]
    Z = fields[:, 0]  # Extract the scalar magnitude from the field vector

    plt.figure(figsize=(10, 8))

    # Use a scatter plot with square markers ('s') to simulate a grid.
    # LogNorm is critical here to see the diffusion ripples.
    sc = plt.scatter(
        X, Y,
        c=Z,
        cmap='magma',
        marker='s',
        s=15,  # Adjust marker size if grid looks too dense or sparse
        norm=mcolors.LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max())
    )

    plt.colorbar(sc, label='Field Mass (Log Scale)')
    plt.title("2D Random Walker Field Distribution (50 Steps)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Ensure axes are scaled equally so the grid is square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.2)

    # Save the figure
    file_path = os.path.join(save_dir, "random_walker_50_steps_domain.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot successfully saved to: {file_path}")
# 2. The Topology Rule
def random_walker_neighbors(state: State) -> Iterable[State]:
    x, y = state.vector[0], state.vector[1]
    return [
        State(np.array([x, y + 1.0])),  # <0, 1>
        State(np.array([x, y - 1.0])),  # <0, -1>
        State(np.array([x + 1.0, y + 1.0])),  # <1, 1>
        State(np.array([x - 1.0, y - 1.0]))  # <-1, -1>
    ]


# 3. The Transition Rule
def uniform_transition(s_j: np.ndarray, s_i: np.ndarray) -> np.ndarray:
    return np.array([1.0], dtype=np.float64)


# ==========================================
# TEST EXECUTION
# ==========================================
def run_random_walker_test():
    print("Initializing Field Algebra...")
    algebra = FieldAlgebra(dimensions=1, dtype=np.float64)

    print("Setting up Initial Configuration (Mapper)...")
    # Initial state X = <0,0> with field value 1.0
    mapper = FieldMapper(algebra=algebra, state_class_ref=State)
    mapper.set_fields_at([np.array([0.0, 0.0])], [np.array([1.0])])

    # Global vacuum (empty mapper)
    global_mapper = FieldMapper(algebra=algebra, state_class_ref=State)

    print("Baking Topology (Maximum Step Size: 50)...")
    topology = Topology(reachable_func=random_walker_neighbors, state_class=State, use_cache=True)

    # Calculate max unique states for a 2D lattice over 50 steps to size our buffers safely.
    # Growth is roughly O(L^2), so 10,000 is plenty of headroom to prevent reallocation.
    max_capacity = 10000

    generator_data = GenericMarkovianFieldGeneratorData(
        mapper=mapper,
        topology=topology,
        transition_function=uniform_transition,
        maximum_step_baking=50,
        max_size=max_capacity,
        state_shape=(2,),
        implicit_norm=True,
        explicit_norm=False
    )

    print("Running 50-step Ping-Pong Execution...")
    final_states, final_fields = GenericMarkovianFieldGeneratorUtility.generate_multi_step_field(
        initial_mapper=mapper,
        global_mapper=global_mapper,
        generator_data=generator_data,
        steps=50
    )

    print("\n--- TEST COMPLETE ---")
    print(f"Total Unique Active States after 50 steps: {len(final_states)}")
    print(f"Total Field Mass: {np.sum(final_fields)}")

    # Optional: Find the peak field value (usually at or near the origin)
    max_idx = np.argmax(final_fields)
    print(f"Peak Field Value: {final_fields[max_idx][0]} at Coordinate: {final_states[max_idx]}")

    # NEW: Trigger the visualization
    save_directory = r"E:\Particle Field Simulation\particle_grid_simulator\test\generator\plot"
    print("Generating and saving plot...")
    plot_and_save_field(final_states, final_fields, save_directory)


if __name__ == "__main__":
    run_random_walker_test()