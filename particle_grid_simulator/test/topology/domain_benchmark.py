import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.domain.topology_domain import Topology
from particle_grid_simulator.src.topology.domain.utility.utility import TopologyUtility


# --- 1. Import your actual project modules here ---
# from particle_grid_simulator.topology import Topology, TopologyUtility, State
# (I am assuming these exist based on your snippet.
#  If State doesn't have a dummy implementation for this script to run standalone,
#  make sure it's imported from your codebase!)

# --- 2. Define the rules of the universe ---
def move_1d(s) -> Iterable:  # Assuming 's' is of type State
    x = s.vector[0]
    # Return new states with numpy arrays
    # Note: State class needs to be defined/imported for this to work
    return [State(np.array([x - 1])), State(np.array([x + 1]))]


# --- 3. The Benchmarking Function ---
def generate_1d_topology_scaling_plot():
    print("🧪 Running 1D Topology Scaling Benchmark...\n")

    # Instantiate the Topology and starting point
    universe = Topology(reachable_func=move_1d, state_class=State)
    s_0 = State(np.array([0]))

    # Define the step limits (l values) we want to test on the X-axis
    # For a 1D grid, you might want to push these numbers higher
    # since 1D scales very efficiently (O(N)).
    step_values = [10, 50, 100, 200, 500]

    times = []
    states_list = []

    plt.figure(figsize=(12, 7))

    for steps in step_values:
        print(f"Calculating basin for l={steps} steps...")

        # Start timer
        start_time = time.time()

        # Run your actual topology utility
        basin = TopologyUtility.get_multi_step_reachable_basin(universe, s_0, l=steps)

        # Stop timer
        end_time = time.time()

        # Record metrics
        exec_time = end_time - start_time
        num_states = len(basin.states)

        times.append(exec_time)
        states_list.append(num_states)

    # --- 4. Plotting the results ---
    plt.plot(step_values, times, marker='o', color='b', label='1D Universe')

    # Annotate each point with (Time, States)
    for i, steps in enumerate(step_values):
        annotation_text = f"({times[i]:.4f}s, {states_list[i]} st)"
        plt.annotate(
            annotation_text,
            (steps, times[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )

    # Formatting the Graph
    plt.title('Topology Scaling: Computation Time vs Steps ($l$)', fontsize=14)
    plt.xlabel('Number of Steps ($l$)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Saving to the specified path ---
    # Using a raw string for the absolute Windows path
    output_dir = r"E:\Particle Field Simulation\particle_grid_simulator\test\topology\plots"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "scaling_wrt_l.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    print(f"\n✅ Plot successfully generated and saved to:\n{file_path}")
    plt.close()

if __name__ == "__main__":
    generate_1d_topology_scaling_plot()