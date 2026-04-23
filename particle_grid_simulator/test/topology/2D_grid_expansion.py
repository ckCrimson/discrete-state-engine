import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit
from numba.typed import List

# --- FDS Architecture Imports ---
from particle_grid_simulator.src.state.domain import State
from particle_grid_simulator.src.topology.component_manager.component_manager import TopologyComponentManager
from particle_grid_simulator.src.topology.kernel.numba.storage.storage_v1 import NumbaTopologyStorage, \
    TopologyKernelDataContract
from particle_grid_simulator.src.topology.kernel.numba.translator.translator_v1 import NumbaTopologyTranslator
from particle_grid_simulator.src.topology.kernel.numba.utility.utility_v1 import NumbaTopologyUtility


# ==========================================
# 1. THE PHYSICS RULE (JIT COMPILED 2D GRID)
# ==========================================
@njit(cache=True)
def move_2d_jit(state_vec: np.ndarray):
    """2D Manhattan Distance Movement (Up, Down, Left, Right)"""
    x, y = state_vec[0], state_vec[1]
    out_list = List()
    out_list.append(np.array([x, y + 1], dtype=np.int32))  # Up
    out_list.append(np.array([x, y - 1], dtype=np.int32))  # Down
    out_list.append(np.array([x - 1, y], dtype=np.int32))  # Left
    out_list.append(np.array([x + 1, y], dtype=np.int32))  # Right
    out_list.append(np.array([x, y ], dtype=np.int32))  #stationary
    return out_list


# ==========================================
# 2. GENERATION SUITE
# ==========================================
def generate_2d_expansion_gif():
    print("🚀 Initializing FDS Topology Engine for 2D Expansion...")

    # --- Setup Architecture ---
    data_contract = TopologyKernelDataContract(
        neighbour_function=move_2d_jit,
        state_class_reference=State,
        initial_capacity=1_000_000,
        dimensions=2,  # Upgraded to 2D
        vector_dtype=np.int32
    )

    manager = TopologyComponentManager.create_from_raw_data(
        data_contract=data_contract,
        storage=NumbaTopologyStorage(data_contract),
        translator=NumbaTopologyTranslator(),
        utility=NumbaTopologyUtility
    )

    start_state = State(np.array([0, 0], dtype=np.int32))
    max_steps = 100

    print("🔥 Warming up engine...")
    manager.warmup([start_state], steps=max_steps)

    print("🧠 Computing Multi-Step Frontiers & Basins...")

    # Pre-compute all states to ensure rendering is fast
    historical_frontiers = []

    for l in range(max_steps + 1):
        # We query the engine for the raw numerical arrays (return_state_class=False)
        frontier_raw = manager.get_reachable_multi_step_frontier(start_state, steps=l, return_state_class=False)
        historical_frontiers.append(frontier_raw)

    print(f"✅ Calculation Complete. Max Frontier Size: {len(historical_frontiers[-1])}")
    print("🎬 Rendering GIF...")

    # --- 3. PLOTTING AND ANIMATION ---
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#050510')
    ax.set_facecolor('#050510')

    # Set limits slightly larger than the max step to fit the final diamond
    ax.set_xlim(-max_steps - 5, max_steps + 5)
    ax.set_ylim(-max_steps - 5, max_steps + 5)
    ax.set_title("2D State Space Expansion (CM Driven)", color='white', pad=15)
    ax.grid(True, color='#202030', linestyle='--', alpha=0.5)
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    # Scatter objects for Basin (Green) and Frontier (Red)
    basin_scat = ax.scatter([], [], color='#44FF44', s=10, alpha=0.3, zorder=1)  # Visited
    frontier_scat = ax.scatter([], [], color='#FF4444', s=15, edgecolors='white', linewidths=0.5,
                               zorder=2)  # Current Edge

    # We accumulate the basin over time
    current_basin = []

    def update(frame):
        if frame == 0:
            current_basin.clear()

        # The current frontier from our pre-computed FDS data
        current_frontier = historical_frontiers[frame]

        # The basin is everything from previous frames
        if frame > 0:
            previous_frontier = historical_frontiers[frame - 1]
            current_basin.extend(previous_frontier)

        # Update scatters
        if len(current_basin) > 0:
            basin_np = np.array(current_basin)
            basin_scat.set_offsets(basin_np)

        if len(current_frontier) > 0:
            frontier_scat.set_offsets(current_frontier)

        ax.set_title(f"2D State Space Expansion (l={frame})", color='white')
        return basin_scat, frontier_scat

    # Create animation (100 frames, 5 fps = 200ms interval)
    ani = FuncAnimation(fig, update, frames=max_steps + 1, interval=200, blit=True)

    output_dir = r"./plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "2d_topology_expansion.gif")

    ani.save(save_path, writer=PillowWriter(fps=10))
    print(f"✅ GIF successfully saved to: {save_path}")


if __name__ == "__main__":
    generate_2d_expansion_gif()