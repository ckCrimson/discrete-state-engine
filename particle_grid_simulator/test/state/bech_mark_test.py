import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Iterable, Callable, Set

# --- Core Engine Imports ---
from hpc_ecs_core.src.hpc_ecs_core.interfaces import KernelDataContract
from particle_grid_simulator.src.state.component_manager.component_manager import StateComponentManager
from particle_grid_simulator.src.state.domain import StateSpace, StateSpaceUtility, State


# ==========================================
# 2. THE PIPELINES
# ==========================================
def run_oop_pipeline(l: int):
    x, y = np.meshgrid(np.arange(-l, l + 1), np.arange(-l, l + 1))
    coords = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    ids = np.arange(1, len(coords) + 1, dtype=np.int64)

    start_time = time.perf_counter()
    original_states = {State(vector=np.array([ids[i], coords[i, 0], coords[i, 1]])) for i in range(len(ids))}
    space = StateSpace(states=set(original_states))

    def scale_state(s: State) -> State:
        return State(vector=np.array([s.vector[0], s.vector[1] * 2.0, s.vector[2] * 2.0]))

    StateSpaceUtility.map_inplace(space, scale_state)

    valid_states = [s for s in space.states if 5.0 <= np.linalg.norm(s.vector[1:3]) <= 13.0]
    StateSpaceUtility.intersection_inplace(space, valid_states)
    StateSpaceUtility.union_inplace(space, original_states)

    return (time.perf_counter() - start_time) * 1000


def run_dual_flow_pipeline(l: int, kernel_type: str):
    x, y = np.meshgrid(np.arange(-l, l + 1), np.arange(-l, l + 1))
    coords = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    ids = np.arange(1, len(coords) + 1, dtype=np.int64)
    raw_init_data = {'ids': ids, 'coords': coords}

    start_time = time.perf_counter()
    manager = StateComponentManager.build(
        initial_data=raw_init_data, dimensions=2, max_count=int(len(ids) * 2.5), kernel=kernel_type
    )

    def scale_coords(ids_arr, coords_arr): return coords_arr * 2.0

    manager.map_in_place(scale_coords)

    def norm_filter(coords_arr):
        norms = (coords_arr[:, 0] ** 2 + coords_arr[:, 1] ** 2) ** 0.5
        return (norms >= 5.0) & (norms <= 13.0)

    manager.filter_in_place(norm_filter)

    offset_original_data = {'ids': ids + 9000000, 'coords': coords}
    manager.union_in_place(offset_original_data)

    # Force a sync so we accurately measure JAX async dispatch time
    _ = manager.count

    return (time.perf_counter() - start_time) * 1000


# ==========================================
# 3. STATISTICAL RUNNER & COMPOSITE PLOTTER
# ==========================================
def generate_scaling_plot(runs_per_size=10):
    print(f"🔥 BOOTING STATISTICAL ARENA ({runs_per_size} Runs Per Size) 🔥")
    print("--> Warming up Dual Flow Kernels (Numba & JAX)...")
    run_dual_flow_pipeline(5, 'Numba')
    run_dual_flow_pipeline(5, 'jax')
    print("✅ Warmup Complete.\n")

    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 25, 30, 50, 80, 100, 120]
    results = []

    for l in test_values:
        points = (2 * l + 1) ** 2
        print(f"Testing Size: {points:<8} states...", end="", flush=True)

        for _ in range(runs_per_size):
            # 1. Pure OOP
            oop_time = run_oop_pipeline(l)
            results.append(
                {'Number of States': points, 'Architecture': 'Pure OOP (Sets)', 'Execution Time (ms)': oop_time})

            # 2. Dual Flow (CPU / Numba)
            numba_time = run_dual_flow_pipeline(l, 'Numba')
            results.append(
                {'Number of States': points, 'Architecture': 'Dual Flow (Numba)', 'Execution Time (ms)': numba_time})

            # 3. Dual Flow (GPU / JAX)
            jax_time = run_dual_flow_pipeline(l, 'jax')
            results.append(
                {'Number of States': points, 'Architecture': 'Dual Flow (JAX)', 'Execution Time (ms)': jax_time})

        print(" Done!")

    # --- ADVANCED COMPOSITE PLOT GENERATION ---
    df = pd.DataFrame(results)
    mean_df = df.groupby(['Number of States', 'Architecture'])['Execution Time (ms)'].mean().reset_index()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(15, 10))

    palette_dict = {'Pure OOP (Sets)': '#e74c3c', 'Dual Flow (Numba)': '#2ecc71', 'Dual Flow (JAX)': '#3498db'}

    # 1. Background Boxplot (Transparent, NO FLIERS)
    sns.boxplot(
        data=df, x='Number of States', y='Execution Time (ms)', hue='Architecture',
        palette=palette_dict, linewidth=1.5, ax=ax,
        boxprops={'alpha': 0.25}, showfliers=False
    )

    # 2. Scatter Overlay (Perfectly colors all data points, including outliers)
    sns.stripplot(
        data=df, x='Number of States', y='Execution Time (ms)', hue='Architecture',
        palette=palette_dict, dodge=True, alpha=0.6, jitter=True, size=5, ax=ax
    )

    # 3. Foreground Line Plot (The Averages)
    sns.pointplot(
        data=mean_df, x='Number of States', y='Execution Time (ms)', hue='Architecture',
        palette={'Pure OOP (Sets)': '#c0392b', 'Dual Flow (Numba)': '#27ae60', 'Dual Flow (JAX)': '#2980b9'},
        markers=['o', 's', '^'], linestyles=['-', '-', '-'],
        errorbar=None, ax=ax, legend=False
    )

    # Add Text Annotations (Staggered to prevent overlap)
    x_categories = sorted(df['Number of States'].unique())
    for i, states in enumerate(x_categories):
        for arch in ['Pure OOP (Sets)', 'Dual Flow (Numba)', 'Dual Flow (JAX)']:
            val = mean_df[(mean_df['Number of States'] == states) & (mean_df['Architecture'] == arch)][
                'Execution Time (ms)'].values[0]

            if 'OOP' in arch:
                y_offset = 20
                color = '#c0392b'
            elif 'Numba' in arch:
                y_offset = -20
                color = '#27ae60'
            else:
                y_offset = 0
                color = '#2980b9'

            if i > 5 or 'JAX' in arch:
                ax.annotate(
                    f"{val:.1f}ms",
                    xy=(i, val),
                    xytext=(15 if 'JAX' in arch else 0, y_offset),
                    textcoords="offset points",
                    ha='left' if 'JAX' in arch else 'center', va='center',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none")
                )

    ax.set_yscale("log")
    ax.set_title("Dual Flow Architecture Scaling: Pure OOP vs. CPU (Numba) vs. GPU (JAX)", fontsize=18,
                 fontweight='bold', pad=20)
    ax.set_xlabel("Total Number of States Processed", fontsize=14, labelpad=10)
    ax.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=14, labelpad=10)

    # Clean up the legend (Stripplot and Boxplot add duplicates)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title="Architecture", loc="upper left")

    ax.grid(True, which="major", ls="-", alpha=0.6)
    ax.grid(True, which="minor", ls="--", alpha=0.3)

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "dual_flow_scaling_composite.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Dual Flow Composite Plot saved successfully to:\n{save_path}")



if __name__ == "__main__":
    generate_scaling_plot(runs_per_size=15)