import os
import time
import numpy as np
from pathlib import Path
from numba import njit
from dataclasses import dataclass
from typing import Protocol, Type, Callable, Any, TypeVar, Set
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from particle_grid_simulator.src.dynamic_system.domain.data.dynamic_systems import DynamicSystemData
from particle_grid_simulator.src.dynamic_system.domain.utility.dynamic_systems import DynamicSystemRunner
from particle_grid_simulator.src.operator.domain.data.operator import GenericOperatorData
from particle_grid_simulator.src.operator.domain.utility.operator import GenericOperatorUtility
from particle_grid_simulator.src.state.domain import State

# ==========================================
# 3. PHYSICS KERNEL & SPECIFIC OPERATOR
# ==========================================
# Global configuration for physics
DT = 0.1
BOUND = 10.0


@njit(cache=True, fastmath=True)
def bouncing_box_kernel(states_batch: np.ndarray) -> np.ndarray:
    """
    C-Speed Physics Kernel.
    State is [px, py, vx, vy].
    """
    new_states = np.empty_like(states_batch)

    for i in range(states_batch.shape[0]):
        px, py, vx, vy = states_batch[i]

        # 1. Kinematics (x = x + v*dt)
        px += vx * DT
        py += vy * DT

        # 2. X-Axis Boundary Reflection
        if px > BOUND:
            px = BOUND - (px - BOUND)
            vx = -vx
        elif px < -BOUND:
            px = -BOUND + (-BOUND - px)
            vx = -vx

        # 3. Y-Axis Boundary Reflection
        if py > BOUND:
            py = BOUND - (py - BOUND)
            vy = -vy
        elif py < -BOUND:
            py = -BOUND + (-BOUND - py)
            vy = -vy

        new_states[i, 0] = px
        new_states[i, 1] = py
        new_states[i, 2] = vx
        new_states[i, 3] = vy

    return new_states





# ==========================================
# 4. CSV VISUALIZATION
# ==========================================
def animate_from_csv(csv_path: Path, n_particles: int, save_path: Path):
    print("   -> Loading CSV for Animation...")
    # Load flattened data and reshape back to (Ticks, N, 4)
    flat_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    history = flat_data.reshape(flat_data.shape[0], n_particles, 4)

    num_frames = len(history)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#111111')

    ax.set_xlim(-BOUND - 1, BOUND + 1)
    ax.set_ylim(-BOUND - 1, BOUND + 1)

    # Draw the Box Boundaries
    box_rect = plt.Rectangle((-BOUND, -BOUND), BOUND * 2, BOUND * 2, fill=False, color='white', linestyle='--',
                             linewidth=2)
    ax.add_patch(box_rect)
    ax.set_title(f"Classical Batch Dynamics: Bouncing Box\n{n_particles} Particles")

    colors = plt.cm.plasma(np.linspace(0, 1, n_particles))
    particle_scatters = [ax.scatter([], [], color=colors[p], s=80, edgecolors='white', zorder=5) for p in
                         range(n_particles)]
    particle_lines = [ax.plot([], [], color=colors[p], alpha=0.4, linewidth=1)[0] for p in range(n_particles)]

    def update(frame):
        # Slice the history to get trails
        trail_length = min(frame + 1, 20)  # Show last 20 ticks as a tail
        start_idx = max(0, frame - trail_length)

        for p in range(n_particles):
            # Pos = Indices 0 and 1
            particle_scatters[p].set_offsets(history[frame, p, :2])
            particle_lines[p].set_data(history[start_idx:frame + 1, p, 0], history[start_idx:frame + 1, p, 1])

        return particle_scatters + particle_lines

    anim = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)
    out_file = save_path / "bouncing_box.gif"
    print(f"🔄 Rendering GIF to {out_file}...")
    anim.save(out_file, writer='pillow', fps=30)
    print("✅ Done!")


import cv2
import imageio
import numpy as np
from pathlib import Path


def fast_render_from_csv(csv_path: Path, n_particles: int, save_path: Path, bound: float = 10.0):
    print("   -> Loading CSV for Fast Render...")
    flat_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    history = flat_data.reshape(flat_data.shape[0], n_particles, 4)

    num_frames = len(history)
    res = 800  # Resolution: 800x800 pixels

    # Pre-calculate distinct colors for particles (BGR format for OpenCV)
    np.random.seed(42)
    colors = np.random.randint(50, 255, (n_particles, 3)).tolist()

    def to_pixels(coords: np.ndarray) -> np.ndarray:
        """Transforms DOD domain coordinates [-10, 10] into screen pixels [0, 800]."""
        normalized = (coords + bound) / (2 * bound)
        # Invert Y axis because pixel (0,0) is top-left in images
        normalized[:, 1] = 1.0 - normalized[:, 1]
        return (normalized * res).astype(np.int32)

    print("   -> Drawing frames directly to C-buffers...")
    frames = []

    for frame_idx in range(num_frames):
        # 1. Pre-allocate the canvas (A contiguous block of memory!)
        # 800x800 array of 8-bit unsigned integers (RGB)
        img = np.zeros((res, res, 3), dtype=np.uint8)

        # Draw bounding box
        cv2.rectangle(img, (0, 0), (res - 1, res - 1), (255, 255, 255), thickness=2)

        # 2. Vectorized conversion of all particle coordinates for this frame
        pixel_coords = to_pixels(history[frame_idx, :, :2])

        # 3. Draw Particles and Trails
        for p in range(n_particles):
            cx, cy = pixel_coords[p]

            # Draw particle
            cv2.circle(img, (cx, cy), radius=5, color=colors[p], thickness=-1)

            # Draw trail (last 10 ticks)
            if frame_idx > 0:
                trail_len = min(frame_idx, 10)
                trail_coords = to_pixels(history[frame_idx - trail_len:frame_idx + 1, p, :2])
                for t in range(len(trail_coords) - 1):
                    pt1 = tuple(trail_coords[t])
                    pt2 = tuple(trail_coords[t + 1])
                    cv2.line(img, pt1, pt2, colors[p], thickness=1)

        # OpenCV uses BGR natively, GIF needs RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)

        if (frame_idx + 1) % 100 == 0:
            print(f"      Rendered {frame_idx + 1}/{num_frames} frames to memory...")

    out_file = save_path / "fast_bouncing_box.gif"
    print(f"🔄 Encoding high-speed GIF to {out_file}...")

    # Write directly to GIF.
    # 'loop=0' means infinite loop, 'duration' is milliseconds per frame.
    imageio.mimsave(out_file, frames, duration=30, loop=0)
    print("✅ Done! You may now burn Matplotlib.")

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
def run_classical_bouncing_box():
    # --- CONFIGURATION ---
    NUM_PARTICLES = 50
    ITERATIONS = 400
    HISTORY_WINDOW = 100  # Flush to disk every 100 ticks
    SAVE_PATH = Path(r"E:\Particle Field Simulation\particle_grid_simulator\test\dynamic_system\plots")

    print("1. Initializing Classical Dynamic System...")
    # Spawn particles with random positions (-9 to 9) and random velocities (-5 to 5)
    initial_positions = np.random.uniform(-9.0, 9.0, (NUM_PARTICLES, 2))
    initial_velocities = np.random.uniform(-5.0, 5.0, (NUM_PARTICLES, 2))

    # Combine into (N, 4) state batch
    s0_batch = np.hstack((initial_positions, initial_velocities)).astype(np.float64)

    # 2. Setup Data Blueprint
    operator_data = GenericOperatorData(bouncing_box_kernel,State)
    system_data = DynamicSystemData(
        _initial_states=s0_batch,
        _operator=operator_data,
        _history_window_size=HISTORY_WINDOW,
        _save_directory=SAVE_PATH
    )

    # 3. Initialize Runner (The Clock)
    runner = DynamicSystemRunner(system_data, GenericOperatorUtility)

    print(f"2. Running Hot Loop for {ITERATIONS} iterations...")
    t0 = time.perf_counter()

    # JIT Warmup (Invisible to the timeline)
    GenericOperatorUtility.evolve_batch(operator_data, s0_batch)

    for _ in range(ITERATIONS):
        runner.next()

    print(f"   -> Loop completed in {time.perf_counter() - t0:.4f}s")

    # 4. Trigger the CSV Compile
    runner.end(compile_csv=True)

    # 5. Animate from the generated telemetry
    csv_file = SAVE_PATH / "compiled_telemetry.csv"
    fast_render_from_csv(csv_file, NUM_PARTICLES, SAVE_PATH)


if __name__ == "__main__":
    run_classical_bouncing_box()