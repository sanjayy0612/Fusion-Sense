"""Generate a long animated simulation demo with visible pauses between values."""
from __future__ import annotations

import argparse
import os
import sys
from itertools import cycle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.simulator import MockRuntime, sample_window
from fusionsense.contract import ACTIVITIES


def build_frame_sequence(n_windows: int = 15, delay_between_values: int = 4, seed: int = 7) -> list:
    """Create a sequence of windows with repeated pause frames between values."""
    print(f"Generating {n_windows} activity windows with {delay_between_values} pause frames between each...")
    runtime = MockRuntime(seed=seed, degrade=False)
    activities = cycle(ACTIVITIES)
    frames = []
    for index in range(n_windows):
        activity = next(activities)
        rng = np.random.default_rng(seed + index)
        window = sample_window(activity, rng, degrade=False)
        frames.append(window)
        for _ in range(delay_between_values):
            frames.append(window)
    return frames


def render_video(output_path: str, n_windows: int = 15, delay_between_values: int = 4, seed: int = 7) -> str:
    """Render an animated GIF demo from the simulation windows."""
    print(f"Rendering animation to {output_path}")
    frames = build_frame_sequence(
        n_windows=n_windows,
        delay_between_values=delay_between_values,
        seed=seed,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    fig.suptitle("Mock Runtime Simulation Demo", fontsize=12)

    def draw_frame(frame_index):
        window = frames[frame_index]
        axes[0].clear()
        axes[0].plot(window.imu[:, :3])
        axes[0].set_title("IMU")
        axes[0].set_ylabel("signal")
        axes[0].set_ylim(-12, 12)

        axes[1].clear()
        axes[1].plot(window.radar[:, 0], label="range")
        axes[1].plot(window.radar[:, 1], label="velocity")
        axes[1].set_title("Radar")
        axes[1].legend(loc="upper right", fontsize=7)

        axes[2].clear()
        axes[2].plot(window.vision[:, :3])
        axes[2].set_title("Vision")
        axes[2].set_ylim(-1.5, 3.5)

        fig.text(0.5, 0.02, f"Window {frame_index + 1}/{len(frames)} | delay={delay_between_values} frames", ha="center")
        return axes

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=500, repeat=False)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    ani.save(output_path, writer=PillowWriter(fps=2))
    plt.close(fig)

    estimated_seconds = len(frames) * 0.5
    print(f"Finished rendering {len(frames)} frames")
    print(f"Estimated duration: {estimated_seconds:.1f} seconds")
    print("Saved to", os.path.abspath(output_path))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a simulation animation with delay frames")
    parser.add_argument("--output", default=os.path.join("figures", "fusion_sense_runtime_demo.gif"), help="Output GIF path")
    parser.add_argument("--windows", type=int, default=15, help="Number of activity windows to render")
    parser.add_argument("--delay", type=int, default=4, help="Number of repeated pause frames between activity values")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    render_video(args.output, n_windows=args.windows, delay_between_values=args.delay, seed=args.seed)
