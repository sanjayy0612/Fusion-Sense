"""Visualize a 'walking' vs a 'falling' FusionWindow so you can SEE the data
the model learns from. Saves a PNG — no torch required.

Run:  python scripts/viz_windows.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.data.simulator import sample_window


def plot_window(ax_row, w, title):
    ax_row[0].plot(w.imu[:, :3]); ax_row[0].set_title(f"{title}: IMU accel")
    ax_row[1].plot(w.radar[:, 0], label="range"); ax_row[1].plot(w.radar[:, 1], label="velocity")
    ax_row[1].set_title(f"{title}: radar"); ax_row[1].legend(fontsize=7)
    ax_row[2].plot(w.vision[:, :3]); ax_row[2].set_title(f"{title}: vision code")


def main():
    rng = np.random.default_rng(1)
    walk = sample_window("walking", rng, degrade=False)
    fall = sample_window("falling", rng, degrade=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    plot_window(axes[0], walk, "walking")
    plot_window(axes[1], fall, "falling")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "window_preview.png")
    fig.savefig(out, dpi=110)
    print("saved", os.path.abspath(out))


if __name__ == "__main__":
    main()
