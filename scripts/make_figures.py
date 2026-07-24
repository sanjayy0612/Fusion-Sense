"""Generate figures for the review deck. The numpy-only figures run without
torch; the results figures (confusion, robustness, trust) are written by
scripts/train.py after training.

Run:  python scripts/make_figures.py
Outputs -> figures/*.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.contract import ACTIVITIES
from fusionsense.data.simulator import sample_window, make_dataset

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def fig_signatures():
    """Per-activity IMU / radar / vision signatures — shows class separability."""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(len(ACTIVITIES), 3, figsize=(12, 12))
    for r, act in enumerate(ACTIVITIES):
        w = sample_window(act, rng, degrade=False)
        axes[r, 0].plot(w.imu[:, :3]); axes[r, 0].set_ylabel(act, fontsize=11)
        axes[r, 1].plot(w.radar[:, 0], label="range")
        axes[r, 1].plot(w.radar[:, 1], label="velocity")
        axes[r, 2].plot(w.vision[:, :3])
        if r == 0:
            axes[r, 0].set_title("IMU accel (x,y,z)")
            axes[r, 1].set_title("Radar (range, velocity)"); axes[r, 1].legend(fontsize=7)
            axes[r, 2].set_title("Vision posture code")
    fig.suptitle("Per-activity multimodal signatures (simulated)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{FIG}/signatures.png", dpi=110); plt.close(fig)


def fig_health_distribution():
    """Histogram of the sensor-health scalars under degradation."""
    ds = make_dataset(n_per_class=400, seed=7, degrade=True)
    H = np.array([w.health_vector() for w in ds])  # (N,3) [imu,radar,vision]
    labels = ["IMU health", "Radar energy", "Image quality"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for i in range(3):
        axes[i].hist(H[:, i], bins=20, color="#3b7dd8")
        axes[i].set_title(labels[i]); axes[i].set_xlabel("reliability [0-1]")
    fig.suptitle("Simulated sensor-health signals (degradation injected)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/health_distribution.png", dpi=110); plt.close(fig)


def fig_dataset_balance():
    ds = make_dataset(n_per_class=200, seed=1)
    counts = np.bincount([w.label for w in ds], minlength=len(ACTIVITIES))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(ACTIVITIES, counts, color="#2e8b57")
    ax.set_title("Simulated dataset — class balance"); ax.set_ylabel("windows")
    fig.tight_layout(); fig.savefig(f"{FIG}/dataset_balance.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    fig_signatures()
    fig_health_distribution()
    fig_dataset_balance()
    print("wrote figures to", os.path.abspath(FIG))
