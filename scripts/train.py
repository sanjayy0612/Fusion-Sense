"""End-to-end training run on simulated data. Requires torch.

Run:  python scripts/train.py
On your RTX 4060 this trains in a couple of minutes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fusionsense.data.simulator import make_dataset
from fusionsense.train.loop import train, evaluate
from fusionsense.train.metrics import robustness_report, print_robustness
from fusionsense.data.dataset import FusionDataset
from torch.utils.data import DataLoader


def main():
    print("generating simulated dataset...")
    train_w = make_dataset(n_per_class=800, seed=0, degrade=True)
    val_w = make_dataset(n_per_class=200, seed=1, degrade=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = train(train_w, val_w, epochs=15, device=device)

    # headline experiment: accuracy when each modality is dropped at inference
    print("\n=== ROBUSTNESS UNDER MODALITY DROPOUT ===")
    rows = robustness_report(model, val_w, device)
    print_robustness(rows)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fusionsense.pt")
    print("\nsaved checkpoints/fusionsense.pt")


if __name__ == "__main__":
    main()
