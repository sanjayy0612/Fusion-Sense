"""Test laptop-computable temporal features on the fixed Step 7 subject split."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import CFG  # noqa: E402
from fusionsense.data.cmhad_loader import load_cmhad_windows  # noqa: E402
from fusionsense.data.splitting import split_paired_windows  # noqa: E402
from fusionsense.device import get_device  # noqa: E402
from fusionsense.inference import classification_metrics  # noqa: E402
from fusionsense.models.encoders import EncoderClassifier  # noqa: E402


def imu_features(values: np.ndarray) -> np.ndarray:
    """Raw axes + orientation-robust magnitudes + first differences."""
    accel_magnitude = np.linalg.norm(values[..., :3], axis=-1, keepdims=True)
    gyro_magnitude = np.linalg.norm(values[..., 3:6], axis=-1, keepdims=True)
    difference = np.diff(values, axis=1, prepend=values[:, :1])
    return np.concatenate(
        [values, accel_magnitude, gyro_magnitude, difference], axis=-1
    ).astype(np.float32)


def vision_features(values: np.ndarray) -> np.ndarray:
    """Pose coordinates plus explicit per-landmark temporal velocity."""
    difference = np.diff(values, axis=1, prepend=values[:, :1])
    return np.concatenate([values, difference], axis=-1).astype(np.float32)


def normalize(train: np.ndarray, validation: np.ndarray):
    flattened = train.reshape(-1, train.shape[-1])
    mean = flattened.mean(axis=0).astype(np.float32)
    std = np.maximum(flattened.std(axis=0), 1e-6).astype(np.float32)
    return (train - mean) / std, (validation - mean) / std, mean, std


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    truth, predicted = [], []
    for values, labels in loader:
        guesses = model(values.to(device)).argmax(1).cpu().tolist()
        truth.extend(labels.tolist())
        predicted.extend(guesses)
    return classification_metrics(truth, predicted)


def train_model(x_train, y_train, x_validation, y_validation, epochs, device):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
        batch_size=128,
    )
    model = EncoderClassifier(x_train.shape[-1], CFG.d_model, CFG.n_classes).to(device)
    counts = torch.bincount(torch.from_numpy(y_train), minlength=CFG.n_classes).float()
    weights = counts.sum() / (CFG.n_classes * counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_metrics, best_epoch = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for values, labels in train_loader:
            loss = criterion(model(values.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = predict(model, validation_loader, device)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_state = deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
        print(
            f"epoch={epoch:02d} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"fall_recall={metrics['fall_recall']:.4f}"
        )
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, **best_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    windows = load_cmhad_windows()
    train_windows, validation_windows, split = split_paired_windows(windows, seed=42)
    y_train = np.asarray([window.label for window in train_windows], dtype=np.int64)
    y_validation = np.asarray(
        [window.label for window in validation_windows], dtype=np.int64
    )
    raw = {
        "imu_engineered": (
            imu_features(np.stack([window.imu for window in train_windows])),
            imu_features(np.stack([window.imu for window in validation_windows])),
        ),
        "vision_with_velocity": (
            vision_features(np.stack([window.vision for window in train_windows])),
            vision_features(np.stack([window.vision for window in validation_windows])),
        ),
    }
    args.output_dir.mkdir(parents=True)
    device = get_device()
    report = {"split": split, "models": {}}
    for name, (x_train, x_validation) in raw.items():
        print(f"\n=== {name} ({x_train.shape[-1]} channels) ===")
        x_train, x_validation, mean, std = normalize(x_train, x_validation)
        model, metrics = train_model(
            x_train.astype(np.float32),
            y_train,
            x_validation.astype(np.float32),
            y_validation,
            args.epochs,
            device,
        )
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
        np.savez(args.output_dir / f"{name}_normalization.npz", mean=mean, std=std)
        report["models"][name] = metrics
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
