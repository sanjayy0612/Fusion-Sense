"""Train Step 7 IMU-only, vision-only, and masked IMU+vision models."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import CFG  # noqa: E402
from fusionsense.contract import ACTIVITIES  # noqa: E402
from fusionsense.data.cmhad_loader import (  # noqa: E402
    DEFAULT_CACHE,
    fit_normalization,
    load_cmhad_windows,
    normalize_windows,
)
from fusionsense.data.dataset import FusionDataset  # noqa: E402
from fusionsense.data.splitting import split_paired_windows  # noqa: E402
from fusionsense.device import get_device  # noqa: E402
from fusionsense.inference import classification_metrics  # noqa: E402
from fusionsense.models.encoders import EncoderClassifier  # noqa: E402
from fusionsense.models.fusion import FusionSense  # noqa: E402
from fusionsense.train.loop import train  # noqa: E402
from fusionsense.train.metrics import robustness_report  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def paired_bimodal_windows(windows):
    """Return labeled windows containing valid IMU+vision and no radar."""
    selected = [
        window
        for window in windows
        if window.imu_valid and window.vision_valid and not window.radar_valid
    ]
    if not selected:
        raise ValueError("No valid paired IMU+camera windows are available")
    labels = {window.label for window in selected}
    expected = set(range(CFG.n_classes))
    if labels != expected:
        raise ValueError(
            "Paired bimodal windows do not contain all classes; "
            f"found {sorted(labels)}"
        )
    return selected


def _baseline_arrays(windows, modality: str):
    return (
        np.stack([getattr(window, modality) for window in windows]).astype(np.float32),
        np.array([window.label for window in windows], dtype=np.int64),
    )


def _predict_baseline(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for values, labels in loader:
            guesses = model(values.to(device)).argmax(1).cpu().numpy()
            truth.extend(labels.numpy().tolist())
            predicted.extend(guesses.tolist())
    return truth, predicted


def train_baseline(
    train_windows,
    validation_windows,
    *,
    modality: str,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    output_path: Path,
):
    x_train, y_train = _baseline_arrays(train_windows, modality)
    x_validation, y_validation = _baseline_arrays(validation_windows, modality)
    in_channels = x_train.shape[2]
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(x_validation), torch.from_numpy(y_validation)
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    model = EncoderClassifier(in_channels, CFG.d_model, CFG.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(torch.from_numpy(y_train), minlength=CFG.n_classes).float()
    weights = counts.sum() / (CFG.n_classes * counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    best_state = None
    best_metrics = None
    best_epoch = 0

    print(f"\n=== {modality.upper()}-ONLY BASELINE ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            loss = criterion(model(values), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)

        truth, predicted = _predict_baseline(model, validation_loader, device)
        metrics = classification_metrics(truth, predicted)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_state = deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
        print(
            f"epoch {epoch:2d} | loss {total_loss / len(train_dataset):.3f} "
            f"| val acc {metrics['accuracy']:.3f} "
            f"| macroF1 {metrics['macro_f1']:.3f} "
            f"| fall-recall {metrics['fall_recall']:.3f}"
        )

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"{modality} baseline produced no validation result")
    model.load_state_dict(best_state)
    torch.save(
        {
            "schema_version": 1,
            "model_type": "single_modality_baseline",
            "modality": modality,
            "input_channels": in_channels,
            "d_model": CFG.d_model,
            "activities": ACTIVITIES,
            "state_dict": best_state,
        },
        output_path,
    )
    print(
        f"saved {output_path} (best epoch {best_epoch}, "
        f"macroF1 {best_metrics['macro_f1']:.3f})"
    )
    return model, {"best_epoch": best_epoch, **best_metrics}


def _predict_fusion(model, windows, device):
    loader = DataLoader(
        FusionDataset(windows, train=False, dropout_p=0.0), batch_size=128
    )
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for imu, radar, vision, valid, health, labels in loader:
            logits = model(
                imu.to(device),
                radar.to(device),
                vision.to(device),
                valid.to(device),
                health.to(device),
            )
            truth.extend(labels.numpy().tolist())
            predicted.extend(logits.argmax(1).cpu().numpy().tolist())
    return classification_metrics(truth, predicted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/cmhad_step7_bimodal")
    )
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--fusion-epochs", type=int, default=20)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fusion-dropout-p", type=float, default=CFG.modality_dropout_p
    )
    parser.add_argument(
        "--fine-tune-encoders",
        action="store_true",
        help="Fine-tune the pretrained IMU and camera encoders during fusion",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-subjects", type=int, default=4)
    args = parser.parse_args()
    if args.baseline_epochs <= 0 or args.fusion_epochs <= 0:
        raise SystemExit("epoch counts must be positive")
    if args.fusion_lr <= 0:
        raise SystemExit("fusion learning rate must be positive")
    if not 0.0 <= args.fusion_dropout_p < 1.0:
        raise SystemExit("fusion dropout probability must be in [0, 1)")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device()

    windows = load_cmhad_windows(args.cache)
    subjects = sorted({window.subject_id for window in windows})
    if len(subjects) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} subjects, found {len(subjects)}"
        )
    train_windows, validation_windows, split = split_paired_windows(
        windows, seed=args.seed
    )
    stats = fit_normalization(train_windows)
    train_windows = normalize_windows(train_windows, stats)
    validation_windows = normalize_windows(validation_windows, stats)
    paired_train = paired_bimodal_windows(train_windows)
    paired_validation = paired_bimodal_windows(validation_windows)

    args.output_dir.mkdir(parents=True)
    np.savez(args.output_dir / "normalization.npz", **stats)
    labels_payload = {
        "activities": ACTIVITIES,
        "subjects": subjects,
        "window_count": len(windows),
        "paired_train_windows": len(paired_train),
        "paired_validation_windows": len(paired_validation),
        "split": split,
        "seed": args.seed,
        "baseline_epochs": args.baseline_epochs,
        "fusion_epochs": args.fusion_epochs,
        "fusion_learning_rate": args.fusion_lr,
        "fusion_dropout_probability": args.fusion_dropout_p,
        "fusion_encoders_fine_tuned": args.fine_tune_encoders,
        "modalities": ["imu", "camera_pose"],
        "radar_active": False,
    }
    (args.output_dir / "labels.json").write_text(
        json.dumps(labels_payload, indent=2) + "\n", encoding="utf-8"
    )

    log_path = args.output_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(
        Tee(sys.stdout, log)
    ):
        print("device:", device)
        print(
            f"windows: {len(windows)}; paired train={len(paired_train)}, "
            f"paired validation={len(paired_validation)}; {split}"
        )
        print("train class counts:", dict(sorted(Counter(
            window.label for window in paired_train
        ).items())))

        imu_model, imu_metrics = train_baseline(
            paired_train,
            paired_validation,
            modality="imu",
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_path=args.output_dir / "baseline_imu.pt",
        )
        vision_model, vision_metrics = train_baseline(
            paired_train,
            paired_validation,
            modality="vision",
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            output_path=args.output_dir / "baseline_vision.pt",
        )

        print("\n=== MASKED IMU + CAMERA FUSION ===")
        fusion = FusionSense(CFG)
        fusion.enc_imu.load_state_dict(imu_model.encoder.state_dict())
        fusion.enc_vis.load_state_dict(vision_model.encoder.state_dict())
        fusion.freeze_encoders(
            imu=not args.fine_tune_encoders,
            radar=True,
            vision=not args.fine_tune_encoders,
        )
        fusion = train(
            paired_train,
            paired_validation,
            epochs=args.fusion_epochs,
            batch_size=args.batch_size,
            device=device,
            model=fusion,
            lr=args.fusion_lr,
            dropout_p=args.fusion_dropout_p,
        )
        fusion_metrics = _predict_fusion(fusion, paired_validation, device)
        robustness = robustness_report(fusion, paired_validation, device)
        bimodal_robustness = {
            "imu_and_camera": robustness["no radar"],
            "imu_only": robustness["imu only"],
            "vision_only": robustness["vision only"],
        }
        torch.save(fusion.state_dict(), args.output_dir / "fusionsense_cmhad.pt")
        print("saved", args.output_dir / "fusionsense_cmhad.pt")
        print("masked fusion validation:", fusion_metrics)
        print("bimodal robustness:", bimodal_robustness)

        metrics = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result": "PASS",
            "dataset": labels_payload,
            "models": {
                "imu_only": imu_metrics,
                "vision_only": vision_metrics,
                "masked_imu_camera_fusion": fusion_metrics,
            },
            "modality_dropout": {
                "training_probability": args.fusion_dropout_p,
                "never_drop_all_valid_modalities": True,
                "radar_permanently_masked": True,
                "validation": bimodal_robustness,
            },
            "artifacts": {
                "imu_baseline": "baseline_imu.pt",
                "vision_baseline": "baseline_vision.pt",
                "masked_fusion": "fusionsense_cmhad.pt",
                "normalization": "normalization.npz",
                "labels": "labels.json",
                "training_log": "training.log",
            },
        }
        (args.output_dir / "step7_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics["models"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
