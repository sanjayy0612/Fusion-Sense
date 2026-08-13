"""Train C-MHAD IMU, camera-pose, then fusion stages with subject isolation."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fusionsense.config import CFG
from fusionsense.contract import ACTIVITIES
from fusionsense.data.cmhad_loader import (
    DEFAULT_CACHE,
    fit_normalization,
    load_cmhad_windows,
    normalize_windows,
)
from fusionsense.data.splitting import split_paired_windows
from fusionsense.device import get_device
from fusionsense.models.fusion import FusionSense
from fusionsense.train.loop import train
from fusionsense.train.metrics import print_robustness, robustness_report
from fusionsense.train.pretrain import pretrain_encoder


def arrays(windows, modality):
    if modality == "vision":
        windows = [window for window in windows if window.vision_valid]
        if not windows:
            raise ValueError("No MediaPipe-valid windows available for camera pretraining")
    return (
        np.stack([getattr(window, modality) for window in windows]).astype(np.float32),
        np.array([window.label for window in windows], dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default="checkpoints/cmhad")
    parser.add_argument("--stage", choices=("all", "imu", "vision", "fusion"), default="all")
    parser.add_argument("--encoder-epochs", type=int, default=20)
    parser.add_argument("--fusion-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="fine-tune pretrained encoders during fusion instead of freezing them",
    )
    args = parser.parse_args()

    device = get_device()
    print("device:", device)
    windows = load_cmhad_windows(args.cache)
    train_w, val_w, description = split_paired_windows(
        windows, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"windows: {len(windows)} (train={len(train_w)}, val={len(val_w)}; {description})")

    stats = fit_normalization(train_w)
    train_w = normalize_windows(train_w, stats)
    val_w = normalize_windows(val_w, stats)
    os.makedirs(args.output_dir, exist_ok=True)
    imu_checkpoint = os.path.join(args.output_dir, "enc_imu.pt")
    vision_checkpoint = os.path.join(args.output_dir, "enc_vis.pt")
    fusion_checkpoint = os.path.join(args.output_dir, "fusionsense_cmhad.pt")
    np.savez(os.path.join(args.output_dir, "normalization.npz"), **stats)
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump({"activities": ACTIVITIES, "split": description}, handle, indent=2)

    if args.stage in ("all", "imu"):
        x_train, y_train = arrays(train_w, "imu")
        x_val, y_val = arrays(val_w, "imu")
        print("\n=== STAGE 1A: IMU ENCODER (FROM SCRATCH) ===")
        pretrain_encoder(
            x_train, y_train, CFG.n_classes, CFG.imu_ch,
            imu_checkpoint, epochs=args.encoder_epochs,
            batch_size=args.batch_size, device=device,
            X_val=x_val, y_val=y_val, seed=args.seed,
        )

    if args.stage in ("all", "vision"):
        x_train, y_train = arrays(train_w, "vision")
        x_val, y_val = arrays(val_w, "vision")
        print("\n=== STAGE 1B: CAMERA-POSE ENCODER (FROM SCRATCH) ===")
        pretrain_encoder(
            x_train, y_train, CFG.n_classes, CFG.vision_dv,
            vision_checkpoint, epochs=args.encoder_epochs,
            batch_size=args.batch_size, device=device,
            X_val=x_val, y_val=y_val, seed=args.seed,
        )

    if args.stage in ("all", "fusion"):
        required = [imu_checkpoint, vision_checkpoint]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing}; run --stage imu and --stage vision first, or --stage all."
            )
        print("\n=== STAGE 2: CROSS-MODAL FUSION ===")
        model = FusionSense(CFG).load_pretrained_encoders(
            imu=required[0], vision=required[1]
        )
        if not args.fine_tune:
            model.freeze_encoders(imu=True, radar=False, vision=True)
        model = train(
            train_w, val_w, epochs=args.fusion_epochs,
            batch_size=args.batch_size, device=device, model=model,
        )
        print_robustness(robustness_report(model, val_w, device))
        torch.save(model.state_dict(), fusion_checkpoint)
        print(f"saved {fusion_checkpoint}")


if __name__ == "__main__":
    main()
