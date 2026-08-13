"""Paired multimodal loader — UP-Fall (camera + wearable IMU), for training the
CROSS-MODAL ATTENTION (the fusion stage needs sensors aligned in time).

UP-Fall (http://sites.google.com/up.edu.mx/har-up/, code: github.com/jpnm561/HAR-UP)
has camera + wearable IMU + ambient sensors for 17 subjects doing 11 activities
(incl. falls). It has NO mmWave radar, so radar is marked invalid per window
(radar_valid=False) — the model's masking handles that natively. You add the
real radar modality later from your own capture (or RadHAR-pretrained encoder).

Expected normalized layout (preprocess UP-Fall into this):
  data/raw/up_fall/<subject>/<activity>/<trial>/imu.csv    cols: ax,ay,az,gx,gy,gz (>=50 Hz)
  data/raw/up_fall/<subject>/<activity>/<trial>/video.mp4  (one camera)

Activity folder name is mapped to the FusionSense 5-class set via ACTIVITY_MAP.

Returns list[FusionWindow] (label set, radar zeroed+invalid).
"""
from __future__ import annotations

import glob
import json
import os
import numpy as np

from ..config import CFG, DATA_ROOT
from ..contract import FusionWindow, LABEL2ID
from .windowing import segment_recording
from .vision_extractor import video_to_pose_sequence, pose_sequence_to_windows

# map UP-Fall activity folder -> our 5 classes (adjust to the folder names you use)
ACTIVITY_MAP = {
    "walking": "walking", "standing": "standing", "sitting": "sitting",
    "lying": "lying", "picking": "sitting",
    "fall_forward": "falling", "fall_backward": "falling",
    "fall_sitting": "falling", "falling": "falling",
}


def _imu_windows(csv_path, cfg):
    # Accept both public-dataset files with six channels and collector files
    # with a leading ESP32 timestamp: t_ms, ax, ay, az, gx, gy, gz.
    first_line = open(csv_path, encoding="utf-8").readline().lower()
    if "ax" in first_line and "gx" in first_line:
        named = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)
        names = {name.lower(): name for name in (named.dtype.names or ())}
        wanted = [names.get(name) for name in ("ax", "ay", "az", "gx", "gy", "gz")]
        arr = np.column_stack([named[name] for name in wanted]) if all(wanted) else np.empty((0, 0))
    else:
        arr = np.genfromtxt(csv_path, delimiter=",", invalid_raise=False)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return []
    return segment_recording(arr[:, :6].astype(np.float32), cfg.imu_hz,
                             cfg.t_imu, cfg.window_seconds)


def load_paired_windows(cfg=CFG, source: str | None = None, use_vision=True):
    root = DATA_ROOT / (source or cfg.paired_dir)
    trials = sorted(glob.glob(str(root / "*" / "*" / "*")))
    if not trials:
        raise FileNotFoundError(
            f"No UP-Fall trials under {root}. See docs/DATASETS.md for download + "
            f"the expected <subject>/<activity>/<trial>/ layout.")

    windows = []
    for trial in trials:
        trial_path = os.path.normpath(trial)
        recording = os.path.basename(trial_path)
        activity = os.path.basename(os.path.dirname(trial))
        subject = os.path.basename(os.path.dirname(os.path.dirname(trial_path)))
        cls = ACTIVITY_MAP.get(activity.lower())
        if cls is None:
            continue
        label = LABEL2ID[cls]
        imu_csv = os.path.join(trial, "imu.csv")
        if not os.path.exists(imu_csv):
            continue
        imu_ws = _imu_windows(imu_csv, cfg)

        vis_ws = None
        video = os.path.join(trial, "video.mp4")
        if use_vision and os.path.exists(video):
            try:
                seq = video_to_pose_sequence(video)
                fps = 30.0
                metadata_path = os.path.join(trial, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, encoding="utf-8") as handle:
                        fps = float(json.load(handle).get("video_fps", fps))
                vis_ws = pose_sequence_to_windows(seq, cfg, src_fps=fps)
            except ImportError:
                vis_ws = None

        n = len(imu_ws)
        for i in range(n):
            vis = vis_ws[i] if (vis_ws and i < len(vis_ws)) else np.zeros((cfg.t_vis, cfg.vision_dv), np.float32)
            vision_valid = vis_ws is not None and i < len(vis_ws)
            windows.append(FusionWindow(
                t_start=0.0,
                imu=imu_ws[i],
                radar=np.zeros((cfg.t_radar, cfg.radar_k), np.float32),
                vision=vis,
                imu_valid=True, radar_valid=False, vision_valid=vision_valid,
                radar_energy=0.0, image_quality=1.0 if vision_valid else 0.0,
                imu_health=1.0, label=label,
                subject_id=subject,
                recording_id=f"{subject}/{activity}/{recording}",
            ))
    if not windows:
        raise RuntimeError(f"Parsed 0 paired windows from {root} — check layout / ACTIVITY_MAP.")
    return windows
