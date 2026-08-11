"""Real IMU loader — SisFall (default) or UCI-HAR, for pretraining enc_imu.

SisFall  (https://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
  - .txt files, filename encodes activity: D## = ADL, F## = fall.
  - 9 numeric columns: ADXL345 acc(3), ITG3200 gyro(3), MMA8451Q acc(3) @ 200 Hz.
  - We use columns [0:3] accel + [3:6] gyro -> 6 channels (ax..gz).

Returns (X, y, n_classes):
  X : (N, t_imu, 6) float32
  y : (N,)          int   (activity id from filename prefix)
"""
from __future__ import annotations

import glob
import os
import numpy as np

from ..config import CFG, DATA_ROOT
from .windowing import segment_recording

SISFALL_HZ = 200.0


def _label_from_name(fname: str):
    base = os.path.basename(fname)
    code = base.split("_")[0]                 # e.g. 'D05' or 'F01'
    return code                                # keep raw code; mapped below


def load_imu_windows(cfg=CFG, source: str | None = None):
    root = DATA_ROOT / (source or cfg.imu_dir)
    files = sorted(glob.glob(str(root / "**" / "*.txt"), recursive=True)) + \
            sorted(glob.glob(str(root / "**" / "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No IMU files under {root}. Download SisFall or UCI-HAR and unzip "
            f"there — see docs/DATASETS.md.")

    X, codes = [], []
    for f in files:
        try:
            # SisFall rows contain nine comma-separated values and end with a
            # semicolon.  The first six are the accel+gyro channels used by
            # this project; selecting them avoids parsing the terminator in
            # the ninth field while remaining compatible with six-column CSVs.
            arr = np.loadtxt(f, delimiter=",", usecols=range(6), ndmin=2)
        except Exception:
            arr = np.genfromtxt(f, delimiter=",", usecols=range(6),
                                invalid_raise=False)
        if arr.ndim != 2 or arr.shape[1] < 6:
            continue
        sig = arr[:, :6].astype(np.float32)               # ax..gz
        for w in segment_recording(sig, SISFALL_HZ, cfg.t_imu, cfg.window_seconds):
            X.append(w); codes.append(_label_from_name(f))

    if not X:
        raise RuntimeError(f"Found files under {root} but parsed 0 windows — check format.")

    # map string codes -> contiguous ids
    uniq = sorted(set(codes))
    code2id = {c: i for i, c in enumerate(uniq)}
    y = np.array([code2id[c] for c in codes], dtype=np.int64)
    X = np.stack(X).astype(np.float32)
    return X, y, len(uniq)
