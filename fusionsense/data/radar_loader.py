"""Real mmWave radar loader — RadHAR (https://github.com/nesl/RadHAR), for
pretraining enc_radar.

RadHAR ships point clouds voxelized into per-frame grids. Preprocessing (see
their repo) yields, per sample, a sequence of frames each summarized as a
feature vector. This loader expects preprocessed arrays saved as .npy:

  data/raw/radhar/<activity>/<sample>.npy   with shape (frames, F)

and reduces F -> cfg.radar_k features per frame (mean-pool of feature groups),
then resamples to cfg.t_radar frames.

If your preprocessing differs, adapt `_to_frame_features` below — that's the
only coupling point.

Returns (X, y, n_classes):  X (N, t_radar, radar_k), y (N,)
"""
from __future__ import annotations

import glob
import os
import numpy as np

from ..config import CFG, DATA_ROOT
from .windowing import resample_to


def _to_frame_features(arr: np.ndarray, k: int) -> np.ndarray:
    """(frames, F) -> (frames, k). Group-average F into k buckets (robust to any F)."""
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    F = arr.shape[1]
    if F == k:
        return arr
    idx = np.linspace(0, F, k + 1).astype(int)
    return np.stack([arr[:, idx[i]:max(idx[i] + 1, idx[i + 1])].mean(1) for i in range(k)], axis=1)


def load_radar_windows(cfg=CFG, source: str | None = None):
    root = DATA_ROOT / (source or cfg.radar_dir)
    files = sorted(glob.glob(str(root / "**" / "*.npy"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No radar .npy files under {root}. Download & preprocess RadHAR "
            f"(github.com/nesl/RadHAR) into {root}/<activity>/*.npy — see docs/DATASETS.md.")

    X, labels = [], []
    for f in files:
        activity = os.path.basename(os.path.dirname(f))
        arr = np.load(f, allow_pickle=True)
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 3:                         # (frames, H, W) voxel grid -> flatten
            arr = arr.reshape(arr.shape[0], -1)
        feats = _to_frame_features(arr, cfg.radar_k)
        X.append(resample_to(feats, cfg.t_radar)); labels.append(activity)

    uniq = sorted(set(labels)); lab2id = {a: i for i, a in enumerate(uniq)}
    y = np.array([lab2id[a] for a in labels], dtype=np.int64)
    return np.stack(X).astype(np.float32), y, len(uniq)
