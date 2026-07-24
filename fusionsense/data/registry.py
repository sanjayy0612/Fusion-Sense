"""Unified data access. One place to ask for pretraining data (per modality) or
paired data, with an optional simulator fallback so smoke tests always run.

  get_pretrain_data("imu")            -> (X, y, n_classes) from real dataset
  get_paired_windows()                -> list[FusionWindow] from UP-Fall
  ...allow_sim_fallback=True          -> use the simulator if real data absent
"""
from __future__ import annotations

import warnings
import numpy as np

from ..config import CFG
from .simulator import make_dataset


def get_pretrain_data(modality: str, cfg=CFG, allow_sim_fallback=False):
    """Return (X, y, n_classes) of single-modality windows for `modality`
    in {'imu','radar','vision'}."""
    try:
        if modality == "imu":
            from .imu_loader import load_imu_windows
            return load_imu_windows(cfg)
        if modality == "radar":
            from .radar_loader import load_radar_windows
            return load_radar_windows(cfg)
        if modality == "vision":
            raise FileNotFoundError(
                "Vision pretraining runs from video via vision_extractor; provide "
                "videos + labels or use the paired UP-Fall data. See docs/DATASETS.md.")
        raise ValueError(f"unknown modality {modality}")
    except (FileNotFoundError, ImportError) as e:
        if not allow_sim_fallback:
            raise
        warnings.warn(f"[registry] real {modality} data unavailable ({e}); "
                      f"using SIMULATOR (smoke test only, not real results).")
        return _sim_single(modality, cfg)


def get_paired_windows(cfg=CFG, allow_sim_fallback=False):
    try:
        from .paired_loader import load_paired_windows
        return load_paired_windows(cfg)
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        if not allow_sim_fallback:
            raise
        warnings.warn(f"[registry] real paired data unavailable ({e}); "
                      f"using SIMULATOR (smoke test only, not real results).")
        return make_dataset(n_per_class=400, seed=0, degrade=True)


def _sim_single(modality, cfg):
    """Carve single-modality (X, y) out of simulated FusionWindows."""
    ds = make_dataset(n_per_class=600, seed=0, degrade=False)
    pick = {"imu": lambda w: w.imu, "radar": lambda w: w.radar, "vision": lambda w: w.vision}[modality]
    X = np.stack([pick(w) for w in ds]).astype(np.float32)
    y = np.array([w.label for w in ds], dtype=np.int64)
    return X, y, cfg.n_classes
