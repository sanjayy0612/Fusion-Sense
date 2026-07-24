"""Generic windowing: turn a long, possibly irregular sensor stream into
fixed-length windows matching the FusionWindow contract.

Used by every real-data loader so simulated and real data are windowed the
SAME way (bucket + resample to a fixed length).
"""
from __future__ import annotations

import numpy as np


def resample_to(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-resample a (T, C) array along time to (target_len, C).
    Works whether the input is longer or shorter than target_len."""
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    if T == target_len:
        return x
    if T == 0:
        return np.zeros((target_len, x.shape[1] if x.ndim > 1 else 1), np.float32)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, target_len)
    if x.ndim == 1:
        return np.interp(dst, src, x).astype(np.float32)
    return np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


def sliding_windows(x: np.ndarray, win_len: int, stride: int):
    """Yield (start_idx, window) of fixed length from a long (T, C) signal.
    Used when a recording is much longer than one window (e.g., continuous ADL)."""
    T = x.shape[0]
    if T < win_len:
        yield 0, resample_to(x, win_len)
        return
    for s in range(0, T - win_len + 1, stride):
        yield s, x[s:s + win_len]


def segment_recording(signal: np.ndarray, src_hz: float, target_len: int,
                      window_seconds: float, overlap: float = 0.5):
    """Split a continuous recording sampled at src_hz into fixed-length windows
    (resampled to target_len). Returns list of (T=target_len, C) arrays."""
    win_src = int(window_seconds * src_hz)
    stride = max(1, int(win_src * (1 - overlap)))
    out = []
    for _, w in sliding_windows(signal, win_src, stride):
        out.append(resample_to(w, target_len))
    return out
