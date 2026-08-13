"""Synthetic FusionWindow generator ("the fake lunchbox maker").

Lets you build and train the entire model with ZERO hardware. Each activity has
a physically-motivated signature across the three modalities, plus realistic
noise and optional sensor degradation (darkness, radar noise, IMU dropout).

This is deliberately simple and readable — it is NOT meant to be a realistic
physics sim. Its job is to produce learnable, separable, degradation-aware data
so the pipeline and the model can be validated before real sensors exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..config import CFG
from ..contract import FusionWindow, LABEL2ID


def _periodic(T, freq, amp, phase=0.0, rng=None):
    t = np.linspace(0, 1, T, endpoint=False)
    sig = amp * np.sin(2 * np.pi * freq * t + phase)
    if rng is not None:
        sig = sig + rng.normal(0, 0.05 * abs(amp) + 1e-3, size=T)
    return sig


def _imu_signature(activity, rng):
    """(T_imu, 6) accel(3)+gyro(3)."""
    T = CFG.t_imu
    x = np.zeros((T, 6), dtype=np.float32)
    g = 9.81
    start, end = activity.split("_to_")
    gravities = {
        "stand": np.array([0.0, 0.0, g]),
        "sit": np.array([0.0, 0.4 * g, 0.9 * g]),
        "lie": np.array([g, 0.0, 0.0]),
        "fall": np.array([g, 0.0, 0.0]),
    }
    alpha = np.linspace(0.0, 1.0, T)[:, None]
    x[:, :3] = (
        (1.0 - alpha) * gravities[start]
        + alpha * gravities[end]
        + rng.normal(0, 0.18, size=(T, 3))
    )
    direction = {
        "stand_to_sit": 1.0, "sit_to_stand": -1.0,
        "sit_to_lie": 2.0, "lie_to_sit": -2.0,
        "lie_to_stand": -3.0, "stand_to_lie": 3.0,
        "stand_to_fall": 5.0,
    }[activity]
    x[:, 3:] = rng.normal(0, 1.5, size=(T, 3))
    x[:, 3] += _periodic(T, freq=0.5, amp=20.0 * direction, rng=rng)
    if activity == "stand_to_fall":
        # free-fall dip then sharp impact spike, then still
        drop = int(T * 0.4)
        impact = int(T * 0.55)
        x[drop:impact, 2] *= 0.15                       # near free-fall
        x[impact:impact + 3, :3] += rng.normal(0, 25, size=(3, 3))  # impact
        x[impact + 5:, 2] = 0.3 * g + rng.normal(0, 0.2, size=T - impact - 5)  # lying after
        x[:, 3:] = rng.normal(0, 2.0, size=(T, 3))
        x[impact:impact + 3, 3:] += rng.normal(0, 200, size=(3, 3))
    return x.astype(np.float32)


def _radar_signature(activity, rng):
    """(T_radar, K). Feature 0 ~ range/distance, 1 ~ velocity/doppler energy,
    others ~ gate energies."""
    T, K = CFG.t_radar, CFG.radar_k
    x = rng.normal(0, 0.1, size=(T, K)).astype(np.float32)
    base_range = rng.uniform(1.5, 4.0)
    x[:, 0] = base_range + rng.normal(0, 0.03, T)
    x[:, 1] = np.abs(_periodic(T, 0.7, 0.5, rng=rng))
    if activity == "stand_to_fall":
        drop = int(T * 0.45)
        x[:, 0] = base_range + rng.normal(0, 0.03, T)
        x[drop:, 0] = base_range - 0.6 + rng.normal(0, 0.05, T - drop)  # height drop
        x[drop:drop + 4, 1] += 3.0                                       # velocity burst
    # gate energies ~ overall motion
    x[:, 2:] += np.abs(x[:, 1:2]) * 0.5
    return x.astype(np.float32)


def _vision_signature(activity, rng):
    """(T_vis, D_v) fake per-frame embedding. First 3 dims encode a crude
    posture code so classes are separable; rest is noise."""
    T, D = CFG.t_vis, CFG.vision_dv
    x = rng.normal(0, 0.3, size=(T, D)).astype(np.float32)
    posture = {
        "stand": np.array([0.0, 1.0, 0.0]),
        "sit": np.array([0.0, 0.5, 0.5]),
        "lie": np.array([0.0, 0.0, 1.0]),
        "fall": np.array([0.0, 0.0, 1.0]),
    }
    start, end = activity.split("_to_")
    alpha = np.linspace(0.0, 1.0, T)[:, None]
    x[:, :3] += (1.0 - alpha) * posture[start] + alpha * posture[end]
    if activity == "stand_to_fall":
        x[T // 2:, :3] += 0.8
    return x.astype(np.float32)


def sample_window(activity: str, rng: np.random.Generator,
                  degrade: bool = True) -> FusionWindow:
    """Generate one FusionWindow for `activity`, optionally with sensor degradation."""
    imu = _imu_signature(activity, rng)
    radar = _radar_signature(activity, rng)
    vision = _vision_signature(activity, rng)

    imu_valid = radar_valid = vision_valid = True
    radar_energy = image_quality = imu_health = 1.0

    if degrade:
        # Independently degrade modalities to teach reliability-aware fusion.
        if rng.random() < 0.25:                          # darkness / occlusion
            image_quality = rng.uniform(0.0, 0.25)
            vision = vision * image_quality + rng.normal(0, 0.5, vision.shape)
            if image_quality < 0.1:
                vision_valid = False
        if rng.random() < 0.15:                          # radar noisy / weak
            radar_energy = rng.uniform(0.0, 0.3)
            radar = radar * radar_energy + rng.normal(0, 0.5, radar.shape)
            if radar_energy < 0.08:
                radar_valid = False
        if rng.random() < 0.10:                          # IMU saturated / off body
            imu_health = rng.uniform(0.0, 0.3)
            if imu_health < 0.08:
                imu_valid = False

    # The model masks invalid modalities. If every modality is invalid, the
    # attention softmax has no legal token and can produce NaNs, so keep the
    # least-bad sensor available for the window.
    if not (imu_valid or radar_valid or vision_valid):
        best = int(np.argmax([imu_health, radar_energy, image_quality]))
        if best == 0:
            imu_valid = True
        elif best == 1:
            radar_valid = True
        else:
            vision_valid = True

    return FusionWindow(
        t_start=0.0,
        imu=imu, radar=radar, vision=vision,
        imu_valid=imu_valid, radar_valid=radar_valid, vision_valid=vision_valid,
        radar_energy=float(radar_energy), image_quality=float(image_quality),
        imu_health=float(imu_health),
        label=LABEL2ID[activity],
    )


def make_dataset(n_per_class: int, seed: int = 0, degrade: bool = True):
    """Return a list[FusionWindow] balanced across activities."""
    from ..contract import ACTIVITIES
    rng = np.random.default_rng(seed)
    windows = []
    for act in ACTIVITIES:
        for _ in range(n_per_class):
            windows.append(sample_window(act, rng, degrade=degrade))
    rng.shuffle(windows)
    return windows


class MockRuntime:
    """A lightweight mock runtime for generating and replaying synthetic windows."""

    def __init__(self, seed: int = 0, degrade: bool = True):
        self.seed = seed
        self.degrade = degrade

    def generate(self, n_per_class: int = 10) -> List[FusionWindow]:
        return make_dataset(n_per_class=n_per_class, seed=self.seed, degrade=self.degrade)

    def record(self, output_path, n_per_class: int = 10) -> List[FusionWindow]:
        return record_simulation_output(
            output_path,
            n_per_class=n_per_class,
            seed=self.seed,
            degrade=self.degrade,
        )

    def replay(self, output_path) -> List[FusionWindow]:
        return load_simulation_output(output_path)


def _stack_or_empty(arrays, shape, dtype):
    if not arrays:
        return np.empty(shape, dtype=dtype)
    return np.stack(arrays, axis=0)


def record_simulation_output(output_path, n_per_class: int = 10, seed: int = 0,
                             degrade: bool = True) -> List[FusionWindow]:
    """Generate synthetic windows and persist them to disk as a replayable artifact."""
    windows = make_dataset(n_per_class=n_per_class, seed=seed, degrade=degrade)
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": np.array([1], dtype=np.int32),
        "n_windows": np.array([len(windows)], dtype=np.int32),
        "t_start": np.array([w.t_start for w in windows], dtype=np.float32),
        "imu": _stack_or_empty([w.imu for w in windows], (len(windows), CFG.t_imu, CFG.imu_ch), np.float32),
        "radar": _stack_or_empty([w.radar for w in windows], (len(windows), CFG.t_radar, CFG.radar_k), np.float32),
        "vision": _stack_or_empty([w.vision for w in windows], (len(windows), CFG.t_vis, CFG.vision_dv), np.float32),
        "imu_valid": _stack_or_empty([np.array([w.imu_valid], dtype=bool) for w in windows], (len(windows), 1), bool),
        "radar_valid": _stack_or_empty([np.array([w.radar_valid], dtype=bool) for w in windows], (len(windows), 1), bool),
        "vision_valid": _stack_or_empty([np.array([w.vision_valid], dtype=bool) for w in windows], (len(windows), 1), bool),
        "radar_energy": np.array([w.radar_energy for w in windows], dtype=np.float32),
        "image_quality": np.array([w.image_quality for w in windows], dtype=np.float32),
        "imu_health": np.array([w.imu_health for w in windows], dtype=np.float32),
        "label": np.array([w.label if w.label is not None else -1 for w in windows], dtype=np.int32),
    }
    np.savez_compressed(save_path, **payload)
    return windows


def load_simulation_output(output_path) -> List[FusionWindow]:
    """Reload previously recorded synthetic windows from disk."""
    data = np.load(Path(output_path), allow_pickle=False)
    windows = []
    for idx in range(int(data["n_windows"][0])):
        windows.append(
            FusionWindow(
                t_start=float(data["t_start"][idx]),
                imu=data["imu"][idx].astype(np.float32),
                radar=data["radar"][idx].astype(np.float32),
                vision=data["vision"][idx].astype(np.float32),
                imu_valid=bool(data["imu_valid"][idx, 0]),
                radar_valid=bool(data["radar_valid"][idx, 0]),
                vision_valid=bool(data["vision_valid"][idx, 0]),
                radar_energy=float(data["radar_energy"][idx]),
                image_quality=float(data["image_quality"][idx]),
                imu_health=float(data["imu_health"][idx]),
                label=int(data["label"][idx]),
            )
        )
    return windows
