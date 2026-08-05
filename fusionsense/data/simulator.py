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
        sig = sig + rng.normal(0, 0.05 * amp + 1e-3, size=T)
    return sig


def _imu_signature(activity, rng):
    """(T_imu, 6) accel(3)+gyro(3)."""
    T = CFG.t_imu
    x = np.zeros((T, 6), dtype=np.float32)
    g = 9.81
    if activity == "walking":
        for a in range(3):
            x[:, a] = _periodic(T, freq=2.0, amp=2.0, phase=a, rng=rng)
        x[:, 2] += g
        for a in range(3, 6):
            x[:, a] = _periodic(T, freq=2.0, amp=30.0, phase=a, rng=rng)
    elif activity in ("standing", "sitting", "lying"):
        # mostly static; gravity vector orientation differs per posture
        grav = {"standing": [0, 0, g], "sitting": [0, 0.4 * g, 0.9 * g],
                "lying": [g, 0, 0]}[activity]
        for a in range(3):
            x[:, a] = grav[a] + rng.normal(0, 0.15, size=T)
        for a in range(3, 6):
            x[:, a] = rng.normal(0, 1.0, size=T)
    elif activity == "falling":
        # free-fall dip then sharp impact spike, then still
        x[:, :3] = rng.normal(0, 0.2, size=(T, 3))
        x[:, 2] += g
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
    if activity == "walking":
        x[:, 0] = base_range - np.linspace(0, 1.2, T) + rng.normal(0, 0.05, T)
        x[:, 1] = np.abs(_periodic(T, 2.0, 1.0, rng=rng)) + 0.5
    elif activity in ("standing", "sitting"):
        x[:, 0] = base_range + rng.normal(0, 0.03, T)
        x[:, 1] = np.abs(rng.normal(0, 0.1, T))
    elif activity == "lying":
        x[:, 0] = base_range + rng.normal(0, 0.03, T)
        x[:, 1] = np.abs(rng.normal(0, 0.05, T))
    elif activity == "falling":
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
    code = {"walking": [1, 0, 0], "standing": [0, 1, 0], "sitting": [0, 0.5, 0.5],
            "lying": [0, 0, 1], "falling": [0, 0, 1]}[activity]
    for i, c in enumerate(code):
        x[:, i] += c
    if activity == "walking":
        x[:, 0] += _periodic(T, 2.0, 0.3, rng=rng)      # gait bob
    if activity == "falling":
        half = T // 2
        x[:half, 1] += 1.0                              # upright -> down transition
        x[half:, 2] += 1.0
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
