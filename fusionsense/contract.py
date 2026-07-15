"""FusionSense data contract.

The single most important interface in the project. The simulator and (later)
the real hardware capture code both emit exactly this object, so everything
downstream (model, training, inference) is hardware-agnostic.

Shapes (for the default 2s window config):
    imu    : (100, 6)   -> 50 Hz * 2 s, [ax, ay, az, gx, gy, gz]
    radar  : (40, K)    -> 20 Hz * 2 s, K per-frame radar features
    vision : (20, D_v)  -> 10 fps * 2 s, per-frame vision *embedding* (not pixels)

Health signals (the V2 "unique angle" hook — cheap scalars the real sensors
report for free; the simulator fakes them for now):
    radar_energy  in [0,1]  -> LD2410 gate energy / signal strength
    image_quality in [0,1]  -> brightness * sharpness (low in dark / blur)
    imu_health    in [0,1]  -> 1 - clipping fraction (low if saturated/disconnected)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Canonical activity label set for v1.
ACTIVITIES = ["walking", "standing", "sitting", "lying", "falling"]
LABEL2ID = {name: i for i, name in enumerate(ACTIVITIES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


@dataclass
class FusionWindow:
    """One synchronized, fixed-size snapshot of all three sensors for one window."""

    t_start: float                 # epoch seconds, window start

    imu: np.ndarray                # (T_imu, 6)   float32
    radar: np.ndarray              # (T_radar, K) float32
    vision: np.ndarray             # (T_vis, D_v) float32

    # validity flags -> graceful degradation (a dropped modality is masked out)
    imu_valid: bool = True
    radar_valid: bool = True
    vision_valid: bool = True

    # sensor-health scalars (the reliability-prior hook). Range [0, 1].
    radar_energy: float = 1.0
    image_quality: float = 1.0
    imu_health: float = 1.0

    label: Optional[int] = None    # set only for training/eval data

    def valid_vector(self) -> np.ndarray:
        return np.array(
            [self.imu_valid, self.radar_valid, self.vision_valid], dtype=bool
        )

    def health_vector(self) -> np.ndarray:
        """Order matches the token order [imu, radar, vision]."""
        return np.array(
            [self.imu_health, self.radar_energy, self.image_quality],
            dtype=np.float32,
        )
