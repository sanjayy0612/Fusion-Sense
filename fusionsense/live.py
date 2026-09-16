"""Rolling synchronized IMU/camera windows for laptop live inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CFG
from .contract import FusionWindow
from .data.imu_units import convert_imu_to_canonical, validate_canonical_imu


@dataclass(frozen=True)
class LiveImuPoint:
    capture_ns: int
    values_g_dps: np.ndarray
    sequence: int | None = None


@dataclass(frozen=True)
class LivePosePoint:
    capture_ns: int
    landmarks: np.ndarray
    valid: bool
    image_quality: float
    sequence: int | None = None


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray):
    insertion = np.searchsorted(source_ns, target_ns, side="left")
    right = np.clip(insertion, 0, len(source_ns) - 1)
    left = np.clip(insertion - 1, 0, len(source_ns) - 1)
    choose_right = np.abs(source_ns[right] - target_ns) < np.abs(
        source_ns[left] - target_ns
    )
    indices = np.where(choose_right, right, left)
    skew_ns = np.abs(source_ns[indices] - target_ns)
    return indices.astype(np.int64), skew_ns.astype(np.int64)


def _sequence_gaps(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    concrete = [int(value) for value in values]
    return sum(
        max(0, current - previous - 1)
        for previous, current in zip(concrete, concrete[1:])
    )


def _coalesce_timestamp_duplicates(points):
    """Keep the latest point for a duplicated host timestamp."""
    by_timestamp = {}
    for point in sorted(points, key=lambda item: item.capture_ns):
        by_timestamp[point.capture_ns] = point
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def assemble_live_window(
    imu_points: list[LiveImuPoint],
    pose_points: list[LivePosePoint],
    *,
    window_end_ns: int,
    session_id: str,
    cfg=CFG,
) -> tuple[FusionWindow, dict]:
    """Resample recent points into one strict FusionSense input contract."""
    if len(imu_points) < 2:
        raise ValueError("live window needs at least two IMU points")
    if not pose_points:
        raise ValueError("live window needs at least one camera pose point")

    original_imu_count = len(imu_points)
    original_pose_count = len(pose_points)
    imu_points = _coalesce_timestamp_duplicates(imu_points)
    pose_points = _coalesce_timestamp_duplicates(pose_points)
    if len(imu_points) < 2:
        raise ValueError("live window needs two distinct IMU timestamps")
    imu_ns = np.asarray([item.capture_ns for item in imu_points], dtype=np.int64)
    pose_ns = np.asarray([item.capture_ns for item in pose_points], dtype=np.int64)
    if np.any(np.diff(imu_ns) <= 0) or np.any(np.diff(pose_ns) <= 0):
        raise ValueError("live source timestamps must be strictly increasing")

    duration_ns = int(cfg.window_seconds * 1e9)
    start_ns = int(window_end_ns) - duration_ns
    imu_target_ns = start_ns + (
        np.arange(cfg.t_imu, dtype=np.int64) * duration_ns // cfg.t_imu
    )
    pose_target_ns = start_ns + (
        np.arange(cfg.t_vis, dtype=np.int64) * duration_ns // cfg.t_vis
    )

    imu_raw = np.stack([item.values_g_dps for item in imu_points]).astype(np.float32)
    if imu_raw.shape[1:] != (cfg.imu_ch,):
        raise ValueError(f"live IMU points must contain {cfg.imu_ch} channels")
    imu = np.stack(
        [
            np.interp(imu_target_ns, imu_ns, imu_raw[:, channel])
            for channel in range(cfg.imu_ch)
        ],
        axis=1,
    ).astype(np.float32)
    imu = convert_imu_to_canonical(imu, acceleration_unit="g")

    imu_indices, imu_skew = _nearest_indices(imu_ns, imu_target_ns)
    pose_indices, pose_skew = _nearest_indices(pose_ns, pose_target_ns)
    pose_values = np.stack([item.landmarks for item in pose_points]).astype(np.float32)
    if pose_values.shape[1:] != (cfg.vision_dv,):
        raise ValueError(f"live pose points must contain {cfg.vision_dv} values")
    vision = pose_values[pose_indices]
    pose_valid = np.asarray([item.valid for item in pose_points], dtype=bool)
    qualities = np.asarray(
        [item.image_quality for item in pose_points], dtype=np.float32
    )

    imu_coverage = float(np.mean(imu_skew <= 30_000_000))
    pose_sampling_coverage = float(np.mean(pose_skew <= 250_000_000))
    pose_valid_ratio = float(np.mean(pose_valid[pose_indices]))
    unit_validation = validate_canonical_imu(imu)
    imu_valid = bool(
        imu_coverage >= 0.90
        and int(np.max(imu_skew)) <= 100_000_000
        and unit_validation.valid
    )
    vision_valid = bool(
        pose_sampling_coverage >= 0.80
        and pose_valid_ratio >= 0.50
        and int(np.max(pose_skew)) <= 250_000_000
    )

    accel_clipped = np.any(np.abs(imu_raw[imu_indices, :3]) >= 1.98, axis=1)
    gyro_clipped = np.any(np.abs(imu_raw[imu_indices, 3:]) >= 247.5, axis=1)
    clipping_ratio = float(np.mean(accel_clipped | gyro_clipped))
    imu_health = float(np.clip(imu_coverage * (1.0 - clipping_ratio), 0.0, 1.0))
    image_quality = float(np.clip(np.mean(qualities[pose_indices]), 0.0, 1.0))

    selected_imu = [imu_points[index].sequence for index in np.unique(imu_indices)]
    selected_pose = [pose_points[index].sequence for index in np.unique(pose_indices)]
    diagnostics = {
        "window_start_monotonic_ns": start_ns,
        "window_end_monotonic_ns": int(window_end_ns),
        "imu": {
            "valid": imu_valid,
            "coverage": round(imu_coverage, 4),
            "health": round(imu_health, 4),
            "clipping_ratio": round(clipping_ratio, 4),
            "max_sampling_skew_ms": round(float(np.max(imu_skew)) / 1e6, 3),
            "source_samples": int(len(np.unique(imu_indices))),
            "sequence_gaps": _sequence_gaps(selected_imu),
            "duplicate_timestamps_discarded": original_imu_count - len(imu_points),
            "unit_validation": unit_validation.as_dict(),
        },
        "camera": {
            "valid": vision_valid,
            "sampling_coverage": round(pose_sampling_coverage, 4),
            "pose_valid_ratio": round(pose_valid_ratio, 4),
            "health": round(image_quality, 4),
            "max_sampling_skew_ms": round(float(np.max(pose_skew)) / 1e6, 3),
            "source_frames": int(len(np.unique(pose_indices))),
            "sequence_gaps": _sequence_gaps(selected_pose),
            "duplicate_timestamps_discarded": original_pose_count - len(pose_points),
        },
    }
    window = FusionWindow(
        t_start=start_ns / 1e9,
        imu=imu,
        radar=np.zeros((cfg.t_radar, cfg.radar_k), dtype=np.float32),
        vision=vision.astype(np.float32),
        imu_valid=imu_valid,
        radar_valid=False,
        vision_valid=vision_valid,
        radar_energy=0.0,
        image_quality=image_quality,
        imu_health=imu_health,
        label=None,
        subject_id="live_esp32",
        recording_id=session_id,
    )
    return window, diagnostics
