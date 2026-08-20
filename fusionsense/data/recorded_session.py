"""Convert a timestamped ESP32 recording into model-ready fusion windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from ..config import CFG
from ..contract import FusionWindow
from .imu_units import (
    convert_imu_to_canonical,
    live_conversion_metadata,
    validate_canonical_imu,
)


@dataclass(frozen=True)
class SessionWindow:
    window: FusionWindow
    diagnostics: dict


def summarize_session_windows(
    outputs: list[SessionWindow], metadata: dict, *, cfg=CFG
) -> dict:
    """Return the Step 5 acceptance report for assembled model windows."""
    imu_valid_count = sum(item.window.imu_valid for item in outputs)
    vision_valid_count = sum(item.window.vision_valid for item in outputs)
    usable_count = sum(
        item.window.imu_valid or item.window.vision_valid for item in outputs
    )
    imu_coverages = [item.diagnostics["imu"]["coverage"] for item in outputs]
    camera_coverages = [
        item.diagnostics["camera"]["sampling_coverage"] for item in outputs
    ]
    pose_ratios = [
        item.diagnostics["camera"]["pose_valid_ratio"] for item in outputs
    ]
    imu_skews = [
        item.diagnostics["imu"]["max_sampling_skew_ms"] for item in outputs
    ]
    camera_skews = [
        item.diagnostics["camera"]["max_sampling_skew_ms"] for item in outputs
    ]
    capture_result = metadata.get("capture_validation", {}).get("result")
    unit_validation = metadata.get("imu_input_validation", {})
    checks = {
        "step4_capture_passed": capture_result == "PASS",
        "at_least_one_window": bool(outputs),
        "canonical_imu_units_valid": unit_validation.get("valid") is True,
        "all_imu_shapes_100_by_6": all(
            item.window.imu.shape == (cfg.t_imu, cfg.imu_ch) for item in outputs
        ),
        "all_vision_shapes_20_by_99": all(
            item.window.vision.shape == (cfg.t_vis, cfg.vision_dv)
            for item in outputs
        ),
        "all_windows_have_a_usable_modality": usable_count == len(outputs),
        "all_windows_have_valid_imu": imu_valid_count == len(outputs),
    }

    def summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": round(float(np.min(values)), 4),
            "median": round(float(np.median(values)), 4),
            "max": round(float(np.max(values)), 4),
        }

    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "windows": len(outputs),
        "validity": {
            "imu_valid_windows": imu_valid_count,
            "vision_valid_windows": vision_valid_count,
            "usable_modality_windows": usable_count,
        },
        "diagnostics": {
            "imu_coverage": summary(imu_coverages),
            "camera_sampling_coverage": summary(camera_coverages),
            "pose_valid_ratio": summary(pose_ratios),
            "imu_max_sampling_skew_ms": summary(imu_skews),
            "camera_max_sampling_skew_ms": summary(camera_skews),
        },
        "imu_input_validation": unit_validation,
        "capture_validation_result": capture_result,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source_ns) == 0:
        raise ValueError("Cannot resample an empty timestamp sequence")
    right = np.searchsorted(source_ns, target_ns, side="left")
    right = np.clip(right, 0, len(source_ns) - 1)
    left = np.clip(right - 1, 0, len(source_ns) - 1)
    choose_left = np.abs(target_ns - source_ns[left]) <= np.abs(source_ns[right] - target_ns)
    indices = np.where(choose_left, left, right)
    skew_ns = np.abs(source_ns[indices] - target_ns)
    return indices.astype(np.int64), skew_ns.astype(np.int64)


def load_live_imu_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read ESP32 IMU rows and convert acceleration from g to m/s²."""
    rows = _read_csv(Path(path))
    if not rows:
        raise ValueError(f"No IMU rows in {path}")
    timestamps = np.array(
        [int(row["host_capture_monotonic_ns"]) for row in rows], dtype=np.int64
    )
    raw = np.array(
        [
            [
                float(row["ax_g"]),
                float(row["ay_g"]),
                float(row["az_g"]),
                float(row["gx_dps"]),
                float(row["gy_dps"]),
                float(row["gz_dps"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"IMU capture timestamps must be strictly increasing in {path}")
    canonical = convert_imu_to_canonical(raw, acceleration_unit="g")
    validation = validate_canonical_imu(canonical).as_dict()
    validation["conversion"] = live_conversion_metadata()
    if not validation["valid"]:
        raise ValueError(
            "Converted IMU values failed the canonical m/s^2 scale check: "
            f"{validation}"
        )
    return timestamps, canonical, validation


def load_camera_manifest(path: str | Path) -> tuple[np.ndarray, list[Path], np.ndarray]:
    manifest = Path(path)
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError(f"No camera rows in {path}")
    timestamps = np.array(
        [int(row["host_capture_monotonic_ns"]) for row in rows], dtype=np.int64
    )
    frame_paths = [manifest.parent / row["relative_path"] for row in rows]
    sequences = np.array([int(row["sequence"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Camera capture timestamps must be strictly increasing in {path}")
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} camera frames; first is {missing[0]}")
    return timestamps, frame_paths, sequences


def extract_pose_cache(
    session_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Decode each captured JPEG once and cache timestamped MediaPipe features."""
    from .vision_extractor import POSE_DIM, PoseExtractor

    session_dir = Path(session_dir)
    output = Path(output_path or (session_dir / "pose_cache.npz"))
    if output.is_file() and not force:
        return output

    timestamps, frame_paths, sequences = load_camera_manifest(
        session_dir / "camera_manifest.csv"
    )
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Recorded camera replay needs opencv-python") from exc

    landmarks = np.zeros((len(frame_paths), POSE_DIM), dtype=np.float32)
    valid = np.zeros(len(frame_paths), dtype=bool)
    quality = np.zeros(len(frame_paths), dtype=np.float32)
    relative_seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    with PoseExtractor() as extractor:
        for index, (path, captured_at) in enumerate(zip(frame_paths, relative_seconds)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            pose = extractor.extract(frame, float(captured_at))
            landmarks[index] = pose.landmarks
            valid[index] = pose.valid
            quality[index] = pose.image_quality

    np.savez_compressed(
        output,
        version=np.array([1], dtype=np.int32),
        host_capture_monotonic_ns=timestamps,
        sequence=sequences,
        landmarks=landmarks,
        valid=valid,
        image_quality=quality,
    )
    return output


def _load_pose_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return (
        data["host_capture_monotonic_ns"].astype(np.int64),
        data["landmarks"].astype(np.float32),
        data["valid"].astype(bool),
        data["image_quality"].astype(np.float32),
    )


def _resample_imu(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    target_ns: np.ndarray,
) -> np.ndarray:
    source = timestamps_ns.astype(np.float64)
    target = target_ns.astype(np.float64)
    return np.stack(
        [np.interp(target, source, values[:, channel]) for channel in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def build_recorded_session_windows(
    session_dir: str | Path,
    *,
    stride_seconds: float = 1.0,
    pose_cache_path: str | Path | None = None,
    extract_poses: bool = True,
    cfg=CFG,
) -> tuple[list[SessionWindow], dict]:
    """Assemble synchronized two-second windows on the laptop clock domain."""
    session_dir = Path(session_dir)
    imu_ns, imu_values, imu_validation = load_live_imu_csv(session_dir / "imu.csv")
    camera_ns, _, _ = load_camera_manifest(session_dir / "camera_manifest.csv")

    pose_cache = Path(pose_cache_path or (session_dir / "pose_cache.npz"))
    if not pose_cache.is_file() and extract_poses:
        pose_cache = extract_pose_cache(session_dir, output_path=pose_cache)
    if pose_cache.is_file():
        pose_ns, pose_values, pose_valid, pose_quality = _load_pose_cache(pose_cache)
    else:
        pose_ns = camera_ns
        pose_values = np.zeros((len(camera_ns), cfg.vision_dv), dtype=np.float32)
        pose_valid = np.zeros(len(camera_ns), dtype=bool)
        pose_quality = np.zeros(len(camera_ns), dtype=np.float32)

    start_ns = max(int(imu_ns[0]), int(camera_ns[0]), int(pose_ns[0]))
    end_ns = min(int(imu_ns[-1]), int(camera_ns[-1]), int(pose_ns[-1]))
    duration_ns = int(cfg.window_seconds * 1e9)
    stride_ns = int(stride_seconds * 1e9)
    if end_ns - start_ns < duration_ns:
        raise ValueError("The shared IMU/camera interval is shorter than one model window")

    session_json = session_dir / "session.json"
    session_summary = json.loads(session_json.read_text(encoding="utf-8")) if session_json.is_file() else {}
    session_id = session_summary.get("session_id", session_dir.name)
    outputs: list[SessionWindow] = []

    for window_start in range(start_ns, end_ns - duration_ns + 1, stride_ns):
        imu_target = window_start + (
            np.arange(cfg.t_imu, dtype=np.int64) * duration_ns // cfg.t_imu
        )
        vision_target = window_start + (
            np.arange(cfg.t_vis, dtype=np.int64) * duration_ns // cfg.t_vis
        )

        imu_indices, imu_skew = _nearest_indices(imu_ns, imu_target)
        vision_indices, vision_skew = _nearest_indices(pose_ns, vision_target)
        imu = _resample_imu(imu_ns, imu_values, imu_target)
        vision = pose_values[vision_indices]

        imu_coverage = float(np.mean(imu_skew <= 30_000_000))
        vision_sampling_coverage = float(np.mean(vision_skew <= 250_000_000))
        pose_valid_ratio = float(np.mean(pose_valid[vision_indices]))
        imu_valid = imu_coverage >= 0.90 and int(np.max(imu_skew)) <= 100_000_000
        vision_valid = (
            vision_sampling_coverage >= 0.80
            and pose_valid_ratio >= 0.50
            and int(np.max(vision_skew)) <= 250_000_000
        )
        imu_health = float(np.clip(imu_coverage, 0.0, 1.0))
        image_quality = float(np.mean(pose_quality[vision_indices]))

        window = FusionWindow(
            t_start=window_start / 1e9,
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
        diagnostics = {
            "window_start_s": round((window_start - start_ns) / 1e9, 3),
            "window_end_s": round((window_start - start_ns) / 1e9 + cfg.window_seconds, 3),
            "imu": {
                "valid": imu_valid,
                "coverage": round(imu_coverage, 4),
                "max_sampling_skew_ms": round(float(np.max(imu_skew)) / 1e6, 3),
                "nearest_source_samples": int(len(np.unique(imu_indices))),
            },
            "camera": {
                "valid": vision_valid,
                "sampling_coverage": round(vision_sampling_coverage, 4),
                "pose_valid_ratio": round(pose_valid_ratio, 4),
                "max_sampling_skew_ms": round(float(np.max(vision_skew)) / 1e6, 3),
                "nearest_source_frames": int(len(np.unique(vision_indices))),
            },
        }
        outputs.append(SessionWindow(window=window, diagnostics=diagnostics))

    metadata = {
        "source": "recorded_esp32_session",
        "session_id": session_id,
        "session_dir": str(session_dir.resolve()),
        "window_seconds": cfg.window_seconds,
        "stride_seconds": stride_seconds,
        "windows": len(outputs),
        "imu_input_validation": imu_validation,
        "capture_validation": session_summary.get("validation", {}),
        "pose_cache": str(pose_cache.resolve()) if pose_cache.is_file() else None,
    }
    return outputs, metadata
