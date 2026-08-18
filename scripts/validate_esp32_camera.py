"""Validate timestamped ESP32-CAM JPEG capture without running MediaPipe."""

from __future__ import annotations

import argparse
import json
import math
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_stream import (
    CameraConnectionError,
    TimestampedMjpegStream,
    esp32_stream_url,
)


@dataclass(frozen=True)
class FrameObservation:
    sequence: int
    device_timestamp_us: int
    host_received_s: float
    width: int
    height: int


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def analyze_observations(
    observations: list[FrameObservation],
    *,
    requested_duration_s: float,
    target_fps: float,
    health: dict[str, object] | None = None,
) -> dict[str, object]:
    if len(observations) < 2:
        return {
            "result": "FAIL",
            "reason": "fewer than two valid timestamped frames",
            "frames": len(observations),
            "checks": {"enough_frames": False},
        }

    sequences = [item.sequence for item in observations]
    timestamps_us = [item.device_timestamp_us for item in observations]
    sequence_deltas = [
        current - previous for previous, current in zip(sequences, sequences[1:])
    ]
    timestamp_deltas_ms = [
        (current - previous) / 1000.0
        for previous, current in zip(timestamps_us, timestamps_us[1:])
    ]
    device_elapsed_s = (timestamps_us[-1] - timestamps_us[0]) / 1_000_000.0
    device_fps = (
        (len(observations) - 1) / device_elapsed_s if device_elapsed_s > 0 else 0.0
    )
    dropped_frames = sum(max(0, delta - 1) for delta in sequence_deltas)
    minimum_frames = max(2, int(requested_duration_s * target_fps * 0.8))
    dimensions = sorted({(item.width, item.height) for item in observations})
    target_interval_ms = 1000.0 / target_fps

    checks: dict[str, bool] = {
        "enough_frames": len(observations) >= minimum_frames,
        "device_fps_8_5_to_11_5": 8.5 <= device_fps <= 11.5,
        "monotonic_sequence": all(delta > 0 for delta in sequence_deltas),
        "no_sequence_gaps": dropped_frames == 0,
        "monotonic_capture_timestamps": all(
            delta > 0 for delta in timestamp_deltas_ms
        ),
        "p95_capture_interval_at_most_2x_target": (
            percentile(timestamp_deltas_ms, 0.95) <= target_interval_ms * 2.0
        ),
        "max_capture_interval_at_most_5x_target": (
            max(timestamp_deltas_ms) <= target_interval_ms * 5.0
        ),
        "qvga_320_by_240": dimensions == [(320, 240)],
    }
    if health is not None:
        checks["camera_capture_errors_zero"] = health.get("capture_errors") == 0

    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "frames": len(observations),
        "sequence": {"first": sequences[0], "last": sequences[-1]},
        "device_timestamp_us": {
            "first": timestamps_us[0],
            "last": timestamps_us[-1],
        },
        "device_elapsed_s": round(device_elapsed_s, 4),
        "device_fps": round(device_fps, 4),
        "capture_interval_ms": {
            "mean": round(statistics.fmean(timestamp_deltas_ms), 4),
            "p95": round(percentile(timestamp_deltas_ms, 0.95), 4),
            "max": round(max(timestamp_deltas_ms), 4),
        },
        "dropped_frames": dropped_frames,
        "dimensions": [list(item) for item in dimensions],
        "health": health,
        "checks": checks,
    }


def read_health(host: str) -> dict[str, object] | None:
    host_value = host.strip().rstrip("/")
    if host_value.startswith(("http://", "https://")):
        host_value = host_value.split("://", 1)[1]
    host_value = host_value.split("/", 1)[0].split(":", 1)[0]
    # Step 4 keeps port 80 responsive while the MJPEG handler owns port 81.
    # Fall back to the Step 2 endpoint so older firmware remains testable.
    for url in (
        f"http://{host_value}/health",
        f"http://{host_value}:81/health",
    ):
        try:
            with urlopen(url, timeout=5.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="ESP32-CAM IP or hostname")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = esp32_stream_url(args.host)
    observations: list[FrameObservation] = []
    print(f"Opening timestamped stream: {source}")

    try:
        with TimestampedMjpegStream(source) as camera:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                frame = camera.read()
                if frame.sequence is None or frame.device_timestamp_us is None:
                    raise CameraConnectionError("frame metadata unexpectedly missing")
                height, width = frame.image.shape[:2]
                observations.append(
                    FrameObservation(
                        sequence=frame.sequence,
                        device_timestamp_us=frame.device_timestamp_us,
                        host_received_s=frame.captured_at,
                        width=width,
                        height=height,
                    )
                )
    except (CameraConnectionError, OSError) as error:
        print(
            json.dumps(
                {
                    "result": "FAIL",
                    "reason": str(error),
                    "frames": len(observations),
                },
                indent=2,
            )
        )
        return 1

    health = read_health(args.host)
    report = analyze_observations(
        observations,
        requested_duration_s=args.duration,
        target_fps=args.target_fps,
        health=health,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
