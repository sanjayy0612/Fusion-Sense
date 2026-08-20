"""Run layered health, snapshot, and stream checks against diagnostic firmware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_stream import (  # noqa: E402
    CameraConnectionError,
    TimestampedMjpegStream,
)


FIRMWARE_BUILD = "camera-diagnostic-v1"
TARGET_FPS = 10.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def read_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def decode_jpeg(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError("response is not a complete JPEG")
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for camera diagnostics") from exc
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the JPEG")
    height, width = image.shape[:2]
    return width, height


def snapshot_gate(host: str, count: int = 3) -> dict[str, Any]:
    durations_ms: list[float] = []
    dimensions: list[list[int]] = []
    sizes: list[int] = []
    timestamps: list[int] = []
    for _ in range(count):
        started = time.monotonic()
        request = Request(f"http://{host}/capture", headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=5.0) as response:
            payload = response.read()
            timestamp = int(response.headers["X-Capture-Timestamp-Us"])
        durations_ms.append((time.monotonic() - started) * 1000.0)
        dimensions.append(list(decode_jpeg(payload)))
        sizes.append(len(payload))
        timestamps.append(timestamp)

    checks = {
        "three_complete_jpegs": len(sizes) == count,
        "qvga_320_by_240": all(item == [320, 240] for item in dimensions),
        "positive_jpeg_sizes": all(item > 0 for item in sizes),
        "monotonic_capture_timestamps": all(
            current > previous for previous, current in zip(timestamps, timestamps[1:])
        ),
        "p95_response_at_most_2000_ms": percentile(durations_ms, 0.95) <= 2000.0,
    }
    return {
        "checks": checks,
        "durations_ms": {
            "mean": round(statistics.fmean(durations_ms), 3),
            "p95": round(percentile(durations_ms, 0.95), 3),
            "max": round(max(durations_ms), 3),
        },
        "dimensions": dimensions,
        "jpeg_bytes": sizes,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }


def stream_gate(host: str, duration_s: float = 10.0) -> dict[str, Any]:
    url = f"http://{host}:8080/stream"
    frames = []
    started = time.monotonic()
    with TimestampedMjpegStream(url, timeout=5.0) as stream:
        while time.monotonic() - started < duration_s:
            frames.append(stream.read())

    timestamps = [frame.device_timestamp_us for frame in frames]
    sequences = [frame.sequence for frame in frames]
    intervals_ms = [
        (current - previous) / 1000.0
        for previous, current in zip(timestamps, timestamps[1:])
        if previous is not None and current is not None
    ]
    device_elapsed_s = (
        (timestamps[-1] - timestamps[0]) / 1_000_000.0
        if len(timestamps) >= 2 and timestamps[0] is not None and timestamps[-1] is not None
        else 0.0
    )
    device_fps = (len(frames) - 1) / device_elapsed_s if device_elapsed_s > 0 else 0.0
    checks = {
        "enough_frames": len(frames) >= 85,
        "device_fps_8_5_to_11_5": 8.5 <= device_fps <= 11.5,
        "qvga_320_by_240": all(frame.image.shape[:2] == (240, 320) for frame in frames),
        "monotonic_capture_timestamps": all(
            current is not None and previous is not None and current > previous
            for previous, current in zip(timestamps, timestamps[1:])
        ),
        "monotonic_sequence": all(
            current is not None and previous is not None and current > previous
            for previous, current in zip(sequences, sequences[1:])
        ),
        "no_sequence_gaps": all(
            current is not None and previous is not None and current == previous + 1
            for previous, current in zip(sequences, sequences[1:])
        ),
        "p95_interval_at_most_200_ms": bool(intervals_ms)
        and percentile(intervals_ms, 0.95) <= 200.0,
        "max_interval_at_most_500_ms": bool(intervals_ms) and max(intervals_ms) <= 500.0,
    }
    return {
        "url": url,
        "frames": len(frames),
        "device_elapsed_s": round(device_elapsed_s, 4),
        "device_fps": round(device_fps, 4),
        "capture_interval_ms": {
            "mean": round(statistics.fmean(intervals_ms), 3) if intervals_ms else None,
            "p95": round(percentile(intervals_ms, 0.95), 3) if intervals_ms else None,
            "max": round(max(intervals_ms), 3) if intervals_ms else None,
        },
        "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="ESP32-CAM IP address")
    parser.add_argument("--stream-seconds", type=float, default=10.0)
    args = parser.parse_args()
    host = args.host.strip().removeprefix("http://").removeprefix("https://").split("/")[0]

    report: dict[str, Any] = {"host": host, "errors": []}
    try:
        initial_health = read_json(f"http://{host}/health")
        local_test = initial_health.get("local_self_test", {})
        health_checks = {
            "diagnostic_firmware": initial_health.get("firmware") == FIRMWARE_BUILD,
            "status_ok": initial_health.get("status") == "ok",
            "psram_detected": initial_health.get("psram") is True,
            "local_self_test_pass": local_test.get("result") == "PASS",
            "local_fps_at_least_8_5": float(local_test.get("fps", 0.0)) >= 8.5,
        }
        report["health"] = {
            "checks": health_checks,
            "response": initial_health,
            "result": "PASS" if all(health_checks.values()) else "FAIL",
        }
        report["snapshots"] = snapshot_gate(host)
        report["stream"] = stream_gate(host, args.stream_seconds)
        report["final_health"] = read_json(f"http://{host}/health")
    except (CameraConnectionError, OSError, RuntimeError, ValueError, KeyError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")

    sections = ("health", "snapshots", "stream")
    passed = not report["errors"] and all(
        report.get(section, {}).get("result") == "PASS" for section in sections
    )
    report["result"] = "PASS" if passed else "FAIL"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
