"""Capture and validate legacy or versioned FusionSense IMU serial streams."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO


@dataclass(frozen=True)
class ImuSample:
    host_monotonic_ns: int
    t_ms: int
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float
    schema_version: int | None = None
    device_id: str | None = None
    session_id: str | None = None
    sequence: int | None = None
    device_timestamp_us: int | None = None


def parse_sample(line: str, host_monotonic_ns: int) -> ImuSample | None:
    """Parse one data row; comments and blank lines are intentionally ignored."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) == 7:
        sample = ImuSample(
            host_monotonic_ns=host_monotonic_ns,
            t_ms=int(fields[0]),
            ax_g=float(fields[1]),
            ay_g=float(fields[2]),
            az_g=float(fields[3]),
            gx_dps=float(fields[4]),
            gy_dps=float(fields[5]),
            gz_dps=float(fields[6]),
        )
    elif len(fields) == 12 and fields[0] == "IMU":
        schema_version = int(fields[1])
        if schema_version != 1:
            raise ValueError(f"unsupported IMU schema version: {schema_version}")
        device_timestamp_us = int(fields[5])
        sample = ImuSample(
            host_monotonic_ns=host_monotonic_ns,
            t_ms=device_timestamp_us // 1000,
            ax_g=float(fields[6]),
            ay_g=float(fields[7]),
            az_g=float(fields[8]),
            gx_dps=float(fields[9]),
            gy_dps=float(fields[10]),
            gz_dps=float(fields[11]),
            schema_version=schema_version,
            device_id=fields[2],
            session_id=fields[3],
            sequence=int(fields[4]),
            device_timestamp_us=device_timestamp_us,
        )
        if not sample.device_id or not sample.session_id or sample.sequence < 0:
            raise ValueError("invalid versioned IMU identity or sequence")
    else:
        raise ValueError(
            f"expected legacy 7-field or versioned 12-field IMU row; "
            f"received {len(fields)} fields"
        )

    numeric_values = (
        sample.host_monotonic_ns,
        sample.t_ms,
        sample.ax_g,
        sample.ay_g,
        sample.az_g,
        sample.gx_dps,
        sample.gy_dps,
        sample.gz_dps,
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("row contains a non-finite value")
    return sample


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _pstdev(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.pstdev(materialized) if len(materialized) > 1 else 0.0


def analyze_samples(
    samples: list[ImuSample],
    *,
    invalid_rows: int = 0,
    target_hz: float = 50.0,
    stationary: bool = True,
) -> dict[str, object]:
    """Return measurable Step 1 metrics and acceptance checks."""
    if len(samples) < 2:
        return {
            "result": "FAIL",
            "reason": "fewer than two valid samples",
            "samples": len(samples),
            "invalid_rows": invalid_rows,
            "checks": {"enough_samples": False},
        }

    expected_interval_ms = 1000.0 / target_hz
    device_times_ms = [
        (
            sample.device_timestamp_us / 1000.0
            if sample.device_timestamp_us is not None
            else float(sample.t_ms)
        )
        for sample in samples
    ]
    intervals_ms = [
        current - previous
        for previous, current in zip(device_times_ms, device_times_ms[1:])
    ]
    positive_intervals = [interval for interval in intervals_ms if interval > 0]
    non_monotonic = len(intervals_ms) - len(positive_intervals)
    device_duration_s = (device_times_ms[-1] - device_times_ms[0]) / 1000.0
    effective_hz = (
        (len(samples) - 1) / device_duration_s if device_duration_s > 0 else 0.0
    )

    missed_slots = sum(
        max(0, round(interval / expected_interval_ms) - 1)
        for interval in positive_intervals
        if interval > expected_interval_ms * 1.5
    )
    expected_slots = max(1, len(samples) + missed_slots)
    missed_ratio = missed_slots / expected_slots

    accel_magnitudes = [
        math.sqrt(sample.ax_g**2 + sample.ay_g**2 + sample.az_g**2)
        for sample in samples
    ]
    gyro_magnitudes = [
        math.sqrt(sample.gx_dps**2 + sample.gy_dps**2 + sample.gz_dps**2)
        for sample in samples
    ]
    gyro_axis_means = {
        "gx_dps": _mean(sample.gx_dps for sample in samples),
        "gy_dps": _mean(sample.gy_dps for sample in samples),
        "gz_dps": _mean(sample.gz_dps for sample in samples),
    }

    checks: dict[str, bool] = {
        "rate_49_to_51_hz": 49.0 <= effective_hz <= 51.0,
        "p95_interval_at_most_22_ms": percentile(positive_intervals, 0.95) <= 22.0,
        "monotonic_device_timestamps": non_monotonic == 0,
        "no_invalid_rows": invalid_rows == 0,
        "missed_slot_ratio_at_most_0_1_percent": missed_ratio <= 0.001,
    }
    sequences = [sample.sequence for sample in samples]
    sequence_gaps = None
    if all(sequence is not None for sequence in sequences):
        concrete_sequences = [int(sequence) for sequence in sequences]
        sequence_deltas = [
            current - previous
            for previous, current in zip(
                concrete_sequences, concrete_sequences[1:]
            )
        ]
        sequence_gaps = sum(max(0, delta - 1) for delta in sequence_deltas)
        checks["monotonic_sequence"] = all(delta > 0 for delta in sequence_deltas)
        checks["no_sequence_gaps"] = sequence_gaps == 0
    if stationary:
        checks.update(
            {
                "stationary_accel_mean_0_95_to_1_05_g": (
                    0.95 <= _mean(accel_magnitudes) <= 1.05
                ),
                "stationary_accel_std_at_most_0_03_g": (
                    _pstdev(accel_magnitudes) <= 0.03
                ),
                "stationary_gyro_axis_bias_at_most_1_5_dps": (
                    max(abs(value) for value in gyro_axis_means.values()) <= 1.5
                ),
            }
        )

    result = "PASS" if all(checks.values()) else "FAIL"
    return {
        "result": result,
        "samples": len(samples),
        "invalid_rows": invalid_rows,
        "device_duration_s": round(device_duration_s, 3),
        "effective_hz": round(effective_hz, 4),
        "interval_ms": {
            "mean": round(_mean(positive_intervals), 4),
            "p95": round(percentile(positive_intervals, 0.95), 4),
            "max": round(max(positive_intervals), 4),
        },
        "non_monotonic_timestamps": non_monotonic,
        "estimated_missed_slots": missed_slots,
        "sequence_gaps": sequence_gaps,
        "missed_slot_ratio": round(missed_ratio, 6),
        "accel_magnitude_g": {
            "mean": round(_mean(accel_magnitudes), 5),
            "std": round(_pstdev(accel_magnitudes), 5),
        },
        "gyro_magnitude_dps": {
            "mean": round(_mean(gyro_magnitudes), 5),
            "std": round(_pstdev(gyro_magnitudes), 5),
        },
        "gyro_axis_mean_dps": {
            name: round(value, 5) for name, value in gyro_axis_means.items()
        },
        "checks": checks,
    }


def write_capture(path: Path, samples: list[ImuSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def capture_serial(
    port: str,
    baud: int,
    duration_s: float,
    startup_timeout_s: float,
    status_stream: TextIO = sys.stderr,
) -> tuple[list[ImuSample], int]:
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "pyserial is required; install the repository requirements.txt"
        ) from error

    samples: list[ImuSample] = []
    invalid_rows = 0
    capture_deadline: float | None = None
    startup_deadline = time.monotonic() + startup_timeout_s
    print(f"Opening {port} at {baud} baud...", file=status_stream)

    with serial.Serial(port=port, baudrate=baud, timeout=1) as connection:
        while True:
            now = time.monotonic()
            if capture_deadline is not None and now >= capture_deadline:
                break
            if capture_deadline is None and now >= startup_deadline:
                print(
                    f"No valid sample within {startup_timeout_s:.0f} s.",
                    file=status_stream,
                )
                break
            raw_line = connection.readline()
            host_time_ns = time.monotonic_ns()
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("#"):
                print(line, file=status_stream)
                continue
            try:
                sample = parse_sample(line, host_time_ns)
            except (ValueError, OverflowError) as error:
                invalid_rows += 1
                print(f"Invalid row ({error}): {line!r}", file=status_stream)
                continue
            if sample is None:
                continue
            samples.append(sample)
            if capture_deadline is None:
                capture_deadline = time.monotonic() + duration_s
                print(
                    f"First valid sample received; capturing {duration_s:.0f} s...",
                    file=status_stream,
                )

    return samples, invalid_rows


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "recordings" / f"imu_step1_{timestamp}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and validate stationary FusionSense IMU output."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM16")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for calibration and the first data row.",
    )
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--moving",
        action="store_true",
        help="Skip stationary acceleration/gyro acceptance checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output or default_output_path()
    try:
        samples, invalid_rows = capture_serial(
            args.port, args.baud, args.duration, args.startup_timeout
        )
    except (OSError, RuntimeError) as error:
        print(f"Capture failed: {error}", file=sys.stderr)
        return 2

    if samples:
        write_capture(output_path, samples)
        print(f"Raw capture: {output_path}", file=sys.stderr)

    report = analyze_samples(
        samples,
        invalid_rows=invalid_rows,
        target_hz=args.target_hz,
        stationary=not args.moving,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
