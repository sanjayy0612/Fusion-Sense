"""Record the versioned FusionSense IMU stream over persistent USB serial."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_imu_stream import ImuSample, analyze_samples, parse_sample


SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
CSV_FIELDS = list(ImuSample.__dataclass_fields__)


def default_session_id() -> str:
    return datetime.now(timezone.utc).strftime("imu_%Y%m%dT%H%M%SZ")


def default_output_directory(session_id: str) -> Path:
    return Path("data") / "recordings" / session_id


def validate_session_id(value: str) -> str:
    if not SESSION_PATTERN.fullmatch(value):
        raise ValueError(
            "session ID must contain 1-32 letters, digits, dots, underscores, "
            "or hyphens"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 USB port, e.g. COM16")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stationary",
        action="store_true",
        help="Also enforce stationary 1 g and gyro-bias checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    try:
        session_id = validate_session_id(args.session_id or default_session_id())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_directory = args.output or default_output_directory(session_id)

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        print("pyserial is missing; install requirements.txt", file=sys.stderr)
        return 2

    samples: list[ImuSample] = []
    invalid_rows = 0
    status_rows: list[tuple[int, str]] = []
    session_acknowledged = False
    startup_deadline = time.monotonic() + args.startup_timeout
    last_session_request = 0.0

    print(f"Opening persistent IMU serial stream: {args.port} @ {args.baud}")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as connection:
            while not session_acknowledged and time.monotonic() < startup_deadline:
                now = time.monotonic()
                if now - last_session_request >= 2.0:
                    connection.write(f"SESSION,{session_id}\n".encode("ascii"))
                    connection.flush()
                    last_session_request = now

                raw_line = connection.readline()
                host_received_ns = time.monotonic_ns()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("#"):
                    status_rows.append((host_received_ns, line))
                    print(line)
                    if line == f"# session_set,id={session_id}":
                        session_acknowledged = True

            if not session_acknowledged:
                print(
                    "IMU did not acknowledge the session. Upload the versioned "
                    "timestamped firmware and keep it stationary during calibration.",
                    file=sys.stderr,
                )
                return 2

            output_directory.mkdir(parents=True, exist_ok=False)
            imu_path = output_directory / "imu.csv"
            status_path = output_directory / "device_status.csv"
            deadline = time.monotonic() + args.duration
            print(
                f"Session {session_id} acknowledged; capturing "
                f"{args.duration:.0f} s..."
            )

            with imu_path.open("w", newline="", encoding="utf-8") as imu_stream:
                writer = csv.DictWriter(imu_stream, fieldnames=CSV_FIELDS)
                writer.writeheader()

                while time.monotonic() < deadline:
                    raw_line = connection.readline()
                    host_received_ns = time.monotonic_ns()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        status_rows.append((host_received_ns, line))
                        print(line)
                        continue
                    try:
                        sample = parse_sample(line, host_received_ns)
                    except (ValueError, OverflowError) as error:
                        invalid_rows += 1
                        print(f"Invalid IMU row ({error}): {line!r}", file=sys.stderr)
                        continue
                    if sample is None:
                        continue
                    if sample.schema_version != 1:
                        print(
                            "Legacy IMU row received; upload the versioned firmware.",
                            file=sys.stderr,
                        )
                        return 2
                    if sample.session_id != session_id:
                        invalid_rows += 1
                        continue
                    samples.append(sample)
                    writer.writerow(asdict(sample))

            with status_path.open("w", newline="", encoding="utf-8") as status_stream:
                writer = csv.writer(status_stream)
                writer.writerow(["host_received_monotonic_ns", "message"])
                writer.writerows(status_rows)
    except (OSError, serial.SerialException) as error:
        print(f"IMU recording failed: {error}", file=sys.stderr)
        return 2

    validation = analyze_samples(
        samples,
        invalid_rows=invalid_rows,
        target_hz=50.0,
        stationary=args.stationary,
    )
    session = {
        "schema_version": 1,
        "device_id": samples[0].device_id if samples else None,
        "session_id": session_id,
        "port": args.port,
        "baud": args.baud,
        "requested_duration_s": args.duration,
        "samples_saved": len(samples),
        "imu_manifest": "imu.csv",
        "device_status": "device_status.csv",
        "units": {
            "acceleration": "g",
            "angular_velocity": "degrees_per_second",
            "device_timestamp": "microseconds_since_esp32_boot",
            "host_received_timestamp": "monotonic_nanoseconds",
        },
        "validation": validation,
    }
    (output_directory / "session.json").write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved recording: {output_directory.resolve()}")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

