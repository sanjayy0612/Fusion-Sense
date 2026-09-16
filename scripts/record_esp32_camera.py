"""Record original timestamped ESP32-CAM JPEGs and a laptop-side manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_stream import (
    CameraConnectionError,
    TimestampedMjpegStream,
    esp32_stream_url,
)
from scripts.validate_esp32_camera import (
    FrameObservation,
    analyze_observations,
    read_health,
)


MANIFEST_FIELDS = [
    "sequence",
    "device_timestamp_us",
    "host_received_monotonic_ns",
    "width",
    "height",
    "jpeg_bytes",
    "relative_path",
]


def default_output_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "recordings" / f"camera_{timestamp}"


def save_original_jpeg(path: Path, payload: bytes) -> None:
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError("camera payload is not a complete JPEG")
    temporary = path.with_suffix(".jpg.part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="ESP32-CAM IP or hostname")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    output_directory = args.output or default_output_directory()
    frame_directory = output_directory / "frames"
    source = esp32_stream_url(args.host)
    observations: list[FrameObservation] = []
    session_started_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()

    print(f"Opening persistent timestamped stream: {source}")
    try:
        with TimestampedMjpegStream(source) as camera:
            first_frame = camera.read()
            output_directory.mkdir(parents=True, exist_ok=False)
            frame_directory.mkdir()
            manifest_path = output_directory / "camera_manifest.csv"

            with manifest_path.open("w", newline="", encoding="utf-8") as stream:
                manifest = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                manifest.writeheader()
                deadline = time.monotonic() + args.duration
                frame = first_frame

                while True:
                    if (
                        frame.sequence is None
                        or frame.device_timestamp_us is None
                        or frame.jpeg_bytes is None
                    ):
                        raise CameraConnectionError(
                            "timestamped JPEG metadata/payload unexpectedly missing"
                        )

                    height, width = frame.image.shape[:2]
                    relative_path = Path("frames") / f"{frame.sequence:010d}.jpg"
                    save_original_jpeg(
                        output_directory / relative_path, frame.jpeg_bytes
                    )
                    host_received_ns = int(round(frame.captured_at * 1_000_000_000))
                    manifest.writerow(
                        {
                            "sequence": frame.sequence,
                            "device_timestamp_us": frame.device_timestamp_us,
                            "host_received_monotonic_ns": host_received_ns,
                            "width": width,
                            "height": height,
                            "jpeg_bytes": len(frame.jpeg_bytes),
                            "relative_path": relative_path.as_posix(),
                        }
                    )
                    stream.flush()
                    observations.append(
                        FrameObservation(
                            sequence=frame.sequence,
                            device_timestamp_us=frame.device_timestamp_us,
                            host_received_s=frame.captured_at,
                            width=width,
                            height=height,
                        )
                    )

                    if time.monotonic() >= deadline:
                        break
                    frame = camera.read()
    except (CameraConnectionError, OSError, ValueError) as error:
        print(f"Camera recording failed: {error}", file=sys.stderr)
        return 2

    elapsed_s = time.monotonic() - started_monotonic
    health = read_health(args.host)
    validation = analyze_observations(
        observations,
        requested_duration_s=args.duration,
        target_fps=args.target_fps,
        health=health,
    )
    session = {
        "schema_version": 1,
        "source": source,
        "session_started_utc": session_started_utc,
        "requested_duration_s": args.duration,
        "host_elapsed_s": round(elapsed_s, 4),
        "frames_saved": len(observations),
        "manifest": "camera_manifest.csv",
        "frames_directory": "frames",
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

