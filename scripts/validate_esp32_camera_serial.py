"""Validate timestamped ESP32-CAM JPEG capture over USB serial."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_serial import (  # noqa: E402
    CameraSerialError,
    SERIAL_CAMERA_BAUD,
    SerialCameraStream,
)
from scripts.validate_esp32_camera import (  # noqa: E402
    FrameObservation,
    analyze_observations,
)


def decode_jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise CameraSerialError(
            "USB camera validation needs OpenCV. Install requirements.txt"
        ) from exc
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CameraSerialError("OpenCV could not decode the camera JPEG")
    height, width = image.shape[:2]
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32-CAM-MB port, e.g. COM10")
    parser.add_argument("--baud", type=int, default=SERIAL_CAMERA_BAUD)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.startup_timeout <= 0:
        raise SystemExit("--startup-timeout must be positive")

    observations: list[FrameObservation] = []
    jpeg_sizes: list[int] = []
    last_capture_errors = 0
    print(f"Opening USB camera: {args.port} @ {args.baud} baud")
    try:
        with SerialCameraStream(
            args.port,
            baud=args.baud,
            frame_timeout=args.startup_timeout,
        ) as camera:
            first_frame = camera.read()
            started = time.monotonic()
            deadline = started + args.duration
            frame = first_frame
            while True:
                decoded_width, decoded_height = decode_jpeg_dimensions(
                    frame.jpeg_bytes
                )
                if (decoded_width, decoded_height) != (frame.width, frame.height):
                    raise CameraSerialError(
                        "JPEG dimensions disagree with packet header: "
                        f"decoded={decoded_width}x{decoded_height}, "
                        f"header={frame.width}x{frame.height}"
                    )
                observations.append(
                    FrameObservation(
                        sequence=frame.sequence,
                        device_timestamp_us=frame.device_timestamp_us,
                        host_received_s=frame.host_received_monotonic_ns / 1e9,
                        width=frame.width,
                        height=frame.height,
                    )
                )
                jpeg_sizes.append(len(frame.jpeg_bytes))
                last_capture_errors = frame.capture_errors
                if time.monotonic() >= deadline:
                    break
                frame = camera.read()
    except (CameraSerialError, OSError) as error:
        print(
            json.dumps(
                {
                    "result": "FAIL",
                    "transport": "usb_serial",
                    "port": args.port,
                    "baud": args.baud,
                    "frames": len(observations),
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    validation = analyze_observations(
        observations,
        requested_duration_s=args.duration,
        target_fps=args.target_fps,
        health={"capture_errors": last_capture_errors},
    )
    validation.update(
        {
            "transport": "usb_serial",
            "port": args.port,
            "baud": args.baud,
            "crc_errors": 0,
            "jpeg_bytes": {
                "mean": round(statistics.fmean(jpeg_sizes), 2),
                "min": min(jpeg_sizes),
                "max": max(jpeg_sizes),
            },
        }
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
