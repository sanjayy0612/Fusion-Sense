"""Collect one labelled, synchronized camera + ESP32-IMU trial.

The output is directly consumable by ``paired_loader``::

    data/raw/up_fall/<subject>/<activity>/<trial>/
        imu.csv       # t_ms, ax, ay, az, gx, gy, gz, ...
        video.mp4
        metadata.json

Run one activity per trial. Keep the subject and activity still for the first
second, perform the activity, and remain still for the last second. The
activity name is the label, so use one of walking, standing, sitting, lying,
or falling.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import DATA_ROOT
from fusionsense.data.camera_stream import CameraStream, esp32_stream_url


ACTIVITIES = {"walking", "standing", "sitting", "lying", "falling"}


def _read_imu(port, baud, ready, recording, stop, rows, errors):
    try:
        import serial
    except ImportError as exc:
        errors.append("pyserial is missing; install requirements.txt")
        return
    try:
        with serial.Serial(port, baudrate=baud, timeout=0.2) as device:
            # Opening an ESP32 serial port can reset the board. Give it time to
            # boot before declaring the IMU side ready for synchronized capture.
            time.sleep(1.0)
            device.reset_input_buffer()
            ready.set()
            while not stop.is_set():
                line = device.readline().decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                fields = [item.strip() for item in line.split(",")]
                if len(fields) < 7:
                    continue
                try:
                    # Preserve the ESP32 timestamp and the six sensor channels.
                    values = [float(item) for item in fields[:7]]
                except ValueError:
                    continue
                if recording.is_set():
                    rows.append(values)
    except Exception as exc:  # report on the main thread after the trial
        errors.append(f"IMU serial error: {exc}")
        ready.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 USB serial port")
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--camera-host", help="ESP32-CAM IP or URL")
    camera.add_argument(
        "--camera-source",
        help="OpenCV camera index or URL; use 0 for the laptop webcam",
    )
    parser.add_argument("--subject", required=True, help="subject id, e.g. s01")
    parser.add_argument("--activity", required=True, choices=sorted(ACTIVITIES))
    parser.add_argument("--trial", required=True, help="trial id, e.g. 01")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    if args.seconds < 4:
        parser.error("--seconds must be at least 4 (the model window is 2 seconds)")

    out = DATA_ROOT / "up_fall" / args.subject / args.activity / args.trial
    out.mkdir(parents=True, exist_ok=False)
    video_path = out / "video.mp4"
    imu_path = out / "imu.csv"

    rows = []
    errors = []
    ready = threading.Event()
    recording = threading.Event()
    stop = threading.Event()
    imu_thread = threading.Thread(
        target=_read_imu,
        args=(args.port, args.baud, ready, recording, stop, rows, errors),
        daemon=True,
    )

    camera_source = args.camera_source
    if args.camera_host:
        camera_source = args.camera_host
        if not camera_source.startswith(("http://", "https://")):
            camera_source = esp32_stream_url(camera_source)

    print(f"Recording {args.activity} -> {out}")
    print("Starting in 3 seconds. Keep still for 1s, perform the activity, then stop still.")
    time.sleep(3)
    imu_thread.start()
    if not ready.wait(timeout=5.0) or errors:
        stop.set()
        imu_thread.join(timeout=2.0)
        raise RuntimeError("Could not prepare ESP32 IMU serial: " + " | ".join(errors))

    writer = None
    frame_count = 0
    start = None
    elapsed = 0.0
    try:
        import cv2
        with CameraStream(camera_source) as camera:
            # Both adapters are now open. This moment is the shared zero point
            # for the paired trial.
            start = time.monotonic()
            recording.set()
            while time.monotonic() - start < args.seconds:
                frame = camera.read()
                if writer is None:
                    height, width = frame.image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create {video_path}")
                writer.write(frame.image)
                frame_count += 1
    finally:
        recording.clear()
        if start is not None:
            elapsed = time.monotonic() - start
        stop.set()
        imu_thread.join(timeout=2.0)
        if writer is not None:
            writer.release()

    with imu_path.open("w", newline="") as handle:
        output = csv.writer(handle)
        output.writerow(["t_ms", "ax", "ay", "az", "gx", "gy", "gz"])
        output.writerows(rows)
    metadata = {
        "subject": args.subject,
        "activity": args.activity,
        "trial": args.trial,
        "duration_seconds": elapsed,
        "video_fps": frame_count / max(elapsed, 1e-6),
        "video_frames": frame_count,
        "imu_samples": len(rows),
        "camera_source": camera_source,
        "created_at_unix": time.time(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if errors:
        print("WARNING: " + " | ".join(errors), file=sys.stderr)
    print(f"Saved {frame_count} video frames and {len(rows)} IMU samples")
    if len(rows) < args.seconds * 25:
        print("WARNING: fewer than half-rate IMU samples; inspect the serial connection", file=sys.stderr)


if __name__ == "__main__":
    main()
