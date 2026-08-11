"""Live ESP32-CAM -> OpenCV -> MediaPipe Pose unit/integration test.

Example:
  python scripts/test_esp32_camera.py --host 192.168.1.42
  python scripts/test_esp32_camera.py --source http://192.168.1.42:81/stream
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.data.camera_stream import CameraStream, esp32_stream_url
from fusionsense.data.vision_extractor import PoseExtractor


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--host", help="ESP32-CAM IP/hostname; builds port-81 stream URL")
    group.add_argument("--source", help="complete MJPEG URL or OpenCV camera index")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    source = esp32_stream_url(args.host) if args.host else args.source
    pose_buffer = deque(maxlen=CFG.t_vis)
    valid_buffer = deque(maxlen=CFG.t_vis)
    started = time.monotonic()
    processed = 0

    import cv2

    with CameraStream(source) as camera, PoseExtractor() as extractor:
        print(f"Connected to camera: {source}")
        while time.monotonic() - started < args.seconds:
            frame = camera.read()
            pose = extractor.extract(frame.image, frame.captured_at)
            pose_buffer.append(pose.landmarks)
            valid_buffer.append(pose.valid)
            processed += 1

            if not args.no_preview:
                preview = frame.image.copy()
                if pose.valid:
                    height, width = preview.shape[:2]
                    for x, y, _ in pose.landmarks.reshape(33, 3):
                        cv2.circle(preview, (int(x * width), int(y * height)), 2, (0, 255, 0), -1)
                cv2.putText(
                    preview,
                    f"pose={pose.valid} quality={pose.image_quality:.2f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if pose.valid else (0, 0, 255),
                    2,
                )
                cv2.imshow("FusionSense ESP32-CAM test (Q to quit)", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    cv2.destroyAllWindows()
    elapsed = max(time.monotonic() - started, 1e-6)
    valid_ratio = float(np.mean(valid_buffer)) if valid_buffer else 0.0
    print(f"frames={processed} processing_fps={processed / elapsed:.1f}")
    print(f"recent_pose_valid_ratio={valid_ratio:.2f}")
    print(f"vision_window_ready={len(pose_buffer) == CFG.t_vis}")
    if len(pose_buffer) == CFG.t_vis:
        print(f"vision_window_shape={np.stack(pose_buffer).shape}")
    if processed == 0 or valid_ratio == 0.0:
        raise SystemExit("FAIL: no valid body pose detected")
    print("PASS: ESP32-CAM -> laptop -> MediaPipe Pose")


if __name__ == "__main__":
    main()

