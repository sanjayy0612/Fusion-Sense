"""Numpy-only tests for the camera-source seam (no camera/OpenCV required)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.camera_stream import (
    CameraConnectionError,
    CameraStream,
    esp32_stream_url,
    normalize_camera_source,
)


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_source_normalization():
    assert normalize_camera_source(0) == 0
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source("http://camera:81/stream") == "http://camera:81/stream"
    assert esp32_stream_url("192.168.1.42") == "http://192.168.1.42:81/stream"
    assert esp32_stream_url("http://192.168.1.42/") == "http://192.168.1.42:81/stream"
    print("PASS test_source_normalization")


def test_read_and_timestamp():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    capture = FakeCapture([image])
    stream = CameraStream(
        "http://camera:81/stream",
        capture_factory=lambda source: capture,
        clock=lambda: 12.5,
    )
    with stream:
        frame = stream.read()
    assert frame.captured_at == 12.5
    assert frame.image.shape == (4, 5, 3)
    assert capture.released
    print("PASS test_read_and_timestamp")


def test_open_failure():
    stream = CameraStream("bad", capture_factory=lambda source: FakeCapture([], False))
    try:
        stream.read()
    except CameraConnectionError:
        print("PASS test_open_failure")
        return
    raise AssertionError("expected CameraConnectionError")


if __name__ == "__main__":
    test_source_normalization()
    test_read_and_timestamp()
    test_open_failure()
    print("\nALL CAMERA STREAM TESTS PASSED")

