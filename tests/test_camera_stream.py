"""Numpy-only tests for the camera-source seam (no camera/OpenCV required)."""
import os
import sys
from io import BytesIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.camera_stream import (
    CameraConnectionError,
    CameraStream,
    TimestampedMjpegStream,
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


def test_timestamped_mjpeg_headers_are_preserved():
    jpeg = b"fake-jpeg-for-injected-decoder"
    multipart = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n".encode()
        + b"X-Frame-Sequence: 42\r\n"
        + b"X-Capture-Timestamp-Us: 9876543\r\n\r\n"
        + jpeg
        + b"\r\n"
    )

    class FakeResponse(BytesIO):
        pass

    decoded = np.ones((3, 4, 3), dtype=np.uint8)
    stream = TimestampedMjpegStream(
        "http://camera:81/stream",
        opener=lambda request, timeout: FakeResponse(multipart),
        decoder=lambda payload: decoded if payload == jpeg else None,
        clock=lambda: 12.75,
    )
    with stream:
        frame = stream.read()
    assert frame.captured_at == 12.75
    assert frame.device_timestamp_us == 9876543
    assert frame.sequence == 42
    assert frame.jpeg_bytes == jpeg
    assert frame.image.shape == (3, 4, 3)
    print("PASS test_timestamped_mjpeg_headers_are_preserved")


def test_timestamped_mjpeg_identity_headers_are_preserved():
    jpeg = b"identity-test"
    multipart = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n".encode()
        + b"X-Device-Id: cam01\r\n"
        + b"X-Session-Id: fusion_test\r\n"
        + b"X-Frame-Sequence: 7\r\n"
        + b"X-Capture-Timestamp-Us: 123000\r\n\r\n"
        + jpeg
    )
    stream = TimestampedMjpegStream(
        "http://camera:81/stream",
        opener=lambda request, timeout: BytesIO(multipart),
        decoder=lambda payload: np.zeros((2, 2, 3), dtype=np.uint8),
        clock=lambda: 1.0,
    )
    with stream:
        frame = stream.read()
    assert frame.device_id == "cam01"
    assert frame.session_id == "fusion_test"
    print("PASS test_timestamped_mjpeg_identity_headers_are_preserved")


if __name__ == "__main__":
    test_source_normalization()
    test_read_and_timestamp()
    test_open_failure()
    test_timestamped_mjpeg_headers_are_preserved()
    test_timestamped_mjpeg_identity_headers_are_preserved()
    print("\nALL CAMERA STREAM TESTS PASSED")

