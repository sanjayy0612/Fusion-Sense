"""Live camera-source seam for FusionSense.

The primary V1 adapter is an ESP32-CAM MJPEG URL such as
``http://192.168.1.42:81/stream``. Integer sources remain supported so the
laptop webcam can be used as a fallback during development.

OpenCV is imported lazily so the numpy-only pipeline tests stay lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Union

import numpy as np

CameraSource = Union[int, str]


class CameraConnectionError(RuntimeError):
    """The configured camera source could not be opened or read."""


@dataclass(frozen=True)
class CameraFrame:
    """A decoded BGR frame timestamped on the laptop's monotonic clock."""

    captured_at: float
    image: np.ndarray


def normalize_camera_source(source: CameraSource) -> CameraSource:
    """Convert numeric CLI values to OpenCV device indexes; preserve URLs."""
    if isinstance(source, int):
        return source
    value = str(source).strip()
    if not value:
        raise ValueError("camera source must not be empty")
    if value.isdecimal():
        return int(value)
    return value


def esp32_stream_url(host: str) -> str:
    """Build the standard CameraWebServer MJPEG endpoint from a host/IP."""
    value = host.strip().rstrip("/")
    if not value:
        raise ValueError("ESP32-CAM host must not be empty")
    if value.startswith(("http://", "https://")):
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split(":", 1)[0]
    return f"http://{value}:81/stream"


class CameraStream:
    """Open, timestamp, and reconnect a local or network OpenCV source.

    The small interface is ``read()`` plus context-manager cleanup. A failed
    read triggers one reconnect attempt before ``CameraConnectionError`` is
    raised. ``capture_factory`` and ``clock`` are dependency seams for tests.
    """

    def __init__(
        self,
        source: CameraSource,
        *,
        capture_factory: Callable[[CameraSource], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.source = normalize_camera_source(source)
        self._capture_factory = capture_factory
        self._clock = clock
        self._capture = None

    def _factory(self):
        if self._capture_factory is not None:
            return self._capture_factory
        try:
            import cv2
        except Exception as exc:
            raise ImportError(
                "Live camera capture needs OpenCV. Install: "
                "python -m pip install opencv-python"
            ) from exc
        return cv2.VideoCapture

    def _open(self):
        self.close()
        capture = self._factory()(self.source)
        if hasattr(capture, "set"):
            try:
                import cv2
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if not capture.isOpened():
            capture.release()
            raise CameraConnectionError(
                f"Could not open camera source {self.source!r}. For ESP32-CAM, "
                "verify the browser stream at http://<ip>:81/stream first."
            )
        self._capture = capture

    def read(self) -> CameraFrame:
        if self._capture is None:
            self._open()
        ok, image = self._capture.read()
        if not ok or image is None:
            self._open()
            ok, image = self._capture.read()
        if not ok or image is None:
            raise CameraConnectionError(
                f"Camera source {self.source!r} opened but returned no frame"
            )
        return CameraFrame(captured_at=float(self._clock()), image=image)

    def close(self):
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

