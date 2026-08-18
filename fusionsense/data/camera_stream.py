"""Live camera-source seam for FusionSense.

The primary V1 adapter is an ESP32-CAM MJPEG URL such as
``http://192.168.1.42:81/stream``. Integer sources remain supported so the
laptop webcam can be used as a fallback during development.

OpenCV is imported lazily so the numpy-only pipeline tests stay lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import BinaryIO, Callable, Union
from urllib.request import Request, urlopen

import numpy as np

CameraSource = Union[int, str]


class CameraConnectionError(RuntimeError):
    """The configured camera source could not be opened or read."""


@dataclass(frozen=True)
class CameraFrame:
    """A decoded BGR frame with host and optional device capture timestamps."""

    captured_at: float
    image: np.ndarray
    device_timestamp_us: int | None = None
    sequence: int | None = None
    jpeg_bytes: bytes | None = None
    device_id: str | None = None
    session_id: str | None = None


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


class TimestampedMjpegStream:
    """Read FusionSense multipart JPEG frames and preserve ESP32 metadata.

    ``captured_at`` remains the laptop monotonic receive time for compatibility.
    ``device_timestamp_us`` is the camera-driver capture time on the ESP32 clock;
    the clock-synchronization phase maps it into the laptop clock domain.
    """

    def __init__(
        self,
        source: str,
        *,
        timeout: float = 10.0,
        opener: Callable[..., BinaryIO] = urlopen,
        decoder: Callable[[bytes], np.ndarray] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        normalized = normalize_camera_source(source)
        if not isinstance(normalized, str) or not normalized.startswith(
            ("http://", "https://")
        ):
            raise ValueError("timestamped MJPEG source must be an HTTP(S) URL")
        self.source = normalized
        self.timeout = timeout
        self._opener = opener
        self._decoder = decoder
        self._clock = clock
        self._response: BinaryIO | None = None

    def _open(self) -> None:
        self.close()
        request = Request(self.source, headers={"Cache-Control": "no-cache"})
        try:
            self._response = self._opener(request, timeout=self.timeout)
        except Exception as exc:
            raise CameraConnectionError(
                f"Could not open timestamped camera stream {self.source!r}"
            ) from exc

    def _decode(self, payload: bytes) -> np.ndarray:
        if self._decoder is not None:
            return self._decoder(payload)
        try:
            import cv2
        except Exception as exc:
            raise ImportError(
                "Timestamped JPEG decoding needs OpenCV. Install: "
                "python -m pip install opencv-python"
            ) from exc
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CameraConnectionError("ESP32-CAM returned an invalid JPEG")
        return image

    def _read_exact(self, length: int) -> bytes:
        assert self._response is not None
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self._response.read(length - len(chunks))
            if not chunk:
                raise EOFError("camera stream ended inside a JPEG payload")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_frame(self) -> CameraFrame:
        assert self._response is not None
        while True:
            line = self._response.readline()
            if not line:
                raise EOFError("camera stream ended before multipart boundary")
            if line.strip().startswith(b"--"):
                break

        headers: dict[str, str] = {}
        while True:
            line = self._response.readline()
            if not line:
                raise EOFError("camera stream ended inside multipart headers")
            if line in (b"\r\n", b"\n"):
                break
            name, separator, value = line.decode("ascii", errors="replace").partition(":")
            if separator:
                headers[name.strip().lower()] = value.strip()

        try:
            content_length = int(headers["content-length"])
            sequence = int(headers["x-frame-sequence"])
            device_timestamp_us = int(headers["x-capture-timestamp-us"])
        except (KeyError, ValueError) as exc:
            raise CameraConnectionError(
                "MJPEG part is missing valid FusionSense timestamp headers"
            ) from exc
        if content_length <= 0 or content_length > 5_000_000:
            raise CameraConnectionError(
                f"invalid MJPEG Content-Length: {content_length}"
            )

        payload = self._read_exact(content_length)
        received_at = float(self._clock())
        image = self._decode(payload)
        return CameraFrame(
            captured_at=received_at,
            image=image,
            device_timestamp_us=device_timestamp_us,
            sequence=sequence,
            jpeg_bytes=payload,
            device_id=headers.get("x-device-id") or None,
            session_id=headers.get("x-session-id") or None,
        )

    def read(self) -> CameraFrame:
        for attempt in range(2):
            try:
                if self._response is None:
                    self._open()
                return self._read_frame()
            except (CameraConnectionError, EOFError, OSError):
                self.close()
                if attempt == 1:
                    raise CameraConnectionError(
                        f"Timestamped stream {self.source!r} returned no valid frame"
                    )
        raise AssertionError("unreachable")

    def close(self) -> None:
        response, self._response = self._response, None
        if response is not None:
            response.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

