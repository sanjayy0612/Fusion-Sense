"""Timestamped ESP32-CAM JPEG transport over a framed USB serial stream."""
from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import BinaryIO, Callable
import zlib


SERIAL_CAMERA_BAUD = 921_600
PACKET_MAGIC = b"FSC1"
SCHEMA_VERSION = 1
PACKET_TYPE_JPEG = 1
HEADER_STRUCT = struct.Struct("<4sBBHIQIHHII")
MAX_JPEG_BYTES = 500_000


class CameraSerialError(RuntimeError):
    """The serial camera stream was unavailable or contained an invalid packet."""


@dataclass(frozen=True)
class SerialCameraFrame:
    sequence: int
    device_timestamp_us: int
    width: int
    height: int
    capture_errors: int
    jpeg_bytes: bytes
    jpeg_crc32: int
    host_header_received_monotonic_ns: int
    host_received_monotonic_ns: int


class SerialCameraStream:
    """Read CRC-protected timestamped JPEG packets from an ESP32-CAM COM port."""

    def __init__(
        self,
        port: str | None = None,
        *,
        baud: int = SERIAL_CAMERA_BAUD,
        frame_timeout: float = 5.0,
        settle_seconds: float = 2.0,
        connection: BinaryIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if connection is None and not port:
            raise ValueError("camera serial port must not be empty")
        if baud <= 0:
            raise ValueError("camera serial baud must be positive")
        if frame_timeout <= 0:
            raise ValueError("frame timeout must be positive")
        self.port = port
        self.baud = int(baud)
        self.frame_timeout = float(frame_timeout)
        self.settle_seconds = max(0.0, float(settle_seconds))
        self._connection = connection
        self._owns_connection = connection is None
        self._clock = clock
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._streaming = False

    def open(self) -> None:
        if self._connection is not None:
            return
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CameraSerialError(
                "USB camera capture needs pyserial. Install requirements.txt"
            ) from exc
        try:
            connection = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.1,
                write_timeout=2.0,
                rtscts=False,
                dsrdtr=False,
            )
            connection.dtr = False
            connection.rts = False
        except (OSError, serial.SerialException) as exc:
            raise CameraSerialError(
                f"Could not open camera serial port {self.port!r} at {self.baud} baud"
            ) from exc
        self._connection = connection
        self._sleep(self.settle_seconds)

    def start(self) -> None:
        self.open()
        assert self._connection is not None
        if hasattr(self._connection, "reset_input_buffer"):
            self._connection.reset_input_buffer()  # type: ignore[attr-defined]
        try:
            written = self._connection.write(b"START\n")
            if written != 6:
                raise CameraSerialError("camera START command was only partly written")
            self._connection.flush()
        except OSError as exc:
            raise CameraSerialError("could not start the camera serial stream") from exc
        self._streaming = True
        self._wait_for_start_ack()

    def _wait_for_start_ack(self) -> None:
        """Consume readable startup lines before the first binary packet.

        The acknowledgement itself contains the text ``FSC1``. Waiting for its
        newline prevents that text from being mistaken for packet magic.
        """
        deadline = self._clock() + self.frame_timeout
        line = bytearray()
        while self._clock() < deadline:
            character = self._read_exact(1, deadline)
            if character == b"\n":
                message = line.decode("ascii", errors="replace").strip()
                if message.startswith("# stream_starting"):
                    return
                line.clear()
                continue
            if character != b"\r":
                line.extend(character)
            if len(line) > 512:
                line.clear()
        raise CameraSerialError("camera did not acknowledge the START command")

    def stop(self) -> None:
        if self._connection is None or not self._streaming:
            return
        try:
            self._connection.write(b"STOP\n")
            self._connection.flush()
        except OSError:
            pass
        self._streaming = False

    def _read_exact(self, length: int, deadline: float) -> bytes:
        assert self._connection is not None
        chunks = bytearray()
        while len(chunks) < length:
            if self._clock() >= deadline:
                raise CameraSerialError(
                    f"camera serial packet timed out after {len(chunks)}/{length} bytes"
                )
            try:
                chunk = self._connection.read(length - len(chunks))
            except OSError as exc:
                raise CameraSerialError("camera serial read failed") from exc
            if chunk:
                chunks.extend(chunk)
        return bytes(chunks)

    def _seek_magic(self, deadline: float) -> None:
        window = bytearray()
        while True:
            window.extend(self._read_exact(1, deadline))
            if len(window) > len(PACKET_MAGIC):
                del window[0]
            if bytes(window) == PACKET_MAGIC:
                return

    def read(self) -> SerialCameraFrame:
        if self._connection is None:
            raise CameraSerialError("camera serial stream has not been opened")
        if not self._streaming:
            raise CameraSerialError("camera serial stream has not been started")

        deadline = self._clock() + self.frame_timeout
        self._seek_magic(deadline)
        remainder = self._read_exact(HEADER_STRUCT.size - len(PACKET_MAGIC), deadline)
        fields = HEADER_STRUCT.unpack(PACKET_MAGIC + remainder)
        (
            magic,
            schema_version,
            packet_type,
            header_bytes,
            sequence,
            capture_timestamp_us,
            jpeg_bytes,
            width,
            height,
            capture_errors,
            expected_crc32,
        ) = fields

        if magic != PACKET_MAGIC:
            raise CameraSerialError("camera packet magic is invalid")
        if schema_version != SCHEMA_VERSION:
            raise CameraSerialError(
                f"unsupported camera serial schema version {schema_version}"
            )
        if packet_type != PACKET_TYPE_JPEG:
            raise CameraSerialError(f"unsupported camera packet type {packet_type}")
        if header_bytes != HEADER_STRUCT.size:
            raise CameraSerialError(f"invalid camera header size {header_bytes}")
        if jpeg_bytes <= 0 or jpeg_bytes > MAX_JPEG_BYTES:
            raise CameraSerialError(f"invalid camera JPEG length {jpeg_bytes}")
        if width <= 0 or height <= 0:
            raise CameraSerialError(f"invalid camera dimensions {width}x{height}")

        # Capture this before reading the variable-length JPEG payload.  The
        # fixed-size header arrival is a substantially less size-dependent
        # clock observation than the time at which the full JPEG is received.
        header_received_ns = int(self._clock_ns())
        payload = self._read_exact(jpeg_bytes, deadline)
        received_ns = int(self._clock_ns())
        actual_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc32 != expected_crc32:
            raise CameraSerialError(
                "camera JPEG CRC mismatch: "
                f"expected 0x{expected_crc32:08x}, got 0x{actual_crc32:08x}"
            )
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise CameraSerialError("camera payload is not a complete JPEG")

        return SerialCameraFrame(
            sequence=sequence,
            device_timestamp_us=capture_timestamp_us,
            width=width,
            height=height,
            capture_errors=capture_errors,
            jpeg_bytes=payload,
            jpeg_crc32=expected_crc32,
            host_header_received_monotonic_ns=header_received_ns,
            host_received_monotonic_ns=received_ns,
        )

    def close(self) -> None:
        self.stop()
        connection, self._connection = self._connection, None
        if connection is not None and self._owns_connection:
            connection.close()

    def __enter__(self) -> "SerialCameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
