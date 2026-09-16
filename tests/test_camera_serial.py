from __future__ import annotations

from pathlib import Path
import sys
import zlib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_serial import (
    CameraSerialError,
    HEADER_STRUCT,
    PACKET_MAGIC,
    PACKET_TYPE_JPEG,
    SCHEMA_VERSION,
    SerialCameraStream,
)


class FakeSerial:
    def __init__(self, incoming: bytes):
        self.incoming = bytearray(incoming)
        self.written = bytearray()
        self.closed = False

    def read(self, length: int) -> bytes:
        chunk_length = min(length, 7, len(self.incoming))
        if chunk_length == 0:
            return b""
        chunk = bytes(self.incoming[:chunk_length])
        del self.incoming[:chunk_length]
        return chunk

    def write(self, payload: bytes) -> int:
        self.written.extend(payload)
        return len(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def packet(payload: bytes, *, crc32: int | None = None) -> bytes:
    checksum = zlib.crc32(payload) & 0xFFFFFFFF if crc32 is None else crc32
    header = HEADER_STRUCT.pack(
        PACKET_MAGIC,
        SCHEMA_VERSION,
        PACKET_TYPE_JPEG,
        HEADER_STRUCT.size,
        17,
        12_345_678,
        len(payload),
        320,
        240,
        2,
        checksum,
    )
    return header + payload


class CameraSerialTests(unittest.TestCase):
    def test_serial_camera_preserves_timestamped_jpeg_metadata(self):
        jpeg = b"\xff\xd8test-jpeg\xff\xd9"
        connection = FakeSerial(
            b"# startup text\r\n"
            b"# stream_starting,binary_protocol=FSC1\r\n"
            + packet(jpeg)
        )
        stream = SerialCameraStream(
            connection=connection,
            settle_seconds=0,
            clock_ns=lambda: 998_877_665,
        )

        with stream:
            frame = stream.read()

        self.assertTrue(connection.written.startswith(b"START\n"))
        self.assertTrue(connection.written.endswith(b"STOP\n"))
        self.assertEqual(frame.sequence, 17)
        self.assertEqual(frame.device_timestamp_us, 12_345_678)
        self.assertEqual((frame.width, frame.height), (320, 240))
        self.assertEqual(frame.capture_errors, 2)
        self.assertEqual(frame.jpeg_bytes, jpeg)
        self.assertEqual(frame.host_header_received_monotonic_ns, 998_877_665)
        self.assertEqual(frame.host_received_monotonic_ns, 998_877_665)

    def test_serial_camera_rejects_corrupt_jpeg_crc(self):
        jpeg = b"\xff\xd8broken\xff\xd9"
        connection = FakeSerial(
            b"# stream_starting,binary_protocol=FSC1\r\n"
            + packet(jpeg, crc32=0)
        )
        stream = SerialCameraStream(connection=connection, settle_seconds=0)
        stream.start()

        with self.assertRaisesRegex(CameraSerialError, "CRC mismatch"):
            stream.read()


if __name__ == "__main__":
    unittest.main()
