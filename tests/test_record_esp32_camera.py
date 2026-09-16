import tempfile
import unittest
from pathlib import Path

from scripts.record_esp32_camera import save_original_jpeg


class CameraRecordingTests(unittest.TestCase):
    def test_original_jpeg_is_saved_without_reencoding(self) -> None:
        payload = b"\xff\xd8payload\xff\xd9"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "frame.jpg"
            save_original_jpeg(destination, payload)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".jpg.part").exists())

    def test_incomplete_jpeg_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "frame.jpg"
            with self.assertRaises(ValueError):
                save_original_jpeg(destination, b"not-a-jpeg")
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

