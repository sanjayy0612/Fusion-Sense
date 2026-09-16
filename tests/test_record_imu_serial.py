import unittest

from scripts.record_imu_serial import validate_session_id


class ImuRecordingTests(unittest.TestCase):
    def test_valid_session_id(self) -> None:
        self.assertEqual(validate_session_id("imu_20260817T180000Z"), "imu_20260817T180000Z")

    def test_session_id_rejects_csv_delimiters(self) -> None:
        with self.assertRaises(ValueError):
            validate_session_id("bad,session")

    def test_session_id_rejects_long_value(self) -> None:
        with self.assertRaises(ValueError):
            validate_session_id("x" * 33)


if __name__ == "__main__":
    unittest.main()

