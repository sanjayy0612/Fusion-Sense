import math
import unittest

from fusionsense.data.motion_sync import MotionPoint, motion_alignment_report


def activity(time_s: float) -> float:
    return sum(
        math.exp(-((time_s - center) ** 2) / (2 * 0.12**2))
        for center in (15.0, 30.0, 45.0)
    )


class MotionSyncTests(unittest.TestCase):
    def test_shared_motion_recovers_clock_lag(self) -> None:
        imu = [
            MotionPoint(int(index * 0.02 * 1e9), activity(index * 0.02))
            for index in range(3_000)
        ]
        camera = [
            MotionPoint(
                int((index * 0.1 + 0.03) * 1e9),
                activity(index * 0.1),
            )
            for index in range(600)
        ]
        report = motion_alignment_report(imu, camera)
        self.assertEqual(report["result"], "PASS")
        self.assertLessEqual(abs(report["best_camera_shift_ms"] + 30), 5)
        self.assertGreater(report["best_correlation"], 0.8)

    def test_stationary_signals_do_not_claim_alignment(self) -> None:
        imu = [MotionPoint(index * 20_000_000, 0.1) for index in range(500)]
        camera = [MotionPoint(index * 100_000_000, 0.01) for index in range(100)]
        report = motion_alignment_report(imu, camera)
        self.assertEqual(report["result"], "FAIL")
        self.assertIn("motion", report["reason"])


if __name__ == "__main__":
    unittest.main()
