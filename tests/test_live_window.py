import unittest

import numpy as np

from fusionsense.config import CFG
from fusionsense.live import LiveImuPoint, LivePosePoint, assemble_live_window


class LiveWindowTests(unittest.TestCase):
    def build_points(self, *, pose_valid=True):
        base = 10_000_000_000
        imu = [
            LiveImuPoint(
                capture_ns=base + index * 20_000_000,
                values_g_dps=np.array([0, 0, 1, 0, 0, 0], np.float32),
                sequence=index,
            )
            for index in range(CFG.t_imu)
        ]
        pose = [
            LivePosePoint(
                capture_ns=base + index * 100_000_000,
                landmarks=np.full(CFG.vision_dv, 0.25, np.float32),
                valid=pose_valid,
                image_quality=0.8,
                sequence=index,
            )
            for index in range(CFG.t_vis)
        ]
        return base, imu, pose

    def test_exact_live_window_is_canonical_and_bimodal(self):
        base, imu, pose = self.build_points()
        window, diagnostics = assemble_live_window(
            imu,
            pose,
            window_end_ns=base + 2_000_000_000,
            session_id="test",
        )
        self.assertEqual(window.imu.shape, (100, 6))
        self.assertEqual(window.vision.shape, (20, 99))
        self.assertTrue(window.imu_valid)
        self.assertTrue(window.vision_valid)
        self.assertAlmostEqual(float(np.median(window.imu[:, 2])), 9.80665, places=4)
        self.assertEqual(diagnostics["imu"]["sequence_gaps"], 0)
        self.assertEqual(diagnostics["camera"]["sequence_gaps"], 0)

    def test_pose_loss_marks_window_degraded(self):
        base, imu, pose = self.build_points(pose_valid=False)
        window, diagnostics = assemble_live_window(
            imu,
            pose,
            window_end_ns=base + 2_000_000_000,
            session_id="test",
        )
        self.assertTrue(window.imu_valid)
        self.assertFalse(window.vision_valid)
        self.assertEqual(diagnostics["camera"]["pose_valid_ratio"], 0.0)

    def test_duplicate_host_timestamp_is_coalesced(self):
        base, imu, pose = self.build_points()
        imu.append(imu[-1])
        window, diagnostics = assemble_live_window(
            imu,
            pose,
            window_end_ns=base + 2_000_000_000,
            session_id="test",
        )
        self.assertTrue(window.imu_valid)
        self.assertEqual(diagnostics["imu"]["duplicate_timestamps_discarded"], 1)


if __name__ == "__main__":
    unittest.main()
