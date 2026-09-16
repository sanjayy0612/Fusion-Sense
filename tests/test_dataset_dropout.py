import unittest
from unittest.mock import patch

import numpy as np

from fusionsense.config import CFG
from fusionsense.contract import FusionWindow
from fusionsense.data.dataset import FusionDataset


class ModalityDropoutIsolationTests(unittest.TestCase):
    def test_dropout_does_not_mutate_source_window(self):
        window = FusionWindow(
            t_start=0.0,
            imu=np.ones((CFG.t_imu, CFG.imu_ch), dtype=np.float32),
            radar=np.zeros((CFG.t_radar, CFG.radar_k), dtype=np.float32),
            vision=np.full((CFG.t_vis, CFG.vision_dv), 2.0, dtype=np.float32),
            imu_valid=True,
            radar_valid=False,
            vision_valid=True,
            label=0,
        )
        original_imu = window.imu.copy()
        original_vision = window.vision.copy()

        # Drop the IMU on the first draw and keep the camera on the second.
        with patch("numpy.random.rand", side_effect=[0.0, 1.0]):
            imu, _, vision, valid, _, _ = FusionDataset(
                [window], train=True, dropout_p=0.5
            )[0]

        self.assertFalse(valid[0])
        self.assertTrue(valid[2])
        self.assertTrue(np.all(imu.numpy() == 0.0))
        self.assertTrue(np.all(vision.numpy() == 2.0))
        np.testing.assert_array_equal(window.imu, original_imu)
        np.testing.assert_array_equal(window.vision, original_vision)


if __name__ == "__main__":
    unittest.main()
