from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.train_bimodal_step7 import paired_bimodal_windows


class Step7TrainingTests(unittest.TestCase):
    def test_training_selection_requires_both_inputs_and_masks_radar(self):
        windows = []
        for label in range(7):
            windows.append(
                SimpleNamespace(
                    label=label,
                    imu_valid=True,
                    vision_valid=True,
                    radar_valid=False,
                )
            )
        windows.extend(
            [
                SimpleNamespace(
                    label=0,
                    imu_valid=True,
                    vision_valid=False,
                    radar_valid=False,
                ),
                SimpleNamespace(
                    label=1,
                    imu_valid=True,
                    vision_valid=True,
                    radar_valid=True,
                ),
            ]
        )

        selected = paired_bimodal_windows(windows)

        self.assertEqual(len(selected), 7)
        self.assertTrue(all(item.imu_valid and item.vision_valid for item in selected))
        self.assertTrue(all(not item.radar_valid for item in selected))


if __name__ == "__main__":
    unittest.main()
