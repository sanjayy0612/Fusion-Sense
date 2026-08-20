import unittest

from scripts.build_step8_background import overlaps_or_follows_fall
from fusionsense.contract import LABEL2ID


class Step8BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"label": LABEL2ID["stand_to_fall"], "start": 10.0, "end": 12.0}
        ]

    def test_window_before_guarded_fall_is_negative(self):
        self.assertFalse(overlaps_or_follows_fall(6.0, 8.0, self.events, 1.0))

    def test_window_entering_guard_is_excluded(self):
        self.assertTrue(overlaps_or_follows_fall(8.0, 10.0, self.events, 1.0))

    def test_post_fall_window_is_excluded(self):
        self.assertTrue(overlaps_or_follows_fall(14.0, 16.0, self.events, 1.0))


if __name__ == "__main__":
    unittest.main()
