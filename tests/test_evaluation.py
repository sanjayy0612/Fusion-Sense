import unittest

import numpy as np

from fusionsense.evaluation import (
    binary_alert_metrics,
    expected_calibration_error,
    shift_time_series,
)


class AlertMetricTests(unittest.TestCase):
    def test_alert_metrics_and_hourly_rate(self):
        report = binary_alert_metrics(
            np.array([1, 1, 0, 0], dtype=bool),
            np.array([0.9, 0.2, 0.8, 0.1]),
            0.6,
            negative_hours=0.5,
        )
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["false_positive"], 1)
        self.assertEqual(report["false_negative"], 1)
        self.assertEqual(report["true_negative"], 1)
        self.assertEqual(report["false_alerts_per_hour"], 2.0)
        self.assertEqual(len(report["precision_wilson_95"]), 2)

    def test_zero_false_alerts_reports_nonzero_rate_upper_bound(self):
        report = binary_alert_metrics(
            np.zeros(10, dtype=bool), np.zeros(10), 0.6, negative_hours=1.0
        )
        self.assertEqual(report["false_alerts_per_hour"], 0.0)
        self.assertGreater(report["zero_count_upper_95_false_alerts_per_hour"], 2.9)

    def test_perfect_calibration_has_zero_error(self):
        truth = np.array([0, 1], dtype=bool)
        self.assertEqual(expected_calibration_error(truth, truth.astype(float)), 0.0)


class SynchronizationShiftTests(unittest.TestCase):
    def test_integer_sample_shift_uses_edge_padding(self):
        values = np.arange(5, dtype=np.float32).reshape(-1, 1)
        shifted = shift_time_series(values, shift_ms=1000.0, rate_hz=1.0)
        np.testing.assert_array_equal(shifted[:, 0], [0, 0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
