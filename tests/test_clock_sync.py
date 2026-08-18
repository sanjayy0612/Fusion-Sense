import unittest

from fusionsense.data.clock_sync import (
    SyncObservation,
    clock_mapping_passes,
    fit_affine_clock,
    nearest_alignment_report,
    parse_imu_sync_response,
)


class ClockSyncTests(unittest.TestCase):
    def test_affine_fit_recovers_offset_and_drift(self) -> None:
        scale = 1.000075
        offset_ns = 8_000_000_000_000
        observations = []
        for index in range(30):
            device_us = 2_000_000 + index * 1_000_000
            midpoint = int(round(scale * device_us * 1000 + offset_ns))
            rtt_ns = 1_000_000 + (index % 4) * 100_000
            observations.append(
                SyncObservation(
                    device_id="imu01",
                    request_id=f"i{index}",
                    device_time_us=device_us,
                    host_send_ns=midpoint - rtt_ns // 2,
                    host_receive_ns=midpoint + rtt_ns // 2,
                )
            )

        mapping = fit_affine_clock(observations)
        self.assertAlmostEqual(mapping.scale, scale, places=9)
        self.assertAlmostEqual(mapping.offset_ns, offset_ns, delta=2.0)
        self.assertTrue(all(clock_mapping_passes(mapping).values()))

    def test_nearest_alignment_uses_mapped_capture_time(self) -> None:
        imu = [index * 20_000_000 for index in range(101)]
        camera = [index * 100_000_000 + 8_000_000 for index in range(20)]
        report = nearest_alignment_report(imu, camera)
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["nearest_imu_skew_ms"]["median"], 8.0)

    def test_imu_sync_parser(self) -> None:
        self.assertEqual(
            parse_imu_sync_response("# SYNC_RESP,i000001,1234567"),
            ("i000001", 1234567),
        )
        self.assertIsNone(parse_imu_sync_response("# health,t_ms=123"))
        with self.assertRaises(ValueError):
            parse_imu_sync_response("# SYNC_RESP,bad")


if __name__ == "__main__":
    unittest.main()
