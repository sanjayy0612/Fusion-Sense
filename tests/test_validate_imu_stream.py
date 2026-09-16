import unittest

from scripts.validate_imu_stream import ImuSample, analyze_samples, parse_sample


def make_stationary_samples(interval_ms: int, count: int = 500) -> list[ImuSample]:
    return [
        ImuSample(
            host_monotonic_ns=index * interval_ms * 1_000_000,
            t_ms=index * interval_ms,
            ax_g=0.0,
            ay_g=0.0,
            az_g=1.0,
            gx_dps=0.1,
            gy_dps=-0.1,
            gz_dps=0.05,
        )
        for index in range(count)
    ]


class ParseSampleTests(unittest.TestCase):
    def test_parses_verified_seven_column_format(self) -> None:
        sample = parse_sample(
            "9783,0.620,-0.727,0.292,-1.06,1.00,0.16", 123
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.t_ms, 9783)
        self.assertAlmostEqual(sample.ay_g, -0.727)

    def test_ignores_status_line(self) -> None:
        self.assertIsNone(parse_sample("# health,effective_hz=50.00", 123))

    def test_rejects_wrong_field_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_sample("1,2,3", 123)

    def test_parses_versioned_timestamped_packet(self) -> None:
        sample = parse_sample(
            "IMU,1,imu01,session_a,42,1234567,0.1,0.2,0.9,1.0,2.0,3.0",
            999,
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.device_id, "imu01")
        self.assertEqual(sample.session_id, "session_a")
        self.assertEqual(sample.sequence, 42)
        self.assertEqual(sample.device_timestamp_us, 1234567)

    def test_versioned_sequence_gap_fails(self) -> None:
        samples = [
            parse_sample(
                f"IMU,1,imu01,s,"
                f"{index + (1 if index >= 10 else 0)},"
                f"{1_000_000 + index * 20_000},0,0,1,0,0,0",
                index * 20_000_000,
            )
            for index in range(100)
        ]
        report = analyze_samples([sample for sample in samples if sample is not None])
        self.assertEqual(report["result"], "FAIL")
        self.assertEqual(report["sequence_gaps"], 1)


class AnalysisTests(unittest.TestCase):
    def test_exact_50_hz_stationary_stream_passes(self) -> None:
        report = analyze_samples(make_stationary_samples(20))
        self.assertEqual(report["result"], "PASS")

    def test_ten_hz_stream_fails_rate_check(self) -> None:
        report = analyze_samples(make_stationary_samples(100))
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(report["checks"]["rate_49_to_51_hz"])

    def test_timestamp_gap_is_counted(self) -> None:
        samples = make_stationary_samples(20)
        shifted = []
        for index, sample in enumerate(samples):
            offset = 40 if index >= 100 else 0
            shifted.append(
                ImuSample(
                    **{
                        **sample.__dict__,
                        "t_ms": sample.t_ms + offset,
                    }
                )
            )
        report = analyze_samples(shifted)
        self.assertEqual(report["estimated_missed_slots"], 2)
        self.assertEqual(report["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
