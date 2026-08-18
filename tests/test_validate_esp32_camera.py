import unittest

from scripts.validate_esp32_camera import FrameObservation, analyze_observations


def make_observations(
    *, interval_us: int = 100_000, count: int = 300
) -> list[FrameObservation]:
    return [
        FrameObservation(
            sequence=index + 1,
            device_timestamp_us=1_000_000 + index * interval_us,
            host_received_s=index * interval_us / 1_000_000,
            width=320,
            height=240,
        )
        for index in range(count)
    ]


class CameraValidationTests(unittest.TestCase):
    def test_clean_ten_fps_run_passes(self) -> None:
        report = analyze_observations(
            make_observations(),
            requested_duration_s=30.0,
            target_fps=10.0,
            health={"capture_errors": 0},
        )
        self.assertEqual(report["result"], "PASS")

    def test_sequence_gap_fails(self) -> None:
        observations = make_observations()
        observations[100] = FrameObservation(
            sequence=102,
            device_timestamp_us=observations[100].device_timestamp_us,
            host_received_s=observations[100].host_received_s,
            width=320,
            height=240,
        )
        report = analyze_observations(
            observations,
            requested_duration_s=30.0,
            target_fps=10.0,
            health={"capture_errors": 0},
        )
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(report["checks"]["no_sequence_gaps"])

    def test_wrong_resolution_fails(self) -> None:
        observations = [
            FrameObservation(
                sequence=item.sequence,
                device_timestamp_us=item.device_timestamp_us,
                host_received_s=item.host_received_s,
                width=640,
                height=480,
            )
            for item in make_observations()
        ]
        report = analyze_observations(
            observations,
            requested_duration_s=30.0,
            target_fps=10.0,
            health={"capture_errors": 0},
        )
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(report["checks"]["qvga_320_by_240"])

    def test_multi_second_capture_stall_fails(self) -> None:
        observations = make_observations()
        stalled = []
        for index, item in enumerate(observations):
            extra_us = 2_000_000 if index >= 100 else 0
            stalled.append(
                FrameObservation(
                    sequence=item.sequence,
                    device_timestamp_us=item.device_timestamp_us + extra_us,
                    host_received_s=item.host_received_s,
                    width=item.width,
                    height=item.height,
                )
            )
        report = analyze_observations(
            stalled,
            requested_duration_s=30.0,
            target_fps=10.0,
            health={"capture_errors": 0},
        )
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(
            report["checks"]["max_capture_interval_at_most_5x_target"]
        )


if __name__ == "__main__":
    unittest.main()
