import csv
import math
import queue
import tempfile
import threading
import unittest
from pathlib import Path

from fusionsense.data.clock_sync import ClockMapping, SyncObservation
from scripts.record_fusion_session import (
    CameraRecord,
    MalformedImuRow,
    build_combined_report,
    flush_imu_to_line_boundary,
    read_imu_until,
    write_camera_manifest,
    write_imu_manifest,
    write_malformed_imu_rows,
    write_sync_manifest,
)
from scripts.validate_imu_stream import ImuSample


def identity_mapping(device_id: str) -> ClockMapping:
    return ClockMapping(
        device_id=device_id,
        scale=1.0,
        offset_ns=5_000_000,
        total_points=5,
        selected_points=5,
        selected_device_span_s=4.0,
        rtt_ms={"min": 1.0, "median": 1.0, "p95": 1.0, "max": 1.0},
        residual_ms={"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0},
    )


def synthetic_activity(time_s: float) -> float:
    return sum(
        math.exp(-((time_s - center) ** 2) / (2 * 0.12**2))
        for center in (15.0, 30.0, 45.0)
    )


class CombinedRecordingTests(unittest.TestCase):
    def test_clean_synthetic_bimodal_session_passes(self) -> None:
        imu_mapping = identity_mapping("imu01")
        camera_mapping = identity_mapping("cam01")
        imu_samples = [
            ImuSample(
                host_monotonic_ns=imu_mapping.map_device_us(
                    1_000_000 + index * 20_000
                )
                + 1_000_000,
                t_ms=1_000 + index * 20,
                ax_g=0.0,
                ay_g=0.0,
                az_g=1.0,
                gx_dps=100.0 * synthetic_activity(index * 0.02),
                gy_dps=0.0,
                gz_dps=0.0,
                schema_version=1,
                device_id="imu01",
                session_id="fusion_test",
                sequence=index,
                device_timestamp_us=1_000_000 + index * 20_000,
            )
            for index in range(3_000)
        ]
        camera_records = [
            CameraRecord(
                device_id="cam01",
                session_id="fusion_test",
                sequence=index + 1,
                device_timestamp_us=1_000_000 + index * 100_000,
                host_received_monotonic_ns=camera_mapping.map_device_us(
                    1_000_000 + index * 100_000
                )
                + 20_000_000,
                width=320,
                height=240,
                jpeg_bytes=1_000,
                relative_path=f"frames/{index + 1:010d}.jpg",
                motion_score=synthetic_activity(index * 0.1),
            )
            for index in range(600)
        ]

        report = build_combined_report(
            imu_samples=imu_samples,
            invalid_imu_rows=0,
            camera_records=camera_records,
            imu_mapping=imu_mapping,
            camera_mapping=camera_mapping,
            duration_s=60.0,
            target_imu_hz=50.0,
            target_camera_fps=10.0,
            stationary=False,
            motion_check=True,
            camera_health={"capture_errors": 0},
            errors=[],
        )

        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["alignment"]["result"], "PASS")

    def test_serial_flush_discards_backlog_through_line_boundary(self) -> None:
        class Connection:
            def __init__(self):
                self.reset_count = 0
                self.lines = [b"partial-row-finished\n"]

            def reset_input_buffer(self):
                self.reset_count += 1

            def readline(self):
                return self.lines.pop(0) if self.lines else b""

        connection = Connection()
        discarded = flush_imu_to_line_boundary(
            connection, clock_ns=iter([0, 1, 2, 3]).__next__
        )
        self.assertEqual(discarded, "partial-row-finished")
        self.assertEqual(connection.reset_count, 1)

    def test_malformed_serial_line_is_preserved_with_reason(self) -> None:
        stop_event = threading.Event()

        class Connection:
            def write(self, payload):
                pass

            def flush(self):
                pass

            def readline(self):
                stop_event.set()
                return b"broken,row\n"

        malformed_rows = []
        read_imu_until(
            Connection(),
            session_id="fusion_test",
            deadline_ns=1_000_000,
            regular_sync_interval_s=2.0,
            samples=[],
            sync_points=[],
            status_rows=[],
            malformed_rows=malformed_rows,
            errors=queue.SimpleQueue(),
            stop_event=stop_event,
            clock_ns=lambda: 0,
        )
        self.assertEqual(len(malformed_rows), 1)
        self.assertEqual(malformed_rows[0].raw_line, "broken,row")
        self.assertEqual(malformed_rows[0].category, "sample_parse_error")

    def test_manifests_preserve_raw_and_mapped_timestamps(self) -> None:
        imu_sample = ImuSample(
            host_monotonic_ns=9_000_000,
            t_ms=1,
            ax_g=0.0,
            ay_g=0.0,
            az_g=1.0,
            gx_dps=0.0,
            gy_dps=0.0,
            gz_dps=0.0,
            schema_version=1,
            device_id="imu01",
            session_id="fusion_test",
            sequence=1,
            device_timestamp_us=1_000,
        )
        camera_record = CameraRecord(
            device_id="cam01",
            session_id="fusion_test",
            sequence=2,
            device_timestamp_us=2_000,
            host_received_monotonic_ns=10_000_000,
            width=320,
            height=240,
            jpeg_bytes=10,
            relative_path="frames/0000000002.jpg",
        )
        sync = SyncObservation("imu01", "i1", 1_000, 5_000_000, 7_000_000)
        malformed = MalformedImuRow(
            host_received_monotonic_ns=11_000_000,
            category="sample_parse_error",
            error="wrong field count",
            raw_line="broken,row",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_imu_manifest(root / "imu.csv", [imu_sample], identity_mapping("imu01"))
            write_camera_manifest(
                root / "camera.csv", [camera_record], identity_mapping("cam01")
            )
            write_sync_manifest(root / "sync.csv", [sync])
            write_malformed_imu_rows(root / "malformed.csv", [malformed])

            with (root / "imu.csv").open(newline="", encoding="utf-8") as stream:
                imu_row = next(csv.DictReader(stream))
            with (root / "camera.csv").open(newline="", encoding="utf-8") as stream:
                camera_row = next(csv.DictReader(stream))
            with (root / "sync.csv").open(newline="", encoding="utf-8") as stream:
                sync_row = next(csv.DictReader(stream))
            with (root / "malformed.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                malformed_row = next(csv.DictReader(stream))

        self.assertEqual(imu_row["device_timestamp_us"], "1000")
        self.assertEqual(imu_row["host_capture_monotonic_ns"], "6000000")
        self.assertEqual(camera_row["host_capture_monotonic_ns"], "7000000")
        self.assertEqual(sync_row["round_trip_ns"], "2000000")
        self.assertEqual(malformed_row["raw_line"], "broken,row")


if __name__ == "__main__":
    unittest.main()
