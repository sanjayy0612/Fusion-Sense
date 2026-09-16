import unittest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fusionsense.config import CFG
from scripts.run_fall_pipeline import DEFAULT_CHECKPOINT as REPLAY_CHECKPOINT
from scripts.run_live_fall_pipeline import (
    DEFAULT_CHECKPOINT as LIVE_CHECKPOINT,
    atomic_write_json,
    base_payload,
    build_event,
    recoverable_camera_error,
    resolve_serial_port,
)


class Step9LiveContractTests(unittest.TestCase):
    def test_replay_and_live_use_step8_selected_checkpoint(self):
        expected = "cmhad_step7_bimodal_dropout_fixed"
        self.assertEqual(REPLAY_CHECKPOINT.name, expected)
        self.assertEqual(LIVE_CHECKPOINT.name, expected)

    def test_live_payload_requires_both_modalities(self):
        engine = SimpleNamespace(
            model_name="test fusion",
            checkpoint_dir=Path("checkpoint"),
            fall_threshold=0.6,
            cfg=CFG,
        )
        payload = base_payload(
            engine,
            session_id="live_test",
            evaluation={},
            imu_port="COM17",
            camera_port="COM14",
        )
        self.assertEqual(
            payload["alert_policy"]["required_modalities"], ["imu", "camera_pose"]
        )
        self.assertIn("valid IMU", payload["input_contract"]["fusion_policy"])

    def test_live_event_exports_dashboard_contract(self):
        prediction = {
            "status": "fall_alert",
            "alert": True,
            "predicted_label": "stand_to_fall",
            "fall_probability": 0.91,
        }
        event = build_event(
            prediction,
            sequence=7,
            window_end_ns=10_000_000_000,
            wall_clock_offset_ns=1_700_000_000_000_000_000,
            diagnostics={"imu": {"valid": True}, "camera": {"valid": True}},
        )
        self.assertEqual(event["alert_state"], "FALL_ALERT")
        self.assertEqual(event["event_id"], "live-000007")
        self.assertTrue(event["event_timestamp_utc"].endswith("+00:00"))
        self.assertIn("window_diagnostics", event)

    def test_missing_requested_port_uses_unique_usb_device_match(self):
        ports = [
            SimpleNamespace(device="COM16", vid=0x10C4, pid=0xEA60),
            SimpleNamespace(device="COM14", vid=0x1A86, pid=0x7523),
        ]
        self.assertEqual(
            resolve_serial_port(
                "COM17",
                expected_vid=0x10C4,
                expected_pid=0xEA60,
                label="imu",
                ports=ports,
            ),
            "COM16",
        )

    def test_camera_crc_mismatch_is_recoverable(self):
        self.assertTrue(
            recoverable_camera_error(
                RuntimeError("camera JPEG CRC mismatch: expected 1, got 2")
            )
        )
        self.assertFalse(recoverable_camera_error(RuntimeError("camera disconnected")))

    def test_dashboard_write_retries_a_transient_windows_lock(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "live_output.json"
            real_replace = os.replace
            attempts = {"count": 0}

            def flaky_replace(source, destination):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise PermissionError(5, "Access is denied")
                return real_replace(source, destination)

            with patch(
                "scripts.run_live_fall_pipeline.os.replace",
                side_effect=flaky_replace,
            ):
                atomic_write_json(
                    target,
                    {"result": "MONITORING"},
                    retries=3,
                    retry_delay=0,
                )
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8"))["result"],
                "MONITORING",
            )
            self.assertEqual(attempts["count"], 3)


if __name__ == "__main__":
    unittest.main()
