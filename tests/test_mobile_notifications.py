import time
import unittest
from unittest.mock import MagicMock, patch

from scripts.send_mobile_notifications import (
    evaluate_event,
    initial_state,
    notification_message,
    send_ntfy,
)


def payload(
    *,
    event_id="live-000001",
    state="running",
    alert_state="FALL_ALERT",
    status="fall_alert",
    alert=True,
    fall_probability=0.91,
    imu_valid=True,
    camera_valid=True,
    timestamp=None,
):
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "mode": "live_esp32",
        "live": {"state": state, "session_id": "live_test"},
        "alert_policy": {"threshold": 0.60},
        "current_event": {
            "event_id": event_id,
            "event_timestamp_utc": timestamp,
            "alert_state": alert_state,
            "status": status,
            "alert": alert,
            "predicted_label": "stand_to_fall" if alert else "stand_to_sit",
            "fall_probability": fall_probability,
            "input": {
                "valid_modalities": {
                    "imu": imu_valid,
                    "camera_pose": camera_valid,
                }
            },
        },
    }


class MobileNotificationTests(unittest.TestCase):
    def test_fresh_valid_fall_transition_sends_once(self):
        now = time.time()
        updated, event, reason = evaluate_event(
            payload(),
            initial_state(),
            now_epoch=now,
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNotNone(event)
        self.assertEqual(reason, "new validated fall transition")
        updated["alert_active"] = True
        duplicate, event, reason = evaluate_event(
            payload(),
            updated,
            now_epoch=now + 1,
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNone(event)
        self.assertEqual(reason, "event already processed")
        self.assertTrue(duplicate["alert_active"])

    def test_invalid_camera_cannot_send(self):
        updated, event, reason = evaluate_event(
            payload(camera_valid=False),
            initial_state(),
            now_epoch=time.time(),
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNone(event)
        self.assertIn("not a validated fall", reason)
        self.assertFalse(updated["alert_active"])

    def test_stopped_or_replay_feed_cannot_send(self):
        stopped = payload(state="stopped")
        updated, event, _ = evaluate_event(
            stopped,
            initial_state(),
            now_epoch=time.time(),
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNone(event)
        stopped["mode"] = "mentor_replay"
        stopped["current_event"]["event_id"] = "live-000002"
        _, event, _ = evaluate_event(
            stopped,
            updated,
            now_epoch=time.time(),
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNone(event)

    def test_two_valid_safe_windows_rearm_after_fall(self):
        now = time.time()
        state = initial_state()
        state["alert_active"] = True
        for sequence in (2, 3):
            safe = payload(
                event_id=f"live-{sequence:06d}",
                alert_state="MONITORING",
                status="monitoring",
                alert=False,
                fall_probability=0.01,
            )
            state, event, _ = evaluate_event(
                safe,
                state,
                now_epoch=now + sequence,
                cooldown_seconds=0,
                rearm_windows=2,
                max_event_age_seconds=15,
            )
            self.assertIsNone(event)
        self.assertFalse(state["alert_active"])
        _, event, _ = evaluate_event(
            payload(event_id="live-000004"),
            state,
            now_epoch=now + 4,
            cooldown_seconds=0,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNotNone(event)

    def test_stale_fall_does_not_send(self):
        timestamp = "2020-01-01T00:00:00+00:00"
        _, event, reason = evaluate_event(
            payload(timestamp=timestamp),
            initial_state(),
            now_epoch=time.time(),
            cooldown_seconds=30,
            rearm_windows=2,
            max_event_age_seconds=15,
        )
        self.assertIsNone(event)
        self.assertIn("stale", reason)

    def test_message_is_minimal_and_actionable(self):
        message = notification_message(payload()["current_event"])
        self.assertIn("Fall detected", message)
        self.assertIn("91.0% confidence", message)
        self.assertIn("Check", message)

    @patch("scripts.send_mobile_notifications.urlopen")
    def test_ntfy_request_is_urgent_and_uses_bearer_token(self, mocked_urlopen):
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"id":"message123"}'
        mocked_urlopen.return_value.__enter__.return_value = response
        result = send_ntfy(
            server="https://ntfy.example",
            topic="private topic",
            message="Fall detected",
            token="secret",
            timeout=3,
            sequence_id="fs-event123",
        )
        self.assertEqual(result, "message123")
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://ntfy.example/private%20topic")
        self.assertEqual(request.get_header("Priority"), "urgent")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(request.get_header("X-sequence-id"), "fs-event123")
        self.assertEqual(request.data, b"Fall detected")


if __name__ == "__main__":
    unittest.main()
