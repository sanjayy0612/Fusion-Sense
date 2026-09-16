"""Send debounced FusionSense fall alerts to a phone through ntfy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEED = REPO_ROOT / "dashboard" / "live_output.json"
DEFAULT_STATE = REPO_ROOT / "dashboard" / "notification_state.json"


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "last_seen_event": None,
        "last_notified_event": None,
        "last_notification_epoch": None,
        "alert_active": False,
        "valid_safe_windows": 0,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return initial_state()
    state = initial_state()
    state.update(load_json(path))
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.25))
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_event_epoch(event: dict[str, Any]) -> float | None:
    raw = event.get("event_timestamp_utc")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def event_key(payload: dict[str, Any], event: dict[str, Any]) -> str | None:
    event_id = event.get("event_id")
    session_id = payload.get("live", {}).get("session_id")
    if not isinstance(event_id, str) or not isinstance(session_id, str):
        return None
    return f"{session_id}:{event_id}"


def is_valid_monitoring_event(event: dict[str, Any]) -> bool:
    valid = event.get("input", {}).get("valid_modalities", {})
    return (
        event.get("alert_state") == "MONITORING"
        and event.get("status") == "monitoring"
        and valid.get("imu") is True
        and valid.get("camera_pose") is True
    )


def is_confirmed_fall(payload: dict[str, Any], event: dict[str, Any]) -> bool:
    valid = event.get("input", {}).get("valid_modalities", {})
    try:
        probability = float(event.get("fall_probability"))
        threshold = float(payload.get("alert_policy", {}).get("threshold"))
    except (TypeError, ValueError):
        return False
    return (
        payload.get("mode") == "live_esp32"
        and payload.get("live", {}).get("state") == "running"
        and event.get("alert_state") == "FALL_ALERT"
        and event.get("status") == "fall_alert"
        and event.get("alert") is True
        and event.get("predicted_label") == "stand_to_fall"
        and probability >= threshold
        and valid.get("imu") is True
        and valid.get("camera_pose") is True
    )


def evaluate_event(
    payload: dict[str, Any],
    state: dict[str, Any],
    *,
    now_epoch: float,
    cooldown_seconds: float,
    rearm_windows: int,
    max_event_age_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Return updated state, an alert event to deliver, and a decision reason."""
    updated = deepcopy(state)
    event = payload.get("current_event")
    if not isinstance(event, dict):
        return updated, None, "no current event"
    key = event_key(payload, event)
    if key is None:
        return updated, None, "event identity missing"
    if key == updated.get("last_seen_event"):
        return updated, None, "event already processed"

    updated["last_seen_event"] = key
    if is_confirmed_fall(payload, event):
        updated["valid_safe_windows"] = 0
        event_epoch = parse_event_epoch(event)
        if event_epoch is None:
            return updated, None, "fall timestamp missing or invalid"
        age = now_epoch - event_epoch
        if age < -5 or age > max_event_age_seconds:
            return updated, None, f"fall event is stale ({age:.1f}s old)"
        if updated.get("alert_active"):
            return updated, None, "fall episode already notified"
        previous = updated.get("last_notification_epoch")
        if previous is not None and now_epoch - float(previous) < cooldown_seconds:
            return updated, None, "notification cooldown active"
        return updated, event, "new validated fall transition"

    if is_valid_monitoring_event(event):
        safe_windows = int(updated.get("valid_safe_windows", 0)) + 1
        updated["valid_safe_windows"] = safe_windows
        if safe_windows >= rearm_windows:
            updated["alert_active"] = False
        return updated, None, "valid non-fall window"

    updated["valid_safe_windows"] = 0
    return updated, None, "not a validated fall; degraded input cannot re-arm"


def notification_message(event: dict[str, Any]) -> str:
    probability = float(event.get("fall_probability", 0.0)) * 100
    raw_time = str(event.get("event_timestamp_utc", "time unavailable"))
    try:
        timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        timestamp = timestamp.astimezone(timezone.utc)
        rendered_time = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        rendered_time = raw_time
    return (
        f"Fall detected ({probability:.1f}% confidence) at {rendered_time}. "
        "Check the monitored person immediately."
    )


def send_ntfy(
    *,
    server: str,
    topic: str,
    message: str,
    token: str | None,
    timeout: float,
    sequence_id: str | None = None,
) -> str | None:
    url = f"{server.rstrip('/')}/{quote(topic, safe='')}"
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Title": "FusionSense fall alert",
        "Priority": "urgent",
        "Tags": "warning",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if sequence_id:
        headers["X-Sequence-ID"] = sequence_id
    request = Request(
        url,
        data=message.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"ntfy returned HTTP {response.status}")
        raw = response.read().decode("utf-8", errors="replace").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    message_id = parsed.get("id")
    return str(message_id) if message_id is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, default=DEFAULT_FEED)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--server", default=os.getenv("FUSIONSENSE_NTFY_SERVER", "https://ntfy.sh"))
    parser.add_argument("--topic", default=os.getenv("FUSIONSENSE_NTFY_TOPIC"))
    parser.add_argument("--token", default=os.getenv("FUSIONSENSE_NTFY_TOKEN"))
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--rearm-windows", type=int, default=2)
    parser.add_argument("--max-event-age-seconds", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--test-notification",
        action="store_true",
        help="send one setup message and exit without reading the model feed",
    )
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.cooldown_seconds < 0
        or args.rearm_windows <= 0
        or args.max_event_age_seconds <= 0
        or args.timeout <= 0
    ):
        parser.error("poll/age/timeout must be positive; cooldown must be nonnegative")
    if not args.topic and not args.dry_run:
        parser.error(
            "--topic or FUSIONSENSE_NTFY_TOPIC is required unless --dry-run is used"
        )

    if args.test_notification:
        message = "FusionSense mobile fall notifications are connected."
        if args.dry_run:
            print(f"DRY RUN: {message}")
            return 0
        try:
            message_id = send_ntfy(
                server=args.server,
                topic=args.topic,
                message=message,
                token=args.token,
                timeout=args.timeout,
            )
        except (HTTPError, URLError, OSError, RuntimeError) as error:
            print(f"Test notification failed: {error}", file=sys.stderr)
            return 2
        print(f"Test notification sent (message_id={message_id or 'unknown'})")
        return 0

    state = load_state(args.state)
    print(f"Watching FusionSense feed: {args.feed}")
    print(
        "Notification target: "
        + ("dry run" if args.dry_run else f"{args.server.rstrip('/')}/{args.topic}")
    )
    while True:
        try:
            payload = load_json(args.feed)
            now_epoch = time.time()
            updated, alert_event, reason = evaluate_event(
                payload,
                state,
                now_epoch=now_epoch,
                cooldown_seconds=args.cooldown_seconds,
                rearm_windows=args.rearm_windows,
                max_event_age_seconds=args.max_event_age_seconds,
            )
            if alert_event is not None:
                message = notification_message(alert_event)
                event_identity = str(updated["last_seen_event"])
                sequence_id = "fs-" + hashlib.sha256(
                    event_identity.encode("utf-8")
                ).hexdigest()[:20]
                if args.dry_run:
                    print(f"DRY RUN: {message}")
                    message_id = "dry-run"
                else:
                    message_id = send_ntfy(
                        server=args.server,
                        topic=args.topic,
                        message=message,
                        token=args.token,
                        timeout=args.timeout,
                        sequence_id=sequence_id,
                    )
                updated["alert_active"] = True
                updated["last_notified_event"] = updated["last_seen_event"]
                updated["last_notification_epoch"] = now_epoch
                updated["last_notification_id"] = message_id
                print(
                    f"FALL notification sent for {updated['last_notified_event']} "
                    f"({message_id or 'no provider id'})"
                )
            elif updated.get("last_seen_event") != state.get("last_seen_event"):
                print(f"No notification: {reason}")
            if updated != state:
                # Update memory before persisting so a transient state-file lock
                # cannot cause the same process to send the alert again.
                state = updated
                write_state(args.state, state)
        except FileNotFoundError:
            print(f"Waiting for feed: {args.feed}", file=sys.stderr)
        except json.JSONDecodeError:
            # The producer writes atomically, but OneDrive can briefly expose a stale handle.
            print("Waiting for a complete dashboard feed", file=sys.stderr)
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
            print(f"Notification consumer warning: {error}", file=sys.stderr)
            if args.once:
                return 2

        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
