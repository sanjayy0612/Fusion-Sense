"""Run strict IMU+camera fall inference continuously on the laptop."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.contract import ACTIVITIES  # noqa: E402
from fusionsense.data.camera_serial import (  # noqa: E402
    SERIAL_CAMERA_BAUD,
    CameraSerialError,
    SerialCameraStream,
)
from fusionsense.data.imu_units import (  # noqa: E402
    CANONICAL_IMU_UNITS,
    live_conversion_metadata,
)
from fusionsense.data.vision_extractor import PoseExtractor, _lazy_imports  # noqa: E402
from fusionsense.inference import FallInferenceEngine  # noqa: E402
from fusionsense.live import (  # noqa: E402
    LiveImuPoint,
    LivePosePoint,
    assemble_live_window,
)
from scripts.record_fusion_session import (  # noqa: E402
    acknowledge_imu_session,
    flush_imu_to_line_boundary,
)
from scripts.record_imu_serial import validate_session_id  # noqa: E402
from scripts.validate_imu_stream import parse_sample  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "cmhad_step7_bimodal_dropout_fixed"
)
DEFAULT_OUTPUT = REPO_ROOT / "dashboard" / "live_output.json"
DEFAULT_STEP8_REPORT = REPO_ROOT / "checkpoints" / "step8_pilot_evaluation_v3.json"


class LiveBuffers:
    def __init__(self):
        self.lock = threading.Lock()
        self.imu = deque(maxlen=1_500)
        self.pose = deque(maxlen=300)
        self.imu_rows = 0
        self.invalid_imu_rows = 0
        self.camera_frames = 0
        self.pose_valid_frames = 0
        self.camera_transport_errors = 0
        self.last_camera_transport_error: str | None = None


def atomic_write_json(
    path: Path,
    payload: dict,
    *,
    retries: int = 30,
    retry_delay: float = 0.05,
) -> None:
    """Atomically publish dashboard state, tolerating brief Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(retries):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 >= retries:
                    raise
                time.sleep(min(retry_delay * (attempt + 1), 0.25))
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_serial_port(
    requested: str,
    *,
    expected_vid: int,
    expected_pid: int,
    label: str,
    ports=None,
) -> str:
    """Use the requested COM port, or recover its USB device after reassignment."""
    if ports is None:
        from serial.tools import list_ports  # type: ignore[import-not-found]

        ports = list(list_ports.comports())
    available = list(ports)
    for port in available:
        if str(port.device).casefold() == requested.casefold():
            return str(port.device)
    matches = [
        str(port.device)
        for port in available
        if port.vid == expected_vid and port.pid == expected_pid
    ]
    if len(matches) == 1:
        print(
            f"# port_reassigned,device={label},requested={requested},using={matches[0]}"
        )
        return matches[0]
    return requested


def recoverable_camera_error(error: Exception) -> bool:
    """Identify an isolated damaged frame that can safely be discarded."""
    message = str(error).casefold()
    return "crc mismatch" in message or "could not decode" in message


def load_step8_metrics(path: Path, checkpoint_name: str) -> dict:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    candidate = report.get("models", {}).get(checkpoint_name, {})
    both = candidate.get("modalities", {}).get("both", {})
    classification = both.get("classification", {})
    alert = both.get("alert_at_0_60", {})
    return {
        "accuracy": classification.get("accuracy"),
        "macro_f1": classification.get("macro_f1"),
        "fall_recall": alert.get("recall"),
        "fall_precision": alert.get("precision"),
        "validation_falls": report.get("validation_falls"),
        "scope": report.get("scope"),
    }


def base_payload(
    engine: FallInferenceEngine,
    *,
    session_id: str,
    evaluation: dict,
    imu_port: str,
    camera_port: str,
) -> dict:
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "STARTING",
        "mode": "live_esp32",
        "title": "FusionSense Live Fall Detection",
        "disclosure": (
            "Live research-prototype inference from synchronized ESP32 IMU and "
            "ESP32-CAM pose. It is not a clinical or emergency-response device."
        ),
        "model": {
            "name": engine.model_name,
            "classes": ACTIVITIES,
            "checkpoint": str(engine.checkpoint_dir.resolve()),
            "scope_note": (
                "Step 8 pilot checkpoint; strict IMU + camera validity is required."
            ),
        },
        "input_contract": {
            "window_seconds": engine.cfg.window_seconds,
            "stride_seconds": 1.0,
            "imu": {"shape": [engine.cfg.t_imu, 6], "units": list(CANONICAL_IMU_UNITS)},
            "camera_pose": {
                "shape": [engine.cfg.t_vis, engine.cfg.vision_dv],
                "rate_fps": engine.cfg.vision_fps,
            },
            "fusion_policy": "valid IMU and valid full-body camera pose required",
            "radar": {"active": False, "note": "Not an input to this project"},
            "live_esp32_conversion": live_conversion_metadata(),
        },
        "evaluation": evaluation,
        "alert_policy": {
            "class": "stand_to_fall",
            "threshold": engine.fall_threshold,
            "condition": (
                "both modalities valid and stand_to_fall probability >= threshold"
            ),
            "required_modalities": ["imu", "camera_pose"],
        },
        "live": {
            "session_id": session_id,
            "imu_port": imu_port,
            "camera_port": camera_port,
            "state": "starting",
        },
        "summary": {
            "inference_windows": 0,
            "alerts": 0,
            "degraded_windows": 0,
        },
        "events": [],
        "current_event": None,
    }


def read_imu_live(
    connection,
    *,
    session_id: str,
    latency_correction_ns: int,
    buffers: LiveBuffers,
    errors: "queue.SimpleQueue[str]",
    stop_event: threading.Event,
) -> None:
    device_clock_offset_ns = None
    try:
        while not stop_event.is_set():
            raw = connection.readline()
            received_ns = time.monotonic_ns()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            try:
                sample = parse_sample(line, received_ns)
            except (ValueError, OverflowError):
                with buffers.lock:
                    buffers.invalid_imu_rows += 1
                continue
            if sample is None:
                continue
            if (
                sample.schema_version != 1
                or sample.device_id != "imu01"
                or sample.session_id != session_id
            ):
                with buffers.lock:
                    buffers.invalid_imu_rows += 1
                continue
            if sample.device_timestamp_us is None:
                with buffers.lock:
                    buffers.invalid_imu_rows += 1
                continue
            if device_clock_offset_ns is None:
                device_clock_offset_ns = (
                    received_ns
                    - latency_correction_ns
                    - int(sample.device_timestamp_us) * 1000
                )
            point = LiveImuPoint(
                capture_ns=(
                    int(sample.device_timestamp_us) * 1000 + device_clock_offset_ns
                ),
                values_g_dps=np.asarray(
                    [
                        sample.ax_g,
                        sample.ay_g,
                        sample.az_g,
                        sample.gx_dps,
                        sample.gy_dps,
                        sample.gz_dps,
                    ],
                    dtype=np.float32,
                ),
                sequence=sample.sequence,
            )
            with buffers.lock:
                buffers.imu.append(point)
                buffers.imu_rows += 1
    except Exception as error:
        errors.put(f"IMU reader failed: {error}")
        stop_event.set()


def read_camera_live(
    camera: SerialCameraStream,
    *,
    latency_correction_ns: int,
    buffers: LiveBuffers,
    errors: "queue.SimpleQueue[str]",
    stop_event: threading.Event,
) -> None:
    device_clock_offset_ns = None
    consecutive_transport_errors = 0
    try:
        cv2, _ = _lazy_imports()
        with PoseExtractor() as extractor:
            while not stop_event.is_set():
                try:
                    frame = camera.read()
                except CameraSerialError as error:
                    if not recoverable_camera_error(error):
                        raise
                    consecutive_transport_errors += 1
                    with buffers.lock:
                        buffers.camera_transport_errors += 1
                        buffers.last_camera_transport_error = str(error)
                    print(
                        "# warning,camera_frame_discarded,"
                        f"consecutive={consecutive_transport_errors},reason={error}",
                        file=sys.stderr,
                    )
                    if consecutive_transport_errors >= 5:
                        raise CameraSerialError(
                            "camera produced 5 consecutive corrupt frames; "
                            f"last error: {error}"
                        )
                    continue
                consecutive_transport_errors = 0
                image = cv2.imdecode(
                    np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    error = CameraSerialError(
                        "OpenCV could not decode a live camera JPEG"
                    )
                    with buffers.lock:
                        buffers.camera_transport_errors += 1
                        buffers.last_camera_transport_error = str(error)
                    print(
                        f"# warning,camera_frame_discarded,reason={error}",
                        file=sys.stderr,
                    )
                    continue
                pose = extractor.extract(image, frame.device_timestamp_us / 1e6)
                if device_clock_offset_ns is None:
                    device_clock_offset_ns = (
                        frame.host_header_received_monotonic_ns
                        - latency_correction_ns
                        - int(frame.device_timestamp_us) * 1000
                    )
                point = LivePosePoint(
                    capture_ns=(
                        int(frame.device_timestamp_us) * 1000
                        + device_clock_offset_ns
                    ),
                    landmarks=pose.landmarks,
                    valid=pose.valid,
                    image_quality=pose.image_quality,
                    sequence=frame.sequence,
                )
                with buffers.lock:
                    buffers.pose.append(point)
                    buffers.camera_frames += 1
                    buffers.pose_valid_frames += int(pose.valid)
    except Exception as error:
        errors.put(f"camera reader failed: {error}")
        stop_event.set()


def build_event(
    prediction: dict,
    *,
    sequence: int,
    window_end_ns: int,
    wall_clock_offset_ns: int,
    diagnostics: dict,
) -> dict:
    event = dict(prediction)
    event["event_id"] = f"live-{sequence:06d}"
    event["sequence"] = sequence
    event["source_mode"] = "live_esp32"
    event["ground_truth"] = None
    event["event_timestamp_utc"] = datetime.fromtimestamp(
        (window_end_ns + wall_clock_offset_ns) / 1e9, timezone.utc
    ).isoformat()
    event["window_diagnostics"] = diagnostics
    event["alert_state"] = (
        "FALL_ALERT"
        if event["status"] == "fall_alert"
        else "DEGRADED"
        if event["status"] == "degraded"
        else "MONITORING"
    )
    return event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-port", default="COM16")
    parser.add_argument("--camera-port", default="COM14")
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--camera-baud", type=int, default=SERIAL_CAMERA_BAUD)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--step8-report", type=Path, default=DEFAULT_STEP8_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fall-threshold", type=float, default=0.60)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--imu-latency-ms",
        type=float,
        default=7.945912,
        help="Step 8 median USB transport compensation",
    )
    parser.add_argument(
        "--camera-latency-ms",
        type=float,
        default=297.408335,
        help="Step 8 median USB transport compensation",
    )
    args = parser.parse_args()
    if args.duration < 0 or args.stride_seconds <= 0 or args.max_events <= 0:
        parser.error("duration must be >= 0; stride and max-events must be positive")

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        print("pyserial is missing; install requirements.txt", file=sys.stderr)
        return 2

    imu_port = resolve_serial_port(
        args.imu_port,
        expected_vid=0x10C4,
        expected_pid=0xEA60,
        label="imu",
    )
    camera_port = resolve_serial_port(
        args.camera_port,
        expected_vid=0x1A86,
        expected_pid=0x7523,
        label="camera",
    )

    session_id = validate_session_id(
        datetime.now(timezone.utc).strftime("live_%Y%m%dT%H%M%SZ")
    )
    engine = FallInferenceEngine(
        args.checkpoint_dir,
        fall_threshold=args.fall_threshold,
        device=args.device,
        require_bimodal=True,
    )
    evaluation = load_step8_metrics(args.step8_report, args.checkpoint_dir.name)
    payload = base_payload(
        engine,
        session_id=session_id,
        evaluation=evaluation,
        imu_port=imu_port,
        camera_port=camera_port,
    )
    payload["input_contract"]["stride_seconds"] = args.stride_seconds
    atomic_write_json(args.output, payload)

    buffers = LiveBuffers()
    errors: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    stop_event = threading.Event()
    events: list[dict] = []
    started = time.monotonic()
    wall_clock_offset_ns = time.time_ns() - time.monotonic_ns()
    last_window_end_ns: int | None = None
    event_sequence = 0
    imu_latency_ns = int(args.imu_latency_ms * 1e6)
    camera_latency_ns = int(args.camera_latency_ms * 1e6)

    print(
        f"Opening live IMU {imu_port} @ {args.imu_baud} and "
        f"camera {camera_port} @ {args.camera_baud}"
    )
    print(f"Dashboard feed: {args.output}")
    try:
        with serial.Serial(imu_port, args.imu_baud, timeout=0.1) as imu_connection:
            acknowledge_imu_session(
                imu_connection, session_id, timeout_s=args.startup_timeout
            )
            with SerialCameraStream(
                camera_port,
                baud=args.camera_baud,
                frame_timeout=args.startup_timeout,
            ) as camera:
                for startup_attempt in range(5):
                    try:
                        camera.read()
                        break
                    except CameraSerialError as error:
                        if not recoverable_camera_error(error) or startup_attempt == 4:
                            raise
                        print(
                            "# warning,camera_startup_frame_discarded,"
                            f"attempt={startup_attempt + 1},reason={error}",
                            file=sys.stderr,
                        )
                flush_imu_to_line_boundary(imu_connection)
                workers = [
                    threading.Thread(
                        target=read_imu_live,
                        kwargs={
                            "connection": imu_connection,
                            "session_id": session_id,
                            "latency_correction_ns": imu_latency_ns,
                            "buffers": buffers,
                            "errors": errors,
                            "stop_event": stop_event,
                        },
                        name="live-imu",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=read_camera_live,
                        kwargs={
                            "camera": camera,
                            "latency_correction_ns": camera_latency_ns,
                            "buffers": buffers,
                            "errors": errors,
                            "stop_event": stop_event,
                        },
                        name="live-camera",
                        daemon=True,
                    ),
                ]
                for worker in workers:
                    worker.start()

                while not stop_event.is_set():
                    if args.duration and time.monotonic() - started >= args.duration:
                        break
                    if not errors.empty():
                        stop_event.set()
                        break
                    with buffers.lock:
                        imu_points = list(buffers.imu)
                        pose_points = list(buffers.pose)
                        counts = {
                            "imu_rows": buffers.imu_rows,
                            "invalid_imu_rows": buffers.invalid_imu_rows,
                            "camera_frames": buffers.camera_frames,
                            "pose_valid_frames": buffers.pose_valid_frames,
                            "camera_transport_errors": (
                                buffers.camera_transport_errors
                            ),
                            "last_camera_transport_error": (
                                buffers.last_camera_transport_error
                            ),
                        }
                    if len(imu_points) >= 2 and pose_points:
                        common_end_ns = min(
                            imu_points[-1].capture_ns + int(1e9 / engine.cfg.imu_hz),
                            pose_points[-1].capture_ns + int(1e9 / engine.cfg.vision_fps),
                        )
                        ready = (
                            common_end_ns
                            - max(imu_points[0].capture_ns, pose_points[0].capture_ns)
                            >= int(engine.cfg.window_seconds * 1e9)
                        )
                        due = (
                            last_window_end_ns is None
                            or common_end_ns - last_window_end_ns
                            >= int(args.stride_seconds * 1e9)
                        )
                        if ready and due:
                            try:
                                window, diagnostics = assemble_live_window(
                                    imu_points,
                                    pose_points,
                                    window_end_ns=common_end_ns,
                                    session_id=session_id,
                                    cfg=engine.cfg,
                                )
                            except ValueError as error:
                                errors.put(f"live window assembly failed: {error}")
                                stop_event.set()
                                continue
                            prediction = engine.predict(window)
                            event_sequence += 1
                            event = build_event(
                                prediction,
                                sequence=event_sequence,
                                window_end_ns=common_end_ns,
                                wall_clock_offset_ns=wall_clock_offset_ns,
                                diagnostics=diagnostics,
                            )
                            events.append(event)
                            events = events[-args.max_events :]
                            last_window_end_ns = common_end_ns
                            payload["generated_at_utc"] = datetime.now(
                                timezone.utc
                            ).isoformat()
                            payload["result"] = event["alert_state"]
                            payload["events"] = events
                            payload["current_event"] = event
                            payload["live"].update(
                                {
                                    "state": "running",
                                    "uptime_seconds": round(time.monotonic() - started, 1),
                                    "transport_compensation_ms": {
                                        "imu": args.imu_latency_ms,
                                        "camera": args.camera_latency_ms,
                                    },
                                    **counts,
                                }
                            )
                            payload["summary"] = {
                                "inference_windows": event_sequence,
                                "alerts": sum(item["alert"] for item in events),
                                "degraded_windows": sum(
                                    item["status"] == "degraded" for item in events
                                ),
                            }
                            atomic_write_json(args.output, payload)
                            print(
                                f"{event['event_id']} {event['alert_state']} "
                                f"activity={event['predicted_label']} "
                                f"fall={event['fall_probability']:.3f}"
                            )
                    time.sleep(0.05)

                stop_event.set()
                for worker in workers:
                    worker.join(timeout=6.0)
    except KeyboardInterrupt:
        stop_event.set()
    except (OSError, RuntimeError, CameraSerialError, serial.SerialException) as error:
        errors.put(str(error))

    error_messages = []
    while not errors.empty():
        error_messages.append(errors.get())
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["live"]["state"] = "error" if error_messages else "stopped"
    payload["live"]["errors"] = error_messages
    if error_messages:
        payload["result"] = "ERROR"
    atomic_write_json(args.output, payload)
    if error_messages:
        print("Live pipeline stopped: " + "; ".join(error_messages), file=sys.stderr)
        return 2
    print(f"Live pipeline stopped after {event_sequence} inference windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
