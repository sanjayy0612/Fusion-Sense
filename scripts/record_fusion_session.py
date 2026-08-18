"""Record synchronized ESP32 IMU + ESP32-CAM data in one laptop session."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.camera_control import CameraControlClient, CameraControlError
from fusionsense.data.camera_stream import (
    CameraConnectionError,
    TimestampedMjpegStream,
    esp32_stream_url,
)
from fusionsense.data.clock_sync import (
    ClockMapping,
    SyncObservation,
    clock_mapping_passes,
    fit_affine_clock,
    nearest_alignment_report,
    parse_imu_sync_response,
    transport_latency_report,
)
from fusionsense.data.motion_sync import MotionPoint, motion_alignment_report
from scripts.record_esp32_camera import save_original_jpeg
from scripts.record_imu_serial import validate_session_id
from scripts.validate_esp32_camera import FrameObservation, analyze_observations
from scripts.validate_imu_stream import ImuSample, analyze_samples, parse_sample


IMU_FIELDS = list(ImuSample.__dataclass_fields__) + ["host_capture_monotonic_ns"]
CAMERA_FIELDS = [
    "device_id",
    "session_id",
    "sequence",
    "device_timestamp_us",
    "host_capture_monotonic_ns",
    "host_received_monotonic_ns",
    "width",
    "height",
    "jpeg_bytes",
    "motion_score",
    "relative_path",
]
MALFORMED_IMU_FIELDS = [
    "host_received_monotonic_ns",
    "category",
    "error",
    "raw_line",
]
SYNC_FIELDS = [
    "device_id",
    "request_id",
    "device_time_us",
    "host_send_monotonic_ns",
    "host_receive_monotonic_ns",
    "host_midpoint_monotonic_ns",
    "round_trip_ns",
]


@dataclass(frozen=True)
class CameraRecord:
    device_id: str
    session_id: str
    sequence: int
    device_timestamp_us: int
    host_received_monotonic_ns: int
    width: int
    height: int
    jpeg_bytes: int
    relative_path: str
    motion_score: float | None = None


@dataclass(frozen=True)
class MalformedImuRow:
    host_received_monotonic_ns: int
    category: str
    error: str
    raw_line: str


def default_session_id() -> str:
    return datetime.now(timezone.utc).strftime("fusion_%Y%m%dT%H%M%SZ")


def default_output_directory(session_id: str) -> Path:
    return Path("data") / "recordings" / session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-port", required=True, help="ESP32 IMU USB port")
    parser.add_argument("--camera-host", required=True, help="ESP32-CAM IP/host")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--target-imu-hz", type=float, default=50.0)
    parser.add_argument("--target-camera-fps", type=float, default=10.0)
    parser.add_argument("--sync-interval", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    activity = parser.add_mutually_exclusive_group()
    activity.add_argument(
        "--stationary",
        action="store_true",
        help="Apply stationary IMU checks during this transport test.",
    )
    activity.add_argument(
        "--motion-check",
        action="store_true",
        help=(
            "Require a shared visible motion event and validate remaining "
            "camera/IMU lag by cross-correlation."
        ),
    )
    return parser


def acknowledge_imu_session(
    connection: object,
    session_id: str,
    *,
    timeout_s: float,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> list[tuple[int, str]]:
    """Assign one session and drain startup output until the ESP32 acknowledges."""
    status_rows: list[tuple[int, str]] = []
    deadline_ns = clock_ns() + int(timeout_s * 1_000_000_000)
    next_request_ns = 0
    while clock_ns() < deadline_ns:
        now_ns = clock_ns()
        if now_ns >= next_request_ns:
            connection.write(f"SESSION,{session_id}\n".encode("ascii"))
            connection.flush()
            next_request_ns = now_ns + 2_000_000_000
        raw_line = connection.readline()
        received_ns = clock_ns()
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace").strip()
        if line.startswith("#"):
            status_rows.append((received_ns, line))
            print(line)
        if line == f"# session_set,id={session_id}":
            return status_rows
    raise RuntimeError(
        "IMU did not acknowledge SESSION; close Serial Monitor and verify the "
        "versioned firmware is running"
    )


def flush_imu_to_line_boundary(
    connection: object,
    *,
    timeout_s: float = 0.5,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> str:
    """Discard accumulated serial bytes and consume through one complete line."""
    if hasattr(connection, "reset_input_buffer"):
        connection.reset_input_buffer()
    deadline_ns = clock_ns() + int(timeout_s * 1_000_000_000)
    discarded = bytearray()
    while clock_ns() < deadline_ns:
        chunk = connection.readline()
        if not chunk:
            continue
        discarded.extend(chunk)
        if chunk.endswith(b"\n"):
            return discarded.decode("utf-8", errors="replace").strip()
    raise RuntimeError("IMU produced no complete line while aligning capture start")


def _next_sync_delay_ns(point_count: int, regular_interval_s: float) -> int:
    # Gather a rapid startup group, then keep sampling to estimate drift.
    interval_s = 0.25 if point_count < 5 else regular_interval_s
    return int(interval_s * 1_000_000_000)


def read_imu_until(
    connection: object,
    *,
    session_id: str,
    deadline_ns: int,
    regular_sync_interval_s: float,
    samples: list[ImuSample],
    sync_points: list[SyncObservation],
    status_rows: list[tuple[int, str]],
    malformed_rows: list[MalformedImuRow],
    errors: "queue.SimpleQueue[str]",
    stop_event: threading.Event,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> None:
    pending_sync: dict[str, int] = {}
    request_number = 0
    next_sync_ns = clock_ns()
    try:
        while clock_ns() < deadline_ns and not stop_event.is_set():
            now_ns = clock_ns()
            if now_ns >= next_sync_ns:
                request_number += 1
                request_id = f"i{request_number:06d}"
                sent_ns = clock_ns()
                connection.write(f"SYNC,{request_id}\n".encode("ascii"))
                connection.flush()
                pending_sync[request_id] = sent_ns
                next_sync_ns = sent_ns + _next_sync_delay_ns(
                    len(sync_points), regular_sync_interval_s
                )

            raw_line = connection.readline()
            received_ns = clock_ns()
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("#"):
                status_rows.append((received_ns, line))
                try:
                    response = parse_imu_sync_response(line)
                except (ValueError, OverflowError) as error:
                    malformed_rows.append(
                        MalformedImuRow(
                            host_received_monotonic_ns=received_ns,
                            category="invalid_sync_response",
                            error=str(error),
                            raw_line=line,
                        )
                    )
                    continue
                if response is not None:
                    request_id, device_time_us = response
                    sent_ns = pending_sync.pop(request_id, None)
                    if sent_ns is not None:
                        sync_points.append(
                            SyncObservation(
                                device_id="imu01",
                                request_id=request_id,
                                device_time_us=device_time_us,
                                host_send_ns=sent_ns,
                                host_receive_ns=received_ns,
                            )
                        )
                continue
            try:
                sample = parse_sample(line, received_ns)
            except (ValueError, OverflowError) as error:
                malformed_rows.append(
                    MalformedImuRow(
                        host_received_monotonic_ns=received_ns,
                        category="sample_parse_error",
                        error=str(error),
                        raw_line=line,
                    )
                )
                continue
            if sample is None:
                continue
            if (
                sample.schema_version != 1
                or sample.session_id != session_id
                or sample.device_id != "imu01"
            ):
                malformed_rows.append(
                    MalformedImuRow(
                        host_received_monotonic_ns=received_ns,
                        category="identity_mismatch",
                        error="expected schema 1, imu01, and the active session",
                        raw_line=line,
                    )
                )
                continue
            samples.append(sample)
    except Exception as error:
        errors.put(f"IMU reader failed: {error}")
        stop_event.set()


def read_camera_until(
    camera: TimestampedMjpegStream,
    *,
    session_id: str,
    frame_directory: Path,
    deadline_ns: int,
    records: list[CameraRecord],
    errors: "queue.SimpleQueue[str]",
    stop_event: threading.Event,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> None:
    previous_gray = None
    try:
        while clock_ns() < deadline_ns and not stop_event.is_set():
            frame = camera.read()
            if (
                frame.device_id != "cam01"
                or frame.session_id != session_id
                or frame.sequence is None
                or frame.device_timestamp_us is None
                or frame.jpeg_bytes is None
            ):
                raise CameraConnectionError(
                    "frame lacks the expected device/session/timestamp headers; "
                    "upload the Step 4 ESP32-CAM firmware"
                )
            height, width = frame.image.shape[:2]
            current_gray = frame.image.astype("float32").mean(axis=2)
            motion_score = None
            if previous_gray is not None and previous_gray.shape == current_gray.shape:
                motion_score = float(abs(current_gray - previous_gray).mean() / 255.0)
            previous_gray = current_gray
            relative_path = Path("frames") / f"{frame.sequence:010d}.jpg"
            save_original_jpeg(frame_directory.parent / relative_path, frame.jpeg_bytes)
            records.append(
                CameraRecord(
                    device_id=frame.device_id,
                    session_id=frame.session_id,
                    sequence=frame.sequence,
                    device_timestamp_us=frame.device_timestamp_us,
                    host_received_monotonic_ns=int(
                        round(frame.captured_at * 1_000_000_000)
                    ),
                    width=width,
                    height=height,
                    jpeg_bytes=len(frame.jpeg_bytes),
                    relative_path=relative_path.as_posix(),
                    motion_score=motion_score,
                )
            )
    except Exception as error:
        errors.put(f"camera reader failed: {error}")
        stop_event.set()


def sync_camera_until(
    control: CameraControlClient,
    *,
    deadline_ns: int,
    regular_sync_interval_s: float,
    sync_points: list[SyncObservation],
    errors: "queue.SimpleQueue[str]",
    stop_event: threading.Event,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> None:
    request_number = 0
    next_sync_ns = clock_ns()
    consecutive_errors = 0
    while clock_ns() < deadline_ns and not stop_event.is_set():
        now_ns = clock_ns()
        if now_ns < next_sync_ns:
            stop_event.wait(min((next_sync_ns - now_ns) / 1e9, 0.1))
            continue
        request_number += 1
        try:
            sync_points.append(control.sync(f"c{request_number:06d}"))
            consecutive_errors = 0
        except CameraControlError as error:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                errors.put(f"camera clock sync failed three times: {error}")
                stop_event.set()
                return
        next_sync_ns = clock_ns() + _next_sync_delay_ns(
            len(sync_points), regular_sync_interval_s
        )


def write_imu_manifest(
    path: Path, samples: list[ImuSample], mapping: ClockMapping | None
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=IMU_FIELDS)
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            row["host_capture_monotonic_ns"] = (
                mapping.map_device_us(sample.device_timestamp_us)
                if mapping is not None and sample.device_timestamp_us is not None
                else ""
            )
            writer.writerow(row)


def write_camera_manifest(
    path: Path, records: list[CameraRecord], mapping: ClockMapping | None
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAMERA_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["host_capture_monotonic_ns"] = (
                mapping.map_device_us(record.device_timestamp_us)
                if mapping is not None
                else ""
            )
            writer.writerow(row)


def write_sync_manifest(path: Path, points: list[SyncObservation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SYNC_FIELDS)
        writer.writeheader()
        for point in sorted(points, key=lambda item: item.host_send_ns):
            writer.writerow(
                {
                    "device_id": point.device_id,
                    "request_id": point.request_id,
                    "device_time_us": point.device_time_us,
                    "host_send_monotonic_ns": point.host_send_ns,
                    "host_receive_monotonic_ns": point.host_receive_ns,
                    "host_midpoint_monotonic_ns": point.host_midpoint_ns,
                    "round_trip_ns": point.rtt_ns,
                }
            )


def write_malformed_imu_rows(path: Path, rows: list[MalformedImuRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MALFORMED_IMU_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fit_or_none(points: list[SyncObservation]) -> ClockMapping | None:
    try:
        return fit_affine_clock(points)
    except ValueError:
        return None


def build_combined_report(
    *,
    imu_samples: list[ImuSample],
    invalid_imu_rows: int,
    camera_records: list[CameraRecord],
    imu_mapping: ClockMapping | None,
    camera_mapping: ClockMapping | None,
    duration_s: float,
    target_imu_hz: float,
    target_camera_fps: float,
    stationary: bool,
    motion_check: bool,
    camera_health: dict[str, object] | None,
    errors: list[str],
) -> dict[str, object]:
    imu_validation = analyze_samples(
        imu_samples,
        invalid_rows=invalid_imu_rows,
        target_hz=target_imu_hz,
        stationary=stationary,
    )
    observations = [
        FrameObservation(
            sequence=item.sequence,
            device_timestamp_us=item.device_timestamp_us,
            host_received_s=item.host_received_monotonic_ns / 1e9,
            width=item.width,
            height=item.height,
        )
        for item in camera_records
    ]
    camera_validation = analyze_observations(
        observations,
        requested_duration_s=duration_s,
        target_fps=target_camera_fps,
        health=camera_health,
    )

    clock_reports: dict[str, object] = {}
    clock_checks: dict[str, bool] = {}
    for name, mapping in (("imu", imu_mapping), ("camera", camera_mapping)):
        if mapping is None:
            clock_reports[name] = {"result": "FAIL", "reason": "no clock fit"}
            clock_checks[f"{name}_clock_fit_available"] = False
            continue
        checks = clock_mapping_passes(mapping)
        clock_reports[name] = {
            "result": "PASS" if all(checks.values()) else "FAIL",
            "mapping": mapping.report(),
            "checks": checks,
        }
        clock_checks.update({f"{name}_{key}": value for key, value in checks.items()})

    alignment: dict[str, object]
    nearest_diagnostic: dict[str, object]
    latency: dict[str, object] = {}
    transport_checks: dict[str, bool] = {}
    if imu_mapping is not None and camera_mapping is not None:
        imu_capture_ns = [
            imu_mapping.map_device_us(int(item.device_timestamp_us))
            for item in imu_samples
            if item.device_timestamp_us is not None
        ]
        camera_capture_ns = [
            camera_mapping.map_device_us(item.device_timestamp_us)
            for item in camera_records
        ]
        nearest_diagnostic = nearest_alignment_report(
            imu_capture_ns, camera_capture_ns
        )
        if motion_check:
            imu_motion_points = [
                MotionPoint(
                    host_time_ns=imu_mapping.map_device_us(
                        int(item.device_timestamp_us)
                    ),
                    score=(
                        (item.gx_dps**2 + item.gy_dps**2 + item.gz_dps**2) ** 0.5
                        + 50.0
                        * abs(
                            (item.ax_g**2 + item.ay_g**2 + item.az_g**2) ** 0.5
                            - 1.0
                        )
                    ),
                )
                for item in imu_samples
                if item.device_timestamp_us is not None
            ]
            camera_motion_points = [
                MotionPoint(
                    host_time_ns=camera_mapping.map_device_us(
                        item.device_timestamp_us
                    ),
                    score=float(item.motion_score),
                )
                for item in camera_records
                if item.motion_score is not None
            ]
            alignment = motion_alignment_report(
                imu_motion_points, camera_motion_points
            )
        else:
            alignment = {
                "result": "NOT_RUN",
                "method": "shared_motion_cross_correlation",
                "reason": (
                    "Step 4 synchronization acceptance requires --motion-check"
                ),
            }
        latency = {
            "imu_ms": transport_latency_report(
                [item.host_monotonic_ns for item in imu_samples], imu_capture_ns
            ),
            "camera_ms": transport_latency_report(
                [item.host_received_monotonic_ns for item in camera_records],
                camera_capture_ns,
            ),
        }
        imu_latency = latency["imu_ms"]
        camera_latency = latency["camera_ms"]
        transport_checks = {
            "imu_latency_p95_at_most_50_ms": imu_latency.get("p95", float("inf"))
            <= 50.0,
            "imu_latency_max_at_most_250_ms": imu_latency.get("max", float("inf"))
            <= 250.0,
            "camera_latency_p95_at_most_500_ms": camera_latency.get(
                "p95", float("inf")
            )
            <= 500.0,
            "camera_latency_max_at_most_1000_ms": camera_latency.get(
                "max", float("inf")
            )
            <= 1000.0,
        }
    else:
        alignment = {"result": "FAIL", "reason": "clock mapping unavailable"}
        nearest_diagnostic = {
            "result": "NOT_AVAILABLE",
            "reason": "clock mapping unavailable",
        }

    overall_pass = (
        not errors
        and imu_validation.get("result") == "PASS"
        and camera_validation.get("result") == "PASS"
        and bool(clock_checks)
        and all(clock_checks.values())
        and alignment.get("result") == "PASS"
        and bool(transport_checks)
        and all(transport_checks.values())
    )
    return {
        "result": "PASS" if overall_pass else "FAIL",
        "errors": errors,
        "imu": imu_validation,
        "camera": camera_validation,
        "clock_sync": clock_reports,
        "alignment": alignment,
        "nearest_sampling_diagnostic": nearest_diagnostic,
        "transport_latency": latency,
        "transport_checks": transport_checks,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0 or args.sync_interval <= 0:
        raise SystemExit("--duration and --sync-interval must be positive")
    try:
        session_id = validate_session_id(args.session_id or default_session_id())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_directory = args.output or default_output_directory(session_id)
    if output_directory.exists():
        raise SystemExit(f"output directory already exists: {output_directory}")

    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        print("pyserial is missing; install requirements.txt", file=sys.stderr)
        return 2

    control = CameraControlClient(args.camera_host)
    source = esp32_stream_url(args.camera_host)
    imu_samples: list[ImuSample] = []
    camera_records: list[CameraRecord] = []
    imu_sync_points: list[SyncObservation] = []
    camera_sync_points: list[SyncObservation] = []
    status_rows: list[tuple[int, str]] = []
    malformed_imu_rows: list[MalformedImuRow] = []
    worker_errors: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    stop_event = threading.Event()
    session_started_utc = datetime.now(timezone.utc).isoformat()

    print(f"Assigning shared session {session_id} to ESP32-CAM...")
    try:
        control.set_session(session_id)
    except CameraControlError as error:
        control.close()
        print(
            f"Camera control setup failed: {error}. Upload the Step 4 camera "
            "firmware and verify http://<camera-ip>/health.",
            file=sys.stderr,
        )
        return 2

    try:
        with serial.Serial(args.imu_port, args.baud, timeout=0.1) as connection:
            print(f"Opening IMU serial: {args.imu_port} @ {args.baud}")
            status_rows.extend(
                acknowledge_imu_session(
                    connection, session_id, timeout_s=args.startup_timeout
                )
            )
            output_directory.mkdir(parents=True, exist_ok=False)
            frame_directory = output_directory / "frames"
            frame_directory.mkdir()

            print(f"Opening persistent camera stream: {source}")
            with TimestampedMjpegStream(source) as camera:
                camera.read()  # Warm the MJPEG connection; do not record this frame.
                flush_imu_to_line_boundary(connection)
                started_ns = time.monotonic_ns()
                status_rows.append(
                    (started_ns, "# host_capture_start,serial_buffer_aligned=true")
                )
                deadline_ns = started_ns + int(args.duration * 1_000_000_000)
                workers = [
                    threading.Thread(
                        target=read_imu_until,
                        kwargs={
                            "connection": connection,
                            "session_id": session_id,
                            "deadline_ns": deadline_ns,
                            "regular_sync_interval_s": args.sync_interval,
                            "samples": imu_samples,
                            "sync_points": imu_sync_points,
                            "status_rows": status_rows,
                            "malformed_rows": malformed_imu_rows,
                            "errors": worker_errors,
                            "stop_event": stop_event,
                        },
                        name="imu-reader",
                    ),
                    threading.Thread(
                        target=read_camera_until,
                        kwargs={
                            "camera": camera,
                            "session_id": session_id,
                            "frame_directory": frame_directory,
                            "deadline_ns": deadline_ns,
                            "records": camera_records,
                            "errors": worker_errors,
                            "stop_event": stop_event,
                        },
                        name="camera-reader",
                    ),
                    threading.Thread(
                        target=sync_camera_until,
                        kwargs={
                            "control": control,
                            "deadline_ns": deadline_ns,
                            "regular_sync_interval_s": args.sync_interval,
                            "sync_points": camera_sync_points,
                            "errors": worker_errors,
                            "stop_event": stop_event,
                        },
                        name="camera-clock-sync",
                    ),
                ]
                print(
                    f"Capturing both devices for {args.duration:.0f} s; "
                    "do not open Serial Monitor or the camera stream elsewhere..."
                )
                if args.motion_check:
                    print(
                        "MOTION CHECK: after 10 seconds, perform three distinct "
                        "sharp side-to-side movements while the IMU remains "
                        "clearly visible to the camera."
                    )
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
                elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
    except (OSError, serial.SerialException, RuntimeError) as error:
        control.close()
        print(f"Combined recording setup failed: {error}", file=sys.stderr)
        return 2

    errors: list[str] = []
    while not worker_errors.empty():
        errors.append(worker_errors.get())
    try:
        camera_health = control.health()
    except CameraControlError as error:
        camera_health = None
        errors.append(f"camera health request failed: {error}")
    finally:
        control.close()

    imu_mapping = _fit_or_none(imu_sync_points)
    camera_mapping = _fit_or_none(camera_sync_points)
    write_imu_manifest(output_directory / "imu.csv", imu_samples, imu_mapping)
    write_camera_manifest(
        output_directory / "camera_manifest.csv", camera_records, camera_mapping
    )
    write_sync_manifest(
        output_directory / "clock_sync.csv",
        imu_sync_points + camera_sync_points,
    )
    write_malformed_imu_rows(
        output_directory / "malformed_imu.csv", malformed_imu_rows
    )
    with (output_directory / "device_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["host_received_monotonic_ns", "message"])
        writer.writerows(status_rows)

    validation = build_combined_report(
        imu_samples=imu_samples,
        invalid_imu_rows=len(malformed_imu_rows),
        camera_records=camera_records,
        imu_mapping=imu_mapping,
        camera_mapping=camera_mapping,
        duration_s=args.duration,
        target_imu_hz=args.target_imu_hz,
        target_camera_fps=args.target_camera_fps,
        stationary=args.stationary,
        motion_check=args.motion_check,
        camera_health=camera_health,
        errors=errors,
    )
    session = {
        "schema_version": 1,
        "session_id": session_id,
        "session_started_utc": session_started_utc,
        "requested_duration_s": args.duration,
        "acceptance_mode": "shared_motion" if args.motion_check else "not_run",
        "host_elapsed_s": round(elapsed_s, 4),
        "sources": {
            "imu": {"device_id": "imu01", "port": args.imu_port, "baud": args.baud},
            "camera": {"device_id": "cam01", "stream": source},
        },
        "artifacts": {
            "imu": "imu.csv",
            "camera": "camera_manifest.csv",
            "camera_frames": "frames",
            "clock_sync": "clock_sync.csv",
            "device_status": "device_status.csv",
            "malformed_imu": "malformed_imu.csv",
        },
        "counts": {
            "imu_samples": len(imu_samples),
            "camera_frames": len(camera_records),
            "imu_sync_points": len(imu_sync_points),
            "camera_sync_points": len(camera_sync_points),
            "malformed_imu_rows": len(malformed_imu_rows),
        },
        "validation": validation,
    }
    (output_directory / "session.json").write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved synchronized recording: {output_directory.resolve()}")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
