"""Rebuild validation and mapped manifests for a recorded USB fusion session."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.clock_sync import (  # noqa: E402
    SyncObservation,
    fit_affine_clock,
    fit_one_way_clock_lower_envelope,
)
from scripts.record_fusion_session import (  # noqa: E402
    CameraRecord,
    build_combined_report,
    calibrate_one_way_camera_offset,
    write_camera_manifest,
    write_imu_manifest,
)
from scripts.validate_imu_stream import ImuSample  # noqa: E402


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def load_imu(path: Path) -> list[ImuSample]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        ImuSample(
            host_monotonic_ns=int(row["host_monotonic_ns"]),
            t_ms=int(row["t_ms"]),
            ax_g=float(row["ax_g"]),
            ay_g=float(row["ay_g"]),
            az_g=float(row["az_g"]),
            gx_dps=float(row["gx_dps"]),
            gy_dps=float(row["gy_dps"]),
            gz_dps=float(row["gz_dps"]),
            schema_version=_optional_int(row["schema_version"]),
            device_id=row["device_id"] or None,
            session_id=row["session_id"] or None,
            sequence=_optional_int(row["sequence"]),
            device_timestamp_us=_optional_int(row["device_timestamp_us"]),
        )
        for row in rows
    ]


def load_camera(path: Path) -> list[CameraRecord]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        CameraRecord(
            device_id=row["device_id"],
            session_id=row["session_id"],
            sequence=int(row["sequence"]),
            device_timestamp_us=int(row["device_timestamp_us"]),
            host_received_monotonic_ns=int(row["host_received_monotonic_ns"]),
            width=int(row["width"]),
            height=int(row["height"]),
            jpeg_bytes=int(row["jpeg_bytes"]),
            relative_path=row["relative_path"],
            capture_errors=int(row.get("capture_errors") or 0),
            motion_score=(
                float(row["motion_score"]) if row.get("motion_score") else None
            ),
        )
        for row in rows
    ]


def load_sync(path: Path) -> list[SyncObservation]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        SyncObservation(
            device_id=row["device_id"],
            request_id=row["request_id"],
            device_time_us=int(row["device_time_us"]),
            host_send_ns=int(row["host_send_monotonic_ns"]),
            host_receive_ns=int(row["host_receive_monotonic_ns"]),
        )
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_directory", type=Path)
    args = parser.parse_args()
    root = args.session_directory.resolve()
    session_path = root / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    camera_source = session.get("sources", {}).get("camera", {})
    if camera_source.get("transport") != "usb_serial":
        raise SystemExit("revalidation currently supports USB-serial camera sessions")

    imu_samples = load_imu(root / "imu.csv")
    camera_records = load_camera(root / "camera_manifest.csv")
    sync_points = load_sync(root / "clock_sync.csv")
    imu_points = [point for point in sync_points if point.device_id == "imu01"]
    camera_points = [point for point in sync_points if point.device_id == "cam01"]
    imu_mapping = fit_affine_clock(imu_points)
    camera_mapping = fit_one_way_clock_lower_envelope(camera_points)
    camera_mapping, calibration = calibrate_one_way_camera_offset(
        imu_samples, camera_records, imu_mapping, camera_mapping
    )

    malformed_path = root / "malformed_imu.csv"
    with malformed_path.open(newline="", encoding="utf-8") as stream:
        invalid_imu_rows = sum(1 for _ in csv.DictReader(stream))
    original_errors = list(session.get("validation", {}).get("errors", []))
    validation = build_combined_report(
        imu_samples=imu_samples,
        invalid_imu_rows=invalid_imu_rows,
        camera_records=camera_records,
        imu_mapping=imu_mapping,
        camera_mapping=camera_mapping,
        duration_s=float(session["requested_duration_s"]),
        target_imu_hz=50.0,
        target_camera_fps=10.0,
        stationary=False,
        motion_check=True,
        camera_health={
            "capture_errors": max(
                (record.capture_errors for record in camera_records), default=0
            )
        },
        errors=original_errors,
    )
    validation["camera_clock_offset_calibration"] = calibration

    write_imu_manifest(root / "imu.csv", imu_samples, imu_mapping)
    write_camera_manifest(root / "camera_manifest.csv", camera_records, camera_mapping)
    session["camera_clock_offset_calibration"] = calibration
    session["revalidated_utc"] = datetime.now(timezone.utc).isoformat()
    session["validation"] = validation
    session_path.write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
