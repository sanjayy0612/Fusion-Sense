import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.data.recorded_session import (
    build_recorded_session_windows,
    summarize_session_windows,
)


def test_recorded_session_builds_synchronized_canonical_windows():
    temporary = tempfile.TemporaryDirectory()
    session = Path(temporary.name) / "session"
    frames = session / "frames"
    frames.mkdir(parents=True)
    base_ns = 500_000_000_000_000

    with (session / "imu.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "host_monotonic_ns", "t_ms", "ax_g", "ay_g", "az_g", "gx_dps",
            "gy_dps", "gz_dps", "schema_version", "device_id", "session_id",
            "sequence", "device_timestamp_us", "host_capture_monotonic_ns",
        ])
        for index in range(151):
            timestamp = base_ns + index * 20_000_000
            writer.writerow([
                timestamp, index * 20, 0, 0, 1, 1, 2, 3, 1, "imu01", "test",
                index, index * 20_000, timestamp,
            ])

    camera_timestamps = np.array(
        [base_ns + index * 100_000_000 for index in range(31)], dtype=np.int64
    )
    with (session / "camera_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "device_id", "session_id", "sequence", "device_timestamp_us",
            "host_capture_monotonic_ns", "host_received_monotonic_ns", "width",
            "height", "jpeg_bytes", "motion_score", "relative_path",
        ])
        for index, timestamp in enumerate(camera_timestamps):
            relative = f"frames/{index:010d}.jpg"
            (session / relative).write_bytes(b"test")
            writer.writerow([
                "cam01", "test", index, index * 100_000, timestamp, timestamp,
                320, 240, 4, 0, relative,
            ])

    np.savez_compressed(
        session / "pose_cache.npz",
        version=np.array([1], np.int32),
        host_capture_monotonic_ns=camera_timestamps,
        sequence=np.arange(31, dtype=np.int64),
        landmarks=np.full((31, CFG.vision_dv), 0.25, dtype=np.float32),
        valid=np.ones(31, dtype=bool),
        image_quality=np.full(31, 0.8, dtype=np.float32),
    )
    (session / "session.json").write_text(
        json.dumps({"session_id": "test", "validation": {"result": "PASS"}}),
        encoding="utf-8",
    )

    outputs, metadata = build_recorded_session_windows(session, extract_poses=False)
    assert len(outputs) == 2
    assert metadata["imu_input_validation"]["valid"]
    assert metadata["imu_input_validation"]["conversion"]["acceleration_scale"] == 9.80665
    first = outputs[0]
    assert first.window.imu.shape == (CFG.t_imu, CFG.imu_ch)
    assert first.window.vision.shape == (CFG.t_vis, CFG.vision_dv)
    assert np.allclose(first.window.imu[:, 2], 9.80665)
    assert np.allclose(first.window.imu[:, 3:], [1, 2, 3])
    assert first.window.imu_valid
    assert first.window.vision_valid
    assert first.diagnostics["camera"]["nearest_source_frames"] == CFG.t_vis
    report = summarize_session_windows(outputs, metadata)
    assert report["result"] == "PASS"
    assert report["validity"]["imu_valid_windows"] == 2
    assert report["validity"]["vision_valid_windows"] == 2
    temporary.cleanup()


if __name__ == "__main__":
    test_recorded_session_builds_synchronized_canonical_windows()
    print("ALL RECORDED SESSION TESTS PASSED")
