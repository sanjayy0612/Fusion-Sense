"""Build and validate Step 5 FusionWindow tensors from a Step 4 session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.data.recorded_session import (  # noqa: E402
    build_recorded_session_windows,
    summarize_session_windows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--force-pose", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--windows-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    if args.stride_seconds <= 0:
        raise SystemExit("--stride-seconds must be positive")
    if args.force_pose and args.skip_pose:
        raise SystemExit("--force-pose and --skip-pose are mutually exclusive")

    session_dir = args.session_dir.resolve()
    pose_cache = session_dir / "pose_cache.npz"
    if args.force_pose and pose_cache.exists():
        pose_cache.unlink()
    outputs, metadata = build_recorded_session_windows(
        session_dir,
        stride_seconds=args.stride_seconds,
        extract_poses=not args.skip_pose,
    )
    report = summarize_session_windows(outputs, metadata)
    windows_output = args.windows_output or (session_dir / "step5_windows.npz")
    report_output = args.report_output or (session_dir / "step5_validation.json")

    np.savez_compressed(
        windows_output,
        schema_version=np.array([1], dtype=np.int32),
        t_start=np.array([item.window.t_start for item in outputs], dtype=np.float64),
        imu=np.stack([item.window.imu for item in outputs]),
        radar=np.stack([item.window.radar for item in outputs]),
        vision=np.stack([item.window.vision for item in outputs]),
        imu_valid=np.array([item.window.imu_valid for item in outputs], dtype=bool),
        radar_valid=np.array([item.window.radar_valid for item in outputs], dtype=bool),
        vision_valid=np.array(
            [item.window.vision_valid for item in outputs], dtype=bool
        ),
        imu_health=np.array(
            [item.window.imu_health for item in outputs], dtype=np.float32
        ),
        image_quality=np.array(
            [item.window.image_quality for item in outputs], dtype=np.float32
        ),
    )
    payload = {
        "schema_version": 1,
        "step": 5,
        "session": metadata,
        "validation": report,
        "artifacts": {
            "pose_cache": metadata.get("pose_cache"),
            "windows": str(windows_output.resolve()),
        },
        "window_diagnostics": [item.diagnostics for item in outputs],
    }
    report_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": report["result"],
                "windows": report["windows"],
                "imu_valid_windows": report["validity"]["imu_valid_windows"],
                "vision_valid_windows": report["validity"][
                    "vision_valid_windows"
                ],
                "windows_output": str(windows_output.resolve()),
                "report_output": str(report_output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
