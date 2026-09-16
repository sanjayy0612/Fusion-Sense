"""Build non-overlapping continuous non-fall windows for Step 8 false alerts/hour."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import CFG  # noqa: E402
from fusionsense.contract import FusionWindow, LABEL2ID  # noqa: E402
from fusionsense.data.cmhad_loader import (  # noqa: E402
    EXPECTED_STREAM_SAMPLES,
    crop_stream_window,
    find_transition_root,
    load_cmhad_windows,
    read_annotations,
    read_imu_stream,
    save_cmhad_windows,
)
from fusionsense.data.vision_extractor import video_to_aligned_pose_windows  # noqa: E402


def overlaps_or_follows_fall(
    start: float, end: float, events: list[dict], guard: float
) -> bool:
    """Reject the fall transition and the post-fall state through recording end.

    A subject remaining on the floor after the annotated transition is not valid
    negative monitoring time. Counting alerts there as false alerts would bias the
    safety evaluation against correct detections.
    """
    fall_id = LABEL2ID["stand_to_fall"]
    return any(
        event["label"] == fall_id
        and end > event["start"] - guard
        for event in events
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--fall-guard-seconds", type=float, default=1.0)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=None,
        help="Filter an older background cache after exclusion-rule changes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cmhad_step8_subject3_background.npz"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.stride_seconds < CFG.window_seconds:
        raise SystemExit("Background windows must not overlap; stride must be >= 2 s")
    if args.fall_guard_seconds < 0:
        raise SystemExit("fall guard must be non-negative")

    root = find_transition_root(args.raw_root)
    subject_dir = root / f"Subject{args.subject}"
    annotation_path = subject_dir / f"ActionOfInterestTraSubject{args.subject}.xlsx"
    events = read_annotations(annotation_path)
    by_recording: dict[int, list[dict]] = {}
    for event in events:
        by_recording.setdefault(event["recording"], []).append(event)

    if args.source_cache is not None:
        cached = load_cmhad_windows(args.source_cache)
        windows = []
        recording_counts = {f"tr{recording}": 0 for recording in range(1, 11)}
        newly_excluded = 0
        for window in cached:
            recording = int(window.recording_id.rsplit("tr", 1)[1])
            if overlaps_or_follows_fall(
                window.t_start,
                window.t_start + CFG.window_seconds,
                by_recording.get(recording, []),
                args.fall_guard_seconds,
            ):
                newly_excluded += 1
                continue
            windows.append(window)
            recording_counts[f"tr{recording}"] += 1
        source_metadata_path = args.source_cache.with_suffix(".json")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        previously_excluded = int(
            source_metadata.get("excluded_fall_overlap_windows", 0)
            + source_metadata.get("excluded_at_or_after_fall_windows", 0)
        )
        save_cmhad_windows(windows, args.output, CFG)
        metadata = {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "C-MHAD continuous TransitionMovementsApplication",
            "derived_from": str(args.source_cache),
            "subject": f"Subject{args.subject}",
            "window_seconds": CFG.window_seconds,
            "stride_seconds": args.stride_seconds,
            "fall_guard_seconds": args.fall_guard_seconds,
            "windows": len(windows),
            "negative_hours": len(windows) * CFG.window_seconds / 3600.0,
            "vision_valid_windows": sum(window.vision_valid for window in windows),
            "excluded_at_or_after_fall_windows": previously_excluded + newly_excluded,
            "newly_excluded_post_fall_windows": newly_excluded,
            "recording_counts": recording_counts,
            "label_semantics": (
                "Every saved window ends before the guarded fall onset; fall and "
                "post-fall states are excluded through recording end"
            ),
        }
        metadata_path = args.output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"saved {len(windows)} synchronized windows -> {args.output}")
        print(json.dumps(metadata, indent=2))
        return 0

    stream_seconds = EXPECTED_STREAM_SAMPLES / CFG.imu_hz
    windows: list[FusionWindow] = []
    recording_counts: dict[str, int] = {}
    excluded_fall_windows = 0
    for recording in range(1, 11):
        recording_events = by_recording.get(recording, [])
        starts = np.arange(
            CFG.window_seconds,
            stream_seconds - CFG.window_seconds + 1e-9,
            args.stride_seconds,
            dtype=np.float64,
        ).tolist()
        selected = []
        for start in starts:
            if overlaps_or_follows_fall(
                start,
                start + CFG.window_seconds,
                recording_events,
                args.fall_guard_seconds,
            ):
                excluded_fall_windows += 1
            else:
                selected.append(float(start))

        imu_path = (
            subject_dir / "InertialData" / f"inertial_sub{args.subject}_tr{recording}.csv"
        )
        video_path = (
            subject_dir / "VideoData" / f"video_sub{args.subject}_tr{recording}.avi"
        )
        imu_stream = read_imu_stream(imu_path)
        pose_windows, valid_ratios, qualities = video_to_aligned_pose_windows(
            str(video_path), selected, cfg=CFG
        )
        for index, start in enumerate(selected):
            windows.append(
                FusionWindow(
                    t_start=start,
                    imu=crop_stream_window(imu_stream, start, CFG),
                    radar=np.zeros((CFG.t_radar, CFG.radar_k), dtype=np.float32),
                    vision=pose_windows[index],
                    imu_valid=True,
                    radar_valid=False,
                    vision_valid=valid_ratios[index] >= 0.5,
                    radar_energy=0.0,
                    image_quality=qualities[index],
                    imu_health=1.0,
                    label=0,
                    subject_id=f"Subject{args.subject}",
                    recording_id=f"Subject{args.subject}/tr{recording}",
                )
            )
        recording_counts[f"tr{recording}"] = len(selected)
        print(
            f"Subject{args.subject}/tr{recording}: "
            f"background_windows={len(selected)}, total={len(windows)}"
        )

    if not windows:
        raise RuntimeError("No background windows were selected")
    save_cmhad_windows(windows, args.output, CFG)
    metadata = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "C-MHAD continuous TransitionMovementsApplication",
        "subject": f"Subject{args.subject}",
        "window_seconds": CFG.window_seconds,
        "stride_seconds": args.stride_seconds,
        "fall_guard_seconds": args.fall_guard_seconds,
        "windows": len(windows),
        "negative_hours": len(windows) * CFG.window_seconds / 3600.0,
        "vision_valid_windows": sum(window.vision_valid for window in windows),
        "excluded_at_or_after_fall_windows": excluded_fall_windows,
        "recording_counts": recording_counts,
        "label_semantics": (
            "Every saved window ends before the guarded fall onset; fall and "
            "post-fall states are excluded through recording end"
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
