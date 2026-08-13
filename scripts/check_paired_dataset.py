"""Check locally recorded camera + IMU trials before fusion training.

This is intentionally lightweight: it verifies structure and sample counts
without running MediaPipe pose extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import CFG, DATA_ROOT
from fusionsense.contract import ACTIVITIES


def _imu_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        required = {"ax", "ay", "az", "gx", "gy", "gz"}
        if not required.issubset({name.strip().lower() for name in header}):
            return -1
        return sum(1 for row in reader if len(row) >= 7)


def inspect_dataset(root: Path, minimum_trials: int = 2) -> tuple[list[str], list[str]]:
    messages = []
    problems = []
    counts = Counter()
    trials = sorted(path.parent for path in root.glob("*/*/*/imu.csv"))

    for trial in trials:
        subject, activity, trial_id = trial.relative_to(root).parts
        counts[activity] += 1
        imu_path = trial / "imu.csv"
        video_path = trial / "video.mp4"
        metadata_path = trial / "metadata.json"
        rows = _imu_rows(imu_path)
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                problems.append(f"{subject}/{activity}/{trial_id}: invalid metadata.json")
        else:
            problems.append(f"{subject}/{activity}/{trial_id}: missing metadata.json")

        seconds = float(metadata.get("duration_seconds", 0.0) or 0.0)
        expected = seconds * CFG.imu_hz
        if rows < 0:
            problems.append(f"{subject}/{activity}/{trial_id}: invalid IMU header")
        elif rows < max(CFG.t_imu, expected * 0.5):
            problems.append(
                f"{subject}/{activity}/{trial_id}: only {rows} IMU rows "
                f"for {seconds:.1f}s"
            )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            problems.append(f"{subject}/{activity}/{trial_id}: missing/empty video.mp4")
        messages.append(
            f"{subject}/{activity}/{trial_id}: {max(rows, 0)} IMU rows, "
            f"{int(metadata.get('video_frames', 0) or 0)} video frames"
        )

    if not trials:
        problems.append(f"No paired trials found under {root}")
    for activity in ACTIVITIES:
        if counts[activity] < minimum_trials:
            problems.append(
                f"{activity}: {counts[activity]} trial(s); need at least "
                f"{minimum_trials} for a leakage-safe train/validation split"
            )
    messages.append(
        "trial totals: " + ", ".join(f"{name}={counts[name]}" for name in ACTIVITIES)
    )
    return messages, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT / CFG.paired_dir)
    parser.add_argument("--minimum-trials", type=int, default=2)
    args = parser.parse_args()

    messages, problems = inspect_dataset(args.root, args.minimum_trials)
    for message in messages:
        print(message)
    if problems:
        print("\nNOT READY:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("\nREADY for: python scripts/train_fusion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
