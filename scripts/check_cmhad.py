"""Strictly validate extracted or prepared C-MHAD data before training."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.contract import ACTIVITIES, ID2LABEL
from fusionsense.data.cmhad_loader import (
    DEFAULT_CACHE,
    find_transition_root,
    load_cmhad_windows,
    read_annotations,
)


def subject_number(path: Path) -> int:
    match = re.fullmatch(r"Subject(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected subject folder: {path}")
    return int(match.group(1))


def validate_raw(raw_root, expected_subjects, errors):
    try:
        root = find_transition_root(raw_root)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        print(f"raw dataset: NOT READY ({exc})")
        return set()

    subjects = sorted(root.glob("Subject*"), key=subject_number)
    if expected_subjects is not None and len(subjects) != expected_subjects:
        errors.append(
            f"Expected {expected_subjects} subject folders, found {len(subjects)}"
        )

    annotation_count = 0
    names = set()
    for subject_dir in subjects:
        number = subject_number(subject_dir)
        names.add(subject_dir.name)
        annotation_path = subject_dir / f"ActionOfInterestTraSubject{number}.xlsx"
        if not annotation_path.is_file():
            errors.append(f"Missing annotation file: {annotation_path}")
            continue
        try:
            annotations = read_annotations(annotation_path)
        except Exception as exc:
            errors.append(f"Cannot read {annotation_path}: {exc}")
            continue
        annotation_count += len(annotations)
        labels = {row["label"] for row in annotations}
        if labels != set(range(len(ACTIVITIES))):
            errors.append(
                f"{subject_dir.name} does not contain all seven labels: {sorted(labels)}"
            )
        recordings = {row["recording"] for row in annotations}
        if recordings != set(range(1, 11)):
            errors.append(
                f"{subject_dir.name} should reference recordings 1..10; got "
                f"{sorted(recordings)}"
            )
        for recording in recordings:
            imu = subject_dir / "InertialData" / f"inertial_sub{number}_tr{recording}.csv"
            video = subject_dir / "VideoData" / f"video_sub{number}_tr{recording}.avi"
            if not imu.is_file():
                errors.append(f"Missing IMU file: {imu}")
            if not video.is_file():
                errors.append(f"Missing video file: {video}")

    print(f"raw dataset: {root}")
    print(f"subjects: {len(subjects)}; annotations: {annotation_count}")
    return names


def validate_cache(cache, expected_subjects, require_cache, raw_subjects, errors):
    try:
        windows = load_cmhad_windows(cache)
    except FileNotFoundError as exc:
        print(f"prepared cache: NOT READY ({exc})")
        if require_cache:
            errors.append(str(exc))
        return
    except Exception as exc:
        errors.append(f"Cannot read prepared cache {cache}: {exc}")
        return

    counts = Counter(ID2LABEL[window.label] for window in windows)
    cache_subjects = {window.subject_id for window in windows}
    print(f"prepared cache: {cache} ({len(windows)} windows)")
    print(f"prepared subjects: {', '.join(sorted(cache_subjects))}")
    for label in ACTIVITIES:
        print(f"  {label:<16} {counts.get(label, 0)}")

    if set(counts) != set(ACTIVITIES):
        errors.append(f"Prepared cache does not contain all labels: {sorted(counts)}")
    if expected_subjects is not None and len(cache_subjects) != expected_subjects:
        errors.append(
            f"Prepared cache should contain {expected_subjects} subjects; "
            f"found {len(cache_subjects)}"
        )
    if raw_subjects and cache_subjects != raw_subjects:
        errors.append(
            f"Raw/cache subject mismatch: raw={sorted(raw_subjects)}, "
            f"cache={sorted(cache_subjects)}"
        )

    invalid = sum(not window.vision_valid for window in windows)
    invalid_ratio = invalid / max(len(windows), 1)
    print(f"MediaPipe-invalid windows: {invalid}/{len(windows)} ({invalid_ratio:.1%})")
    if invalid_ratio > 0.2:
        errors.append(
            "More than 20% of prepared windows have invalid MediaPipe poses; "
            "inspect the videos before training"
        )

    for index, window in enumerate(windows):
        if window.imu.shape != (CFG.t_imu, CFG.imu_ch):
            errors.append(f"Window {index} has invalid IMU shape {window.imu.shape}")
            break
        if window.vision.shape != (CFG.t_vis, CFG.vision_dv):
            errors.append(f"Window {index} has invalid vision shape {window.vision.shape}")
            break
        if not np.isfinite(window.imu).all() or not np.isfinite(window.vision).all():
            errors.append(f"Window {index} contains NaN or infinite values")
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/cmhad")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--expected-subjects",
        type=int,
        choices=range(1, 13),
        metavar="N",
        help="fail unless raw data and prepared cache contain exactly N subjects",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="fail when the processed cache does not exist",
    )
    args = parser.parse_args()

    errors = []
    raw_subjects = validate_raw(args.raw_root, args.expected_subjects, errors)
    validate_cache(
        args.cache, args.expected_subjects, args.require_cache, raw_subjects, errors
    )
    if errors:
        print("\nC-MHAD VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("\nC-MHAD VALIDATION PASSED")


if __name__ == "__main__":
    main()
