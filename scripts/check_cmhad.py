"""Validate an extracted or prepared C-MHAD dataset without training."""
import argparse
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.contract import ID2LABEL
from fusionsense.data.cmhad_loader import (
    DEFAULT_CACHE,
    find_transition_root,
    load_cmhad_windows,
    read_annotations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/cmhad")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    args = parser.parse_args()

    try:
        root = find_transition_root(args.raw_root)
        subjects = sorted(root.glob("Subject*"))
        annotation_count = 0
        for subject_dir in subjects:
            number = subject_dir.name.replace("Subject", "")
            annotation_count += len(read_annotations(
                subject_dir / f"ActionOfInterestTraSubject{number}.xlsx"
            ))
        print(f"raw dataset: {root}")
        print(f"subjects: {len(subjects)}; annotations: {annotation_count}")
    except FileNotFoundError as exc:
        print(f"raw dataset: NOT READY ({exc})")

    try:
        windows = load_cmhad_windows(args.cache)
        counts = Counter(ID2LABEL[window.label] for window in windows)
        print(f"prepared cache: {args.cache} ({len(windows)} windows)")
        for label, count in sorted(counts.items()):
            print(f"  {label:<16} {count}")
        invalid = sum(not window.vision_valid for window in windows)
        print(f"MediaPipe-invalid windows: {invalid}/{len(windows)}")
    except FileNotFoundError as exc:
        print(f"prepared cache: NOT READY ({exc})")


if __name__ == "__main__":
    main()
