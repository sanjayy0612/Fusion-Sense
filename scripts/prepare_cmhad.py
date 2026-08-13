"""Prepare C-MHAD transition recordings into aligned MediaPipe+IMU windows."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.cmhad_loader import DEFAULT_CACHE, prepare_cmhad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw/cmhad",
        help="extracted C-MHAD folder or its TransitionMovementsApplication folder",
    )
    parser.add_argument("--output", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--max-subjects",
        type=int,
        help="debug-only limit; omit this for real training",
    )
    args = parser.parse_args()
    prepare_cmhad(args.raw_root, args.output, max_subjects=args.max_subjects)


if __name__ == "__main__":
    main()
