"""Download the official MediaPipe Pose Landmarker Lite model bundle."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.vision_extractor import DEFAULT_MODEL_PATH, POSE_MODEL_URL


def download(destination: Path, force: bool = False) -> Path:
    destination = destination.resolve()
    if destination.exists() and not force:
        print(f"Pose model already exists: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(POSE_MODEL_URL, timeout=60) as response:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix="pose-model-", delete=False
        ) as temporary:
            shutil.copyfileobj(response, temporary)
            temporary_path = Path(temporary.name)
    if temporary_path.stat().st_size < 1_000_000:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded pose model is unexpectedly small")
    temporary_path.replace(destination)
    print(f"Downloaded MediaPipe Pose model: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.force)
