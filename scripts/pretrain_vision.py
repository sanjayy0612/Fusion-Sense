"""Stage 1 — pretrain the vision encoder on real pose sequences.

Real mode expects labeled video clips laid out as:
  data/raw/vision_videos/<label>/<clip>.mp4
Each clip -> MediaPipe pose -> (t_vis, vision_dv) windows.
Set CFG.vision_dv = 99 for full pose (33 landmarks x 3).

  python scripts/pretrain_vision.py                       # real videos
  python scripts/pretrain_vision.py --sim                 # simulator fallback
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.config import CFG, DATA_ROOT
from fusionsense.train.pretrain import pretrain_encoder
from fusionsense.data.registry import get_pretrain_data


def load_real_videos():
    from fusionsense.data.vision_extractor import video_to_pose_sequence, pose_sequence_to_windows
    root = DATA_ROOT / "vision_videos"
    clips = sorted(glob.glob(str(root / "*" / "*.mp4")))
    if not clips:
        raise FileNotFoundError(f"No videos under {root} — see docs/DATASETS.md")
    X, labels = [], []
    for c in clips:
        label = os.path.basename(os.path.dirname(c))
        for w in pose_sequence_to_windows(video_to_pose_sequence(c), CFG):
            X.append(w); labels.append(label)
    uniq = sorted(set(labels)); m = {l: i for i, l in enumerate(uniq)}
    return np.stack(X).astype(np.float32), np.array([m[l] for l in labels]), len(uniq)


if __name__ == "__main__":
    sim = "--sim" in sys.argv
    try:
        if sim:
            raise FileNotFoundError("forced sim")
        X, y, n_classes = load_real_videos()
    except (FileNotFoundError, ImportError) as e:
        if not sim:
            raise
        print(f"[vision] {e}; using simulator (smoke test)")
        X, y, n_classes = get_pretrain_data("vision", allow_sim_fallback=True)
    print(f"vision pretrain: X={X.shape}, classes={n_classes}")
    pretrain_encoder(X, y, n_classes, in_ch=CFG.vision_dv, out_path="checkpoints/enc_vis.pt")
