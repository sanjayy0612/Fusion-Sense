"""Real vision features via MediaPipe Pose — turns a video into per-frame body
keypoint vectors (NOT raw pixels), for pretraining enc_vis and for the paired
loader.

Why pose keypoints: privacy-preserving (no identifiable image leaves the node),
compact, and directly encodes posture — exactly what the fusion model needs.

Output per frame: 33 landmarks x (x, y, z) = 99 dims.
  => set CFG.vision_dv = 99 when training on real pose (see DATASETS.md).

Requires: mediapipe, opencv-python (install only when using real video):
  pip install mediapipe opencv-python
"""
from __future__ import annotations

import numpy as np

from ..config import CFG
from .windowing import resample_to

N_LANDMARKS = 33
POSE_DIM = N_LANDMARKS * 3   # 99


def _lazy_imports():
    try:
        import cv2, mediapipe as mp        # noqa
        return cv2, mp
    except Exception as e:
        raise ImportError(
            "Real vision needs mediapipe + opencv-python. Install: "
            "pip install mediapipe opencv-python") from e


def video_to_pose_sequence(video_path: str) -> np.ndarray:
    """Return (num_frames, 99) pose sequence for a whole video. Missing
    detections are zero-filled."""
    cv2, mp = _lazy_imports()
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            vec = np.array([[p.x, p.y, p.z] for p in lm], np.float32).reshape(-1)
        else:
            vec = np.zeros(POSE_DIM, np.float32)
        seq.append(vec)
    cap.release(); pose.close()
    return np.stack(seq) if seq else np.zeros((1, POSE_DIM), np.float32)


def pose_sequence_to_windows(seq: np.ndarray, cfg=CFG, src_fps: float = 30.0):
    """Segment a (T, 99) pose sequence into (t_vis, vision_dv) windows.
    If vision_dv < 99 the vector is truncated (set vision_dv=99 for full pose)."""
    from .windowing import segment_recording
    seq = seq[:, :cfg.vision_dv] if seq.shape[1] >= cfg.vision_dv else \
        np.pad(seq, ((0, 0), (0, cfg.vision_dv - seq.shape[1])))
    return segment_recording(seq, src_fps, cfg.t_vis, cfg.window_seconds)
