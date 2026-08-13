"""MediaPipe Pose features for live cameras and recorded videos.

Raw pixels stop at this module. Every successful pose becomes 33 normalized
landmarks x (x, y, z) = 99 float32 values, matching ``CFG.vision_dv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CFG

N_LANDMARKS = 33
POSE_DIM = N_LANDMARKS * 3
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PoseFrame:
    """One timestamped pose result ready for windowing and health reporting."""

    captured_at: float
    landmarks: np.ndarray
    valid: bool
    image_quality: float


def _lazy_imports():
    try:
        import cv2
        import mediapipe as mp
        return cv2, mp
    except Exception as exc:
        raise ImportError(
            "Camera pose extraction needs MediaPipe and OpenCV. Install: "
            "python -m pip install mediapipe opencv-python"
        ) from exc


def image_quality(frame: np.ndarray) -> float:
    """Return a cheap [0,1] brightness-and-sharpness health signal."""
    cv2, _ = _lazy_imports()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 250.0, 1.0)
    return float(np.clip(np.sqrt(brightness * sharpness), 0.0, 1.0))


class PoseExtractor:
    """Current MediaPipe Tasks adapter with a fixed FusionSense output.

    ``extract`` accepts a decoded BGR image and laptop-monotonic timestamp. It
    returns zeros with ``valid=False`` when no body is detected. Timestamps are
    made strictly increasing because MediaPipe video mode requires that
    invariant.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        cv2, mp = _lazy_imports()
        self._cv2 = cv2
        self._mp = mp
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found at {self.model_path}. Run "
                "`python scripts/download_pose_model.py` first."
            )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def extract(self, frame: np.ndarray, captured_at: float) -> PoseFrame:
        timestamp_ms = max(int(captured_at * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        quality = image_quality(frame)
        if not result.pose_landmarks:
            return PoseFrame(
                captured_at=captured_at,
                landmarks=np.zeros(POSE_DIM, dtype=np.float32),
                valid=False,
                image_quality=quality,
            )
        landmarks = result.pose_landmarks[0]
        vector = np.array(
            [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
        ).reshape(-1)
        if vector.shape != (POSE_DIM,):
            raise RuntimeError(f"Expected {POSE_DIM} pose values, got {vector.shape}")
        return PoseFrame(
            captured_at=captured_at,
            landmarks=vector,
            valid=True,
            image_quality=quality,
        )

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def video_to_pose_sequence(
    video_path: str, model_path: str | Path = DEFAULT_MODEL_PATH
) -> np.ndarray:
    """Return ``(num_frames, 99)`` pose features for a recorded video."""
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    seq = []
    index = 0
    try:
        with PoseExtractor(model_path) as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pose = extractor.extract(frame, captured_at=index / fps)
                seq.append(pose.landmarks)
                index += 1
    finally:
        cap.release()
    return np.stack(seq) if seq else np.zeros((1, POSE_DIM), np.float32)


def pose_sequence_to_windows(seq: np.ndarray, cfg=CFG, src_fps: float = 30.0):
    """Segment a pose sequence into fixed ``(t_vis, vision_dv)`` windows."""
    from .windowing import segment_recording

    seq = (
        seq[:, :cfg.vision_dv]
        if seq.shape[1] >= cfg.vision_dv
        else np.pad(seq, ((0, 0), (0, cfg.vision_dv - seq.shape[1])))
    )
    return segment_recording(seq, src_fps, cfg.t_vis, cfg.window_seconds)


def video_to_aligned_pose_windows(
    video_path: str,
    window_starts: list[float],
    cfg=CFG,
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    """Extract only the frames needed for timestamp-aligned C-MHAD windows.

    The video is decoded once per recording. Requested source-frame indices are
    deduplicated, which is much faster than running MediaPipe over every frame
    in each two-minute stream. Returns ``(pose_windows, valid_ratios,
    quality_means)`` in the same order as ``window_starts``.
    """
    cv2, _ = _lazy_imports()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    requested = []
    for start in window_starts:
        indices = [
            int(round((start + offset / cfg.vision_fps) * src_fps))
            for offset in range(cfg.t_vis)
        ]
        requested.append([max(0, min(index, frame_count - 1)) for index in indices])

    unique_indices = sorted({index for indices in requested for index in indices})
    extracted = {}
    try:
        with PoseExtractor(model_path) as extractor:
            for index in unique_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    extracted[index] = PoseFrame(
                        captured_at=index / src_fps,
                        landmarks=np.zeros(POSE_DIM, dtype=np.float32),
                        valid=False,
                        image_quality=0.0,
                    )
                    continue
                extracted[index] = extractor.extract(frame, captured_at=index / src_fps)
    finally:
        cap.release()

    windows, valid_ratios, qualities = [], [], []
    for indices in requested:
        frames = [extracted[index] for index in indices]
        windows.append(np.stack([frame.landmarks for frame in frames]).astype(np.float32))
        valid_ratios.append(float(np.mean([frame.valid for frame in frames])))
        qualities.append(float(np.mean([frame.image_quality for frame in frames])))
    return windows, valid_ratios, qualities
