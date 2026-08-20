"""Checkpoint-backed FusionSense inference and fall-alert policy."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np

from .config import CFG
from .contract import ACTIVITIES, LABEL2ID, FusionWindow
from .data.imu_units import CANONICAL_IMU_UNITS


class FallInferenceEngine:
    """Load one trained checkpoint and evaluate canonical fusion windows."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints/cmhad_step7_bimodal_dropout_fixed",
        *,
        fall_threshold: float = 0.60,
        device: str = "cpu",
        require_bimodal: bool = True,
        cfg=CFG,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Model inference needs PyTorch") from exc

        from .models.fusion import FusionSense

        self.torch = torch
        self.cfg = cfg
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fall_threshold = float(fall_threshold)
        self.require_bimodal = bool(require_bimodal)
        if not 0.0 < self.fall_threshold < 1.0:
            raise ValueError("fall_threshold must be between zero and one")

        normalization_path = self.checkpoint_dir / "normalization.npz"
        labels_path = self.checkpoint_dir / "labels.json"
        model_path = self.checkpoint_dir / "fusionsense_cmhad.pt"
        for path in (normalization_path, labels_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing inference artifact {path}")

        stats = np.load(normalization_path, allow_pickle=False)
        self.stats = {name: stats[name].astype(np.float32) for name in stats.files}
        self.labels_metadata = json.loads(labels_path.read_text(encoding="utf-8"))
        if self.labels_metadata.get("activities") != ACTIVITIES:
            raise ValueError("Checkpoint class labels do not match the FusionSense contract")

        self.model = FusionSense(cfg).to(self.device)
        self.model_name = "FusionSense corrected masked IMU + camera fusion"
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _normalize(self, window: FusionWindow) -> FusionWindow:
        return replace(
            window,
            imu=((window.imu - self.stats["imu_mean"]) / self.stats["imu_std"]).astype(
                np.float32
            ),
            vision=(
                (window.vision - self.stats["vision_mean"]) / self.stats["vision_std"]
            ).astype(np.float32),
        )

    def predict(self, window: FusionWindow) -> dict:
        valid_vector = window.valid_vector()
        base = {
            "model": self.model_name,
            "checkpoint": str(self.checkpoint_dir.resolve()),
            "window_start": window.t_start,
            "window_seconds": self.cfg.window_seconds,
            "recording_id": window.recording_id,
            "subject_id": window.subject_id,
            "input": {
                "imu_shape": list(window.imu.shape),
                "imu_units": list(CANONICAL_IMU_UNITS),
                "vision_shape": list(window.vision.shape),
                "radar_shape": list(window.radar.shape),
                "valid_modalities": {
                    "imu": bool(window.imu_valid),
                    "radar": bool(window.radar_valid),
                    "camera_pose": bool(window.vision_valid),
                },
                "health": {
                    "imu": round(float(window.imu_health), 4),
                    "radar": round(float(window.radar_energy), 4),
                    "camera": round(float(window.image_quality), 4),
                },
                "normalization": "training-subject per-channel mean/std",
            },
            "fall_threshold": self.fall_threshold,
            "fusion_policy": (
                "require_valid_imu_and_camera_pose"
                if self.require_bimodal
                else "graceful_degradation"
            ),
        }
        if self.require_bimodal and not (
            window.imu_valid and window.vision_valid and not window.radar_valid
        ):
            return {
                **base,
                "status": "degraded",
                "alert": False,
                "reason": (
                    "Bimodal inference requires valid IMU and camera pose, "
                    "with radar disabled"
                ),
                "predicted_label": None,
                "confidence": 0.0,
                "fall_probability": 0.0,
                "class_probabilities": {},
                "sensor_trust": {"imu": 0.0, "radar": 0.0, "camera": 0.0},
            }
        if not valid_vector.any():
            return {
                **base,
                "status": "degraded",
                "alert": False,
                "reason": "No valid modality is available for this window",
                "predicted_label": None,
                "confidence": 0.0,
                "fall_probability": 0.0,
                "class_probabilities": {},
                "sensor_trust": {"imu": 0.0, "radar": 0.0, "camera": 0.0},
            }

        normalized = self._normalize(window)
        torch = self.torch
        inputs = [
            torch.from_numpy(normalized.imu).unsqueeze(0).to(self.device),
            torch.from_numpy(normalized.radar).unsqueeze(0).to(self.device),
            torch.from_numpy(normalized.vision).unsqueeze(0).to(self.device),
            torch.from_numpy(valid_vector).unsqueeze(0).to(self.device),
            torch.from_numpy(window.health_vector()).unsqueeze(0).to(self.device),
        ]
        started = time.perf_counter()
        with torch.inference_mode():
            logits, trust = self.model(*inputs, return_trust=True)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            trust_values = trust[0].cpu().numpy()
        inference_ms = (time.perf_counter() - started) * 1000.0

        predicted_id = int(np.argmax(probabilities))
        predicted_label = ACTIVITIES[predicted_id]
        fall_probability = float(probabilities[LABEL2ID["stand_to_fall"]])
        alert = fall_probability >= self.fall_threshold
        return {
            **base,
            "status": "fall_alert" if alert else "monitoring",
            "alert": alert,
            "predicted_label": predicted_label,
            "predicted_class_id": predicted_id,
            "confidence": round(float(probabilities[predicted_id]), 6),
            "fall_probability": round(fall_probability, 6),
            "class_probabilities": {
                label: round(float(probabilities[index]), 6)
                for index, label in enumerate(ACTIVITIES)
            },
            "sensor_trust": {
                "imu": round(float(trust_values[0]), 6),
                "radar": round(float(trust_values[1]), 6),
                "camera": round(float(trust_values[2]), 6),
            },
            "inference_ms": round(inference_ms, 3),
        }


def classification_metrics(truth: list[int], predicted: list[int]) -> dict:
    truth_array = np.asarray(truth, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    if len(truth_array) == 0 or len(truth_array) != len(predicted_array):
        raise ValueError("Metrics require equal, non-empty truth and prediction sequences")

    per_class_f1 = []
    per_class_recall = []
    confusion = np.zeros((len(ACTIVITIES), len(ACTIVITIES)), dtype=np.int64)
    for actual, guess in zip(truth_array, predicted_array):
        confusion[actual, guess] += 1
    for class_id in range(len(ACTIVITIES)):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class_f1.append(f1)
        per_class_recall.append(recall)
    return {
        "samples": int(len(truth_array)),
        "accuracy": round(float(np.mean(truth_array == predicted_array)), 4),
        "macro_f1": round(float(np.mean(per_class_f1)), 4),
        "fall_recall": round(float(per_class_recall[LABEL2ID["stand_to_fall"]]), 4),
        "confusion_matrix": confusion.tolist(),
    }
