"""Safety-oriented evaluation helpers for fall-alert models."""
from __future__ import annotations

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.9599639845):
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def binary_alert_metrics(
    truth: np.ndarray,
    fall_probability: np.ndarray,
    threshold: float,
    *,
    negative_hours: float | None = None,
) -> dict:
    """Return alert-policy metrics for binary fall/non-fall ground truth."""
    truth = np.asarray(truth, dtype=bool)
    probability = np.asarray(fall_probability, dtype=np.float64)
    if truth.ndim != 1 or probability.shape != truth.shape or len(truth) == 0:
        raise ValueError("truth and fall_probability must be equal non-empty vectors")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    alert = probability >= threshold
    tp = int(np.sum(alert & truth))
    fp = int(np.sum(alert & ~truth))
    fn = int(np.sum(~alert & truth))
    tn = int(np.sum(~alert & ~truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    result = {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "precision_wilson_95": wilson_interval(tp, tp + fp),
        "recall_wilson_95": wilson_interval(tp, tp + fn),
    }
    if negative_hours is not None:
        if negative_hours <= 0:
            raise ValueError("negative_hours must be positive")
        result["negative_hours"] = float(negative_hours)
        result["false_alerts_per_hour"] = float(fp / negative_hours)
        if fp == 0:
            # One-sided 95% Poisson upper bound when no events are observed.
            result["zero_count_upper_95_false_alerts_per_hour"] = float(
                -np.log(0.05) / negative_hours
            )
    return result


def binary_brier_score(truth: np.ndarray, fall_probability: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    probability = np.asarray(fall_probability, dtype=np.float64)
    if truth.shape != probability.shape or truth.size == 0:
        raise ValueError("truth and fall_probability must have equal non-empty shapes")
    return float(np.mean((probability - truth) ** 2))


def expected_calibration_error(
    truth: np.ndarray, fall_probability: np.ndarray, bins: int = 10
) -> float:
    """Binary expected calibration error with equal-width probability bins."""
    truth = np.asarray(truth, dtype=np.float64)
    probability = np.asarray(fall_probability, dtype=np.float64)
    if truth.shape != probability.shape or truth.size == 0:
        raise ValueError("truth and fall_probability must have equal non-empty shapes")
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(truth[selected].mean())
            )
    return float(error)


def shift_time_series(values: np.ndarray, shift_ms: float, rate_hz: float) -> np.ndarray:
    """Shift a time series with linear interpolation and edge-value padding."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("values must have shape (time, channels)")
    offset = float(shift_ms) * float(rate_hz) / 1000.0
    source = np.arange(len(values), dtype=np.float64)
    requested = source - offset
    shifted = np.stack(
        [
            np.interp(
                requested,
                source,
                values[:, channel],
                left=float(values[0, channel]),
                right=float(values[-1, channel]),
            )
            for channel in range(values.shape[1])
        ],
        axis=1,
    )
    return shifted.astype(np.float32)
