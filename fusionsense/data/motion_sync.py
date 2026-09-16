"""Validate cross-device timing using motion observed by both modalities."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class MotionPoint:
    host_time_ns: int
    score: float


def _robust_activity(values: list[float]) -> tuple[list[float], dict[str, float]]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    scale = mad * 1.4826
    if scale < 1e-9:
        scale = statistics.pstdev(values) if len(values) > 1 else 0.0
    scale = max(scale, 1e-9)
    activity = [max(0.0, min(20.0, (value - median) / scale)) for value in values]
    return activity, {
        "baseline": round(median, 6),
        "robust_scale": round(scale, 6),
        "peak_z": round(max(activity), 4),
        "active_points_over_3z": sum(value >= 3.0 for value in activity),
    }


def _interpolate(times: list[int], values: list[float], target: int) -> float | None:
    index = bisect_left(times, target)
    if index == 0:
        return values[0] if target == times[0] else None
    if index == len(times):
        return values[-1] if target == times[-1] else None
    before_time, after_time = times[index - 1], times[index]
    before_value, after_value = values[index - 1], values[index]
    if after_time == before_time:
        return after_value
    fraction = (target - before_time) / (after_time - before_time)
    return before_value + fraction * (after_value - before_value)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return math.nan
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-12:
        return math.nan
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def motion_alignment_report(
    imu_points: list[MotionPoint],
    camera_points: list[MotionPoint],
    *,
    search_range_ms: int = 500,
    lag_step_ms: int = 5,
    maximum_accepted_lag_ms: float = 50.0,
    minimum_correlation: float = 0.25,
) -> dict[str, object]:
    """Cross-correlate shared motion and report the remaining clock lag.

    ``best_camera_shift_ms`` is the amount that would need to be added to mapped
    camera capture timestamps to line their motion up with the IMU signal.
    """
    if len(imu_points) < 10 or len(camera_points) < 5:
        return {
            "result": "FAIL",
            "reason": "not enough mapped motion samples",
            "method": "shared_motion_cross_correlation",
        }
    if search_range_ms <= 0 or lag_step_ms <= 0:
        raise ValueError("motion lag search parameters must be positive")

    imu_ordered = sorted(imu_points, key=lambda item: item.host_time_ns)
    camera_ordered = sorted(camera_points, key=lambda item: item.host_time_ns)
    imu_times = [item.host_time_ns for item in imu_ordered]
    camera_times = [item.host_time_ns for item in camera_ordered]
    imu_activity, imu_evidence = _robust_activity(
        [float(item.score) for item in imu_ordered]
    )
    camera_activity, camera_evidence = _robust_activity(
        [float(item.score) for item in camera_ordered]
    )
    motion_present = (
        imu_evidence["peak_z"] >= 3.0
        and imu_evidence["active_points_over_3z"] >= 3
        and camera_evidence["peak_z"] >= 3.0
        and camera_evidence["active_points_over_3z"] >= 2
    )
    if not motion_present:
        return {
            "result": "FAIL",
            "reason": (
                "shared motion was not strong enough; perform three sharp "
                "side-to-side movements while the IMU is visible to the camera"
            ),
            "method": "shared_motion_cross_correlation",
            "imu_motion": imu_evidence,
            "camera_motion": camera_evidence,
        }

    candidates: list[tuple[float, int, int]] = []
    for lag_ms in range(-search_range_ms, search_range_ms + 1, lag_step_ms):
        lag_ns = lag_ms * 1_000_000
        camera_values: list[float] = []
        interpolated_imu: list[float] = []
        for camera_time, camera_value in zip(camera_times, camera_activity):
            imu_value = _interpolate(imu_times, imu_activity, camera_time + lag_ns)
            if imu_value is None:
                continue
            camera_values.append(camera_value)
            interpolated_imu.append(imu_value)
        correlation = _correlation(camera_values, interpolated_imu)
        if math.isfinite(correlation):
            candidates.append((correlation, lag_ms, len(camera_values)))

    if not candidates:
        return {
            "result": "FAIL",
            "reason": "motion streams do not overlap after clock mapping",
            "method": "shared_motion_cross_correlation",
        }
    best_correlation, best_lag_ms, compared_points = max(
        candidates, key=lambda item: item[0]
    )
    checks = {
        "shared_motion_detected": motion_present,
        "correlation_at_least_0_25": best_correlation >= minimum_correlation,
        "absolute_lag_at_most_50_ms": (
            abs(best_lag_ms) <= maximum_accepted_lag_ms
        ),
    }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "method": "shared_motion_cross_correlation",
        "best_camera_shift_ms": best_lag_ms,
        "absolute_lag_ms": abs(best_lag_ms),
        "best_correlation": round(best_correlation, 6),
        "compared_camera_points": compared_points,
        "search_range_ms": search_range_ms,
        "lag_step_ms": lag_step_ms,
        "imu_motion": imu_evidence,
        "camera_motion": camera_evidence,
        "checks": checks,
    }
