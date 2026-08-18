"""Map independent ESP32 monotonic clocks into the laptop clock domain.

Clock synchronization uses request/response observations.  The laptop records
the monotonic time immediately before sending a request and immediately after
receiving the reply.  The device timestamp is mapped to the midpoint of that
round trip.  An affine fit accounts for offset and slow oscillator drift.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class SyncObservation:
    """One host/device request-response timing observation."""

    device_id: str
    request_id: str
    device_time_us: int
    host_send_ns: int
    host_receive_ns: int

    def __post_init__(self) -> None:
        if not self.device_id or not self.request_id:
            raise ValueError("sync device/request IDs must not be empty")
        if self.device_time_us < 0:
            raise ValueError("device sync timestamp must be non-negative")
        if self.host_receive_ns < self.host_send_ns:
            raise ValueError("sync receive time precedes send time")

    @property
    def rtt_ns(self) -> int:
        return self.host_receive_ns - self.host_send_ns

    @property
    def host_midpoint_ns(self) -> int:
        return self.host_send_ns + self.rtt_ns // 2


@dataclass(frozen=True)
class ClockMapping:
    """Affine ``host_ns = scale * device_us * 1000 + offset_ns`` mapping."""

    device_id: str
    scale: float
    offset_ns: float
    total_points: int
    selected_points: int
    selected_device_span_s: float
    rtt_ms: dict[str, float]
    residual_ms: dict[str, float]

    def map_device_us(self, device_time_us: int) -> int:
        return int(round(self.scale * device_time_us * 1000.0 + self.offset_ns))

    @property
    def drift_ppm(self) -> float:
        return (self.scale - 1.0) * 1_000_000.0

    def report(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "equation": "host_ns = scale * device_us * 1000 + offset_ns",
            "scale": round(self.scale, 12),
            "drift_ppm": round(self.drift_ppm, 4),
            "offset_ns": int(round(self.offset_ns)),
            "total_sync_points": self.total_points,
            "selected_low_rtt_points": self.selected_points,
            "selected_device_span_s": round(self.selected_device_span_s, 6),
            "rtt_ms": self.rtt_ms,
            "fit_residual_ms": self.residual_ms,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summary_ms(values_ns: list[float]) -> dict[str, float]:
    values_ms = [value / 1_000_000.0 for value in values_ns]
    return {
        "min": round(min(values_ms), 6),
        "median": round(statistics.median(values_ms), 6),
        "p95": round(_percentile(values_ms, 0.95), 6),
        "max": round(max(values_ms), 6),
    }


def _select_low_rtt_spanning(
    observations: list[SyncObservation], maximum: int
) -> list[SyncObservation]:
    """Keep the lowest-RTT point in each time bucket across the session."""
    ordered = sorted(observations, key=lambda item: item.device_time_us)
    if len(ordered) <= maximum:
        return ordered

    selected: list[SyncObservation] = []
    for index in range(maximum):
        start = index * len(ordered) // maximum
        end = (index + 1) * len(ordered) // maximum
        selected.append(min(ordered[start:end], key=lambda item: item.rtt_ns))
    return selected


def fit_affine_clock(
    observations: list[SyncObservation], *, maximum_selected_points: int = 16
) -> ClockMapping:
    """Fit a stable affine clock mapping from low-RTT observations."""
    if not observations:
        raise ValueError("at least one sync observation is required")
    if maximum_selected_points < 2:
        raise ValueError("maximum_selected_points must be at least two")
    device_ids = {item.device_id for item in observations}
    if len(device_ids) != 1:
        raise ValueError("clock mapping observations must belong to one device")

    selected = _select_low_rtt_spanning(observations, maximum_selected_points)
    x_values = [item.device_time_us * 1000.0 for item in selected]
    y_values = [float(item.host_midpoint_ns) for item in selected]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if len(selected) == 1 or denominator == 0.0:
        scale = 1.0
    else:
        numerator = sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values)
        )
        scale = numerator / denominator
    offset_ns = y_mean - scale * x_mean
    residuals_ns = [
        abs(y_value - (scale * x_value + offset_ns))
        for x_value, y_value in zip(x_values, y_values)
    ]
    span_s = (max(x_values) - min(x_values)) / 1_000_000_000.0
    return ClockMapping(
        device_id=next(iter(device_ids)),
        scale=scale,
        offset_ns=offset_ns,
        total_points=len(observations),
        selected_points=len(selected),
        selected_device_span_s=span_s,
        rtt_ms=_summary_ms([float(item.rtt_ns) for item in selected]),
        residual_ms=_summary_ms(residuals_ns),
    )


def clock_mapping_passes(mapping: ClockMapping) -> dict[str, bool]:
    """Return conservative MVP acceptance checks for a fitted mapping."""
    return {
        "at_least_4_sync_points": mapping.total_points >= 4,
        "selected_span_at_least_2_seconds": mapping.selected_device_span_s >= 2.0,
        "absolute_drift_at_most_2000_ppm": abs(mapping.drift_ppm) <= 2000.0,
        "fit_residual_p95_at_most_20_ms": mapping.residual_ms["p95"] <= 20.0,
    }


def nearest_alignment_report(
    imu_host_times_ns: list[int], camera_host_times_ns: list[int]
) -> dict[str, object]:
    """Measure each camera frame's distance from its nearest IMU sample."""
    if not imu_host_times_ns or not camera_host_times_ns:
        return {
            "result": "FAIL",
            "reason": "both mapped IMU and camera timestamps are required",
            "pairs": 0,
        }

    imu_times = sorted(imu_host_times_ns)
    skews_ns: list[int] = []
    for camera_time in camera_host_times_ns:
        index = bisect_left(imu_times, camera_time)
        candidates: list[int] = []
        if index < len(imu_times):
            candidates.append(abs(imu_times[index] - camera_time))
        if index > 0:
            candidates.append(abs(imu_times[index - 1] - camera_time))
        skews_ns.append(min(candidates))

    summary = _summary_ms([float(value) for value in skews_ns])
    return {
        "result": "PASS" if summary["median"] <= 50.0 else "FAIL",
        "pairs": len(skews_ns),
        "nearest_imu_skew_ms": summary,
        "median_alignment_at_most_50_ms": summary["median"] <= 50.0,
    }


def transport_latency_report(
    host_receive_times_ns: list[int], host_capture_times_ns: list[int]
) -> dict[str, float]:
    if len(host_receive_times_ns) != len(host_capture_times_ns):
        raise ValueError("receive and capture timestamp counts must match")
    if not host_receive_times_ns:
        return {}
    return _summary_ms(
        [
            float(received - captured)
            for received, captured in zip(
                host_receive_times_ns, host_capture_times_ns
            )
        ]
    )


def parse_imu_sync_response(line: str) -> tuple[str, int] | None:
    """Parse ``# SYNC_RESP,<request_id>,<device_us>`` status lines."""
    prefix = "# SYNC_RESP,"
    if not line.startswith(prefix):
        return None
    fields = line[len(prefix) :].split(",")
    if len(fields) != 2 or not fields[0]:
        raise ValueError("invalid IMU sync response")
    device_time_us = int(fields[1])
    if device_time_us < 0:
        raise ValueError("negative IMU sync timestamp")
    return fields[0], device_time_us
