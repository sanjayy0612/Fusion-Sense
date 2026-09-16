"""Explicit IMU unit handling for dataset and ESP32 inputs.

FusionSense uses one canonical model-input contract:

``[ax, ay, az, gx, gy, gz] = [m/s^2, m/s^2, m/s^2, deg/s, deg/s, deg/s]``.

The ESP32 firmware reports acceleration in ``g`` and angular velocity in
``degrees/s``.  C-MHAD already uses the canonical units, so only the three
live acceleration channels are scaled before applying training statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STANDARD_GRAVITY_MPS2 = np.float32(9.80665)
CANONICAL_IMU_UNITS = (
    "m/s^2",
    "m/s^2",
    "m/s^2",
    "deg/s",
    "deg/s",
    "deg/s",
)


def normalize_acceleration_unit(unit: str) -> str:
    compact = unit.strip().lower().replace(" ", "")
    if compact in {"g", "gravity", "gravities"}:
        return "g"
    if compact in {
        "m/s^2",
        "m/s2",
        "m/(s^2)",
        "m/(s²)",
        "m/s²",
        "meter/second^2",
        "metres/second^2",
    }:
        return "m/s^2"
    raise ValueError(f"Unsupported acceleration unit {unit!r}; expected g or m/s^2")


def normalize_angular_velocity_unit(unit: str) -> str:
    compact = unit.strip().lower().replace(" ", "")
    if compact in {"deg/s", "degree/s", "degrees/s", "dps", "°/s"}:
        return "deg/s"
    raise ValueError(
        f"Unsupported angular-velocity unit {unit!r}; expected degrees/s"
    )


def validate_imu_unit_row(units: list[str] | tuple[str, ...]) -> None:
    """Reject a dataset header that does not match the model's unit contract."""
    if len(units) < 6:
        raise ValueError(f"Expected six IMU units, got {units}")
    acceleration = [normalize_acceleration_unit(value) for value in units[:3]]
    gyroscope = [normalize_angular_velocity_unit(value) for value in units[3:6]]
    if acceleration != ["m/s^2"] * 3 or gyroscope != ["deg/s"] * 3:
        raise ValueError(
            "C-MHAD must provide acceleration in m/s^2 and angular velocity in deg/s; "
            f"got {units[:6]}"
        )


def convert_imu_to_canonical(
    values: np.ndarray,
    *,
    acceleration_unit: str,
    angular_velocity_unit: str = "deg/s",
) -> np.ndarray:
    """Convert an ``(..., 6)`` IMU array to the canonical model units."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] != 6:
        raise ValueError(f"Expected an IMU array ending in six channels, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("IMU input contains NaN or infinite values")

    accel_unit = normalize_acceleration_unit(acceleration_unit)
    normalize_angular_velocity_unit(angular_velocity_unit)
    output = source.copy()
    if accel_unit == "g":
        output[..., :3] *= STANDARD_GRAVITY_MPS2
    return output


@dataclass(frozen=True)
class ImuValidation:
    valid: bool
    samples: int
    acceleration_norm_median_mps2: float
    acceleration_norm_p95_mps2: float
    finite: bool
    canonical_units: tuple[str, ...] = CANONICAL_IMU_UNITS

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "samples": self.samples,
            "acceleration_norm_median_mps2": round(
                self.acceleration_norm_median_mps2, 5
            ),
            "acceleration_norm_p95_mps2": round(self.acceleration_norm_p95_mps2, 5),
            "finite": self.finite,
            "canonical_units": list(self.canonical_units),
        }


def validate_canonical_imu(values: np.ndarray) -> ImuValidation:
    """Run a broad physical-scale check after conversion.

    This is deliberately tolerant of falls and other high acceleration.  Its
    purpose is to catch missing conversion (roughly 1 instead of 9.81), NaNs,
    and wildly implausible units—not to classify an activity.
    """
    array = np.asarray(values, dtype=np.float32)
    shape_ok = array.ndim == 2 and array.shape[1] == 6 and len(array) > 0
    finite = bool(shape_ok and np.isfinite(array).all())
    if not finite:
        return ImuValidation(False, len(array) if array.ndim else 0, 0.0, 0.0, False)
    norms = np.linalg.norm(array[:, :3], axis=1)
    median = float(np.median(norms))
    p95 = float(np.percentile(norms, 95))
    # A stationary device is near 9.81 m/s^2. Human motion may be much wider,
    # hence the intentionally broad bounds.
    scale_ok = 3.0 <= median <= 25.0 and p95 <= 100.0
    return ImuValidation(scale_ok, len(array), median, p95, True)


def live_conversion_metadata() -> dict:
    return {
        "source_acceleration_unit": "g",
        "source_angular_velocity_unit": "deg/s",
        "target_units": list(CANONICAL_IMU_UNITS),
        "acceleration_formula": "m/s^2 = g * 9.80665",
        "acceleration_scale": round(float(STANDARD_GRAVITY_MPS2), 5),
        "gyroscope_formula": "deg/s unchanged",
    }
