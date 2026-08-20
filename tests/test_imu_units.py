import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.data.imu_units import (
    STANDARD_GRAVITY_MPS2,
    convert_imu_to_canonical,
    validate_canonical_imu,
    validate_imu_unit_row,
)


def test_live_g_is_converted_and_gyro_is_unchanged():
    raw = np.array([[1.0, -0.5, 0.25, 4.0, -5.0, 6.0]], dtype=np.float32)
    converted = convert_imu_to_canonical(raw, acceleration_unit="g")
    assert np.allclose(converted[0, :3], raw[0, :3] * STANDARD_GRAVITY_MPS2)
    assert np.array_equal(converted[0, 3:], raw[0, 3:])


def test_dataset_canonical_units_are_not_rescaled():
    values = np.array([[0.0, 0.0, 9.80665, 1.0, 2.0, 3.0]], dtype=np.float32)
    converted = convert_imu_to_canonical(values, acceleration_unit="m/(s^2)")
    assert np.array_equal(converted, values)
    validate_imu_unit_row(
        ["m/(s^2)", "m/(s^2)", "m/(s^2)", "degrees/s", "degrees/s", "degrees/s"]
    )


def test_scale_check_catches_unconverted_g_values():
    raw_g = np.tile(np.array([[0, 0, 1, 0, 0, 0]], dtype=np.float32), (100, 1))
    assert not validate_canonical_imu(raw_g).valid
    assert validate_canonical_imu(
        convert_imu_to_canonical(raw_g, acceleration_unit="g")
    ).valid


def test_unknown_units_are_rejected():
    try:
        convert_imu_to_canonical(np.zeros((2, 6), np.float32), acceleration_unit="ft/s2")
    except ValueError:
        return
    raise AssertionError("Unknown acceleration units must be rejected")


if __name__ == "__main__":
    test_live_g_is_converted_and_gyro_is_unchanged()
    test_dataset_canonical_units_are_not_rescaled()
    test_scale_check_catches_unconverted_g_values()
    test_unknown_units_are_rejected()
    print("ALL IMU UNIT TESTS PASSED")
