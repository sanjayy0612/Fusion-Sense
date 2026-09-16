# FusionSense timestamped IMU USB node

This sketch streams the calibrated 50 Hz MPU-60x0/65x0 measurements to the
laptop over one persistent USB serial connection.

## Packet contract

```text
IMU,1,imu01,<session_id>,<seq>,<t_device_us>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>
```

The laptop sends `SESSION,<session_id>` after opening the port. Sequence numbers
represent scheduled acquisition slots, so I2C failures or missed scheduler slots
produce visible gaps. `t_device_us` is captured with `esp_timer_get_time()` at
acquisition. Status and command replies begin with `#`.

Supported commands:

```text
SESSION,<session_id>
SYNC,<request_id>
INFO
```

`SYNC` is included for the following clock-mapping phase.

## Upload and record

Select **ESP32 Dev Module** and the CP210x USB port. Keep the sensor stationary
during the five-second startup calibration. Close Arduino Serial Monitor before
starting the recorder:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_imu_serial.py --port <imu-com-port> --duration 60 --stationary
```

The output directory contains `imu.csv`, `device_status.csv`, and `session.json`.
The acceptance run must report 50 Hz, monotonic timestamps/sequence, zero
sequence gaps, zero malformed rows, and (with `--stationary`) plausible 1 g and
gyro-bias statistics.

## Verified live result

The 2026-08-17 hardware run passed: 3,000 samples over 59.98 device seconds at
50.0 Hz, with a 20.0 ms mean interval and zero invalid rows, missed slots,
sequence gaps, or non-monotonic timestamps. Mean acceleration magnitude was
0.99962 g and mean gyroscope magnitude was 0.12035 dps. The complete result is
stored in `data/recordings/imu_20260817T174209Z/session.json`.
