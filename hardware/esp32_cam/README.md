# ESP32-CAM camera node

The ESP32-CAM is FusionSense V1's primary camera. It captures OV2640 JPEG frames
and exposes a Wi-Fi MJPEG stream. All pose extraction and ML remain on the
laptop.

## USB-serial transport (current hardware workaround)

`fusionsense_camera_serial/fusionsense_camera_serial.ino` removes Wi-Fi from
the camera path and sends timestamped QVGA JPEGs through the ESP32-CAM-MB USB
serial connection at 921600 baud. Each binary frame carries a sequence number,
camera capture timestamp, dimensions, JPEG length, cumulative capture-error
count, and CRC32. The original Wi-Fi firmware remains available as a fallback.

After uploading the serial sketch and confirming its readable startup output at
921600 baud, close Serial Monitor and run:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_esp32_camera_serial.py --port COM10 --duration 30
```

Replace `COM10` with the ESP32-CAM-MB port (COM14 in the verified setup). The laptop sends `START`, validates
the binary stream and JPEG CRCs, and sends `STOP` after the gate.

Live verification on COM14 passed: 302 QVGA frames over 30.0029 device seconds
at 10.0324 FPS, capture interval p95/max 108.095/108.153 ms, and zero capture
errors, CRC errors, dropped frames, sequence gaps, or timestamp regressions.

## Timestamped firmware

Open
`hardware/esp32_cam/fusionsense_camera/fusionsense_camera.ino` in Arduino IDE.
Copy `secrets.example.h` to `secrets.h` in the same sketch folder and replace
the placeholder Wi-Fi values. `secrets.h` is ignored by Git.

Select the **AI Thinker ESP32-CAM** board, enable PSRAM, and upload through the
ESP32-CAM-MB. The firmware captures QVGA JPEG at a target 10 FPS. Every MJPEG
part carries `X-Device-Id`, `X-Session-Id`, `X-Frame-Sequence`, and
`X-Capture-Timestamp-Us`; the timestamp comes from the camera driver's
framebuffer at capture rather than laptop arrival.

Step 4 adds a separate control server on port 80. It remains responsive while
the persistent MJPEG stream occupies port 8080, allowing repeated round-trip
clock probes during capture. After boot, verify these endpoints:

```text
http://<esp32-ip>/health
http://<esp32-ip>/sync?id=test1
http://<esp32-ip>:8080/stream
```

Then test the complete laptop path:

```bash
python scripts/download_pose_model.py
python scripts/test_esp32_camera.py --host <esp32-ip> --seconds 30
```

First validate capture/transport without MediaPipe:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_esp32_camera.py --host <esp32-ip> --duration 30
```

This prints device FPS, capture interval statistics, sequence gaps, dimensions,
and camera health. A `PASS` requires approximately 10 device FPS, QVGA frames,
monotonic sequence/timestamps, no gaps, and no camera capture errors. Then run
`test_esp32_camera.py` to validate the later MediaPipe path.

## Verified result (2026-08-17)

The physical ESP32-CAM passed the validator with 283 QVGA frames over 30.255
seconds (9.32 device FPS), zero dropped frames, and zero capture errors. The
reported stream disconnect count of one is the expected client close at the end
of validation, not a capture failure.

## Record timestamped JPEGs on the laptop

The MJPEG response remains open for the whole recording, so this is one
persistent Wi-Fi TCP connection rather than one connection per image. Record
the camera before combining it with IMU acquisition:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_esp32_camera.py --host <esp32-ip> --duration 60
```

Each recording is stored under `data/recordings/camera_<UTC>/` with the exact
original JPEG payloads, `camera_manifest.csv`, and `session.json`. The manifest
contains frame sequence, ESP32 capture time, laptop monotonic receive time,
dimensions, byte length, and relative JPEG path. These are the camera inputs to
the upcoming clock-mapping and multimodal windowing phase.

The regular wearable ESP32 and MPU-6050 are a separate node. Do not connect the
OV2640 ribbon camera to that board.

## Step 4 synchronized camera + IMU recording

Use the USB-serial camera sketch above; Wi-Fi and the HTTP control endpoints are
not required. The IMU firmware already supports `SESSION` and `SYNC`. Close
both Arduino Serial Monitors, connect both USB devices, then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM17 --camera-port COM14 --duration 60 --motion-check
```

The command records both devices concurrently and writes `imu.csv`,
`camera_manifest.csv`, original JPEGs under `frames/`, `clock_sync.csv`,
`device_status.csv`, `malformed_imu.csv`, and `session.json`. After ten seconds,
perform three distinct sharp side-to-side movements while the IMU is visible to
the camera. A Step 4 `PASS` requires healthy rates/sequence, no malformed IMU
rows, at least four sync points per device, no more than 20 ms p95 clock-fit
residual, no capture gap above five target frame intervals, bounded delivery
latency, and shared-motion correlation with no more than 50 ms remaining lag.
Nearest-IMU sample distance is not an acceptance test.

The verified session is `data/recordings/fusion_usb_step4_20260819/`: 3,000
IMU samples at 50.0 Hz and 601 QVGA frames at 10.005 FPS, no invalid rows or
camera/sequence errors, clock residual p95 of 7.61 ms (IMU) and 14.89 ms
(camera), shared-motion correlation 0.597, and 0 ms remaining calibrated lag.
Its `session.json` reports top-level `PASS`.

