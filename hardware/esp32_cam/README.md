# ESP32-CAM camera node

The ESP32-CAM is FusionSense V1's primary camera. It captures OV2640 JPEG frames
and exposes a Wi-Fi MJPEG stream. All pose extraction and ML remain on the
laptop.

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
the persistent MJPEG stream occupies port 81, allowing repeated round-trip
clock probes during capture. After boot, verify these endpoints:

```text
http://<esp32-ip>/health
http://<esp32-ip>/sync?id=test1
http://<esp32-ip>:81/stream
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

Re-upload the current camera sketch before this step; the earlier Step 2 image
does not provide the separate `/session` and `/sync` control endpoints. The IMU
firmware already supports `SESSION` and `SYNC`, so it does not need another
upload. Close both Arduino Serial Monitors, keep the camera powered and joined
to Wi-Fi, connect the IMU ESP32 over USB, then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM16 --camera-host <esp32-ip> --duration 60 --motion-check
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

