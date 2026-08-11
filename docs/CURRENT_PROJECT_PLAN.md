# FusionSense — Current Project Plan

This is the authoritative plan for the first practical FusionSense version.
Older Pi-first and radar-first plans are not the V1 direction.

## 1. V1 objective

Build a working **camera + wearable IMU** activity-recognition and fall-alert
system. mmWave radar remains the intended third modality, but it is integrated
only after the two-modality pipeline works end to end.

```text
MPU-6050 → wearable ESP32 → timestamped IMU ──┐
                                                   ├─→ Laptop synchronizer
OV2640 → ESP32-CAM → Wi-Fi MJPEG → OpenCV ─┘          │
                                                              ├─→ ML model
                                                   MediaPipe Pose ─┘      │
                                                                         └─→ Dashboard → Fall alerts
```

## 2. V1 system decisions

- One wearable ESP32 reads the MPU-6050 and transmits timestamped accelerometer
  and gyroscope samples.
- A dedicated ESP32-CAM captures OV2640 frames and streams MJPEG over Wi-Fi.
- OpenCV receives the ESP32-CAM stream on the laptop.
- MediaPipe Pose Landmarker runs on the laptop and converts decoded frames into
  33 landmark coordinates (99 values per pose frame).
- The laptop is the clock authority and synchronizes camera and IMU samples into
  two-second windows.
- The model fuses the camera-pose and IMU branches.
- The radar tensor stays zero-filled and `radar_valid=False` for every V1
  training and inference window.
- The dashboard reports activity, confidence, camera/IMU health, and fall alerts.
- No Raspberry Pi is used. V1 has two ESP32 roles: wearable IMU acquisition and
  fixed ESP32-CAM video streaming.
- A future radar node may use another microcontroller; it is not part of V1.

## 3. Data contract and future compatibility

V1 keeps the existing three-slot `FusionWindow` contract:

```text
imu     (100, 6)   valid=True when enough wearable samples are present
vision  (20, 99)   valid=True when MediaPipe detects usable pose landmarks
radar   (40, K)    all zeros, radar_valid=False, radar_energy=0.0
```

Keeping the dormant radar slot is deliberate. The future camera + IMU + mmWave
system can activate that slot without changing the window builder, model API,
training loop, dashboard event schema, or present hardware pipeline.

## 4. Practical build order

1. Validate MPU-6050 sampling and timestamped ESP32-to-laptop transport.
2. Flash ESP32-CAM CameraWebServer and validate its browser MJPEG stream.
3. Validate OpenCV capture and MediaPipe Pose extraction from that stream.
4. Synchronize both streams into two-second `FusionWindow`s.
5. Train or load the IMU encoder and train the camera–IMU fusion head on paired
   data such as UP-Fall or local captures.
6. Add live laptop inference, sensor-health reporting, activity confidence,
   dashboard updates, and fall alerts.
7. Test sensor loss: IMU missing, camera stream missing/dark, and recovery.
8. Only after V1 works, integrate mmWave as the third modality.

## 5. V1 training commands

```bash
# Plumbing and simulator checks
python tests/test_pipeline.py
python tests/test_camera_stream.py
python scripts/download_pose_model.py
python scripts/test_esp32_camera.py --host <esp32-cam-ip>
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim

# Real V1 model path
python scripts/pretrain_imu.py
python scripts/train_fusion.py
```

MediaPipe supplies the camera pose features, so raw-camera-model training is
not required. Radar pretraining is optional future-extension work and is not on
the V1 critical path.

## 6. V1 completion criteria

V1 is complete when:

- one wearable ESP32 continuously streams valid MPU-6050 data;
- the ESP32-CAM streams stable OV2640 frames to the laptop;
- the laptop continuously extracts MediaPipe pose landmarks from that stream;
- the laptop creates synchronized two-second camera–IMU windows with radar
  disabled;
- the model produces an activity label and confidence for each window;
- the dashboard shows activity, confidence, camera health, and IMU health;
- a detected fall produces a visible alert; and
- unplugging or degrading one active sensor does not crash the pipeline.

## 7. Later mmWave extension

The final target remains **camera + wearable IMU + fixed mmWave radar**. A
second ESP32 may acquire the radar stream and transmit it to the same laptop.
The laptop then populates the existing radar tensor and changes
`radar_valid=True` for healthy radar windows. This extends V1; it does not
replace it.

## 8. Honest project claims

- V1 is a practical bi-modal camera–IMU HAR and fall-alert prototype.
- MediaPipe performs pose extraction; FusionSense learns temporal sensor fusion.
- Simulator results validate plumbing only, not real-world accuracy.
- Tri-modal accuracy or robustness claims require later paired camera + IMU +
  mmWave captures.
