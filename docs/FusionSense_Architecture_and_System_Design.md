# FusionSense — Architecture & System Design

**Camera–IMU Fusion for Real-Time Human Activity Understanding**

Authors: Sanjay E (24CS0836), Shirdithan Pon (24CS0899)

Document type: System architecture and design decisions

Status: Current V1 plan · Updated: 2026-08-11

FusionSense is a general Human Activity Recognition (HAR) framework demonstrated
through fall detection. The first practical version uses **camera pose and a
wearable IMU**. mmWave radar remains the planned third modality and uses an
existing compatibility seam, but it is not required or active in V1.

## 1. V1 scope

### Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | Acquire timestamped MPU-6050 accelerometer and gyroscope samples through one wearable ESP32. |
| F2 | Receive OV2640 frames from an ESP32-CAM Wi-Fi stream and extract MediaPipe Pose landmarks on the laptop. |
| F3 | Synchronize IMU and pose samples into fixed two-second windows. |
| F4 | Classify each window into an activity such as walking, standing, sitting, lying, or falling. |
| F5 | Fuse camera and IMU dynamically while masking unavailable sensors. |
| F6 | Report activity, confidence, active-sensor health, and fall alerts to a laptop dashboard. |
| F7 | Keep the radar interface dormant with `radar_valid=False` so mmWave can be added later. |

### Non-functional requirements

| ID | Requirement | V1 target |
|----|-------------|-----------|
| N1 | Window length | 2 seconds |
| N2 | Compute location | Laptop, with CPU/GPU fallback |
| N3 | Privacy | Process frames in memory; do not transmit or persist raw video by default |
| N4 | Robustness | A missing camera or IMU must not crash the pipeline |
| N5 | Hardware | Wearable ESP32 + MPU-6050, ESP32-CAM + OV2640, and laptop; no Raspberry Pi |
| N6 | Extensibility | Add mmWave without replacing the V1 data or model interfaces |

V1 assumes one person indoors. Multi-person tracking and mmWave acquisition are
out of scope.

## 2. V1 architecture

```text
MPU-6050 ──I²C──► wearable ESP32 ──USB serial/Wi-Fi──┐
                                                            ├──► Laptop synchronizer
OV2640 ──► ESP32-CAM ──MJPEG──► OpenCV ──► MediaPipe Pose ─┘
                                                               │
                                                               ▼
                                                       FusionWindow (2 s)
                                                      IMU valid, vision valid,
                                                       radar_valid=False
                                                               │
                                                               ▼
                                                     Camera–IMU fusion model
                                                               │
                                                               ▼
                                            Activity + confidence + sensor health
                                                               │
                                                               ▼
                                                    Dashboard + fall alert
```

### Responsibility split

- **Wearable ESP32:** sample the MPU-6050, timestamp readings, and transmit
  framed IMU records. It does not process video or run the ML model.
- **ESP32-CAM:** capture OV2640 frames and publish an MJPEG stream. It does not
  run pose extraction or the FusionSense model.
- **Laptop:** receive the camera stream through OpenCV; own MediaPipe Pose,
  clock correction, windowing, inference, dashboard service, and alerts.
- **Dashboard:** display results; it must not own classification logic.

The two V1 ESP32 boards have separate roles. A later radar gateway is an
additional future node.

## 3. Canonical data contract

The simulator, dataset loaders, live capture, training code, and inference code
all exchange the same `FusionWindow`:

```python
@dataclass
class FusionWindow:
    t_start: float
    imu: np.ndarray       # (100, 6): ax, ay, az, gx, gy, gz
    radar: np.ndarray     # (40, K): zeros throughout V1
    vision: np.ndarray    # (20, 99): 33 MediaPipe landmarks × xyz

    imu_valid: bool
    radar_valid: bool     # always False in V1
    vision_valid: bool

    imu_health: float
    radar_energy: float   # 0.0 in V1
    image_quality: float
    label: Optional[int]
```

The radar fields are retained only for forward compatibility. Every V1 window
must use a zero-filled radar tensor, `radar_valid=False`, and
`radar_energy=0.0`. Validity masking ensures the dormant radar encoder cannot
affect the result.

## 4. Synchronization and windowing

The two active streams run at different rates and use different clocks:

- MPU-6050: 50 Hz target, giving 100 samples per two-second window.
- ESP32-CAM: QVGA JPEG at approximately 10–15 fps initially.
- MediaPipe pose: 10 fps target, giving 20 pose frames per window.
- Window stride: initially 1 second, giving 50% overlap.

The laptop's monotonic clock is authoritative. ESP32 records include a local
timestamp; the receiver estimates clock offset and timestamps arrival. At each
window boundary, the laptop selects samples inside the interval and resamples
them to the fixed contract shape.

If too few IMU samples arrive, set `imu_valid=False` and reduce `imu_health`.
If pose is absent or unusable, set `vision_valid=False` and reduce
`image_quality`. A window with both active modalities invalid must not be sent
to the model as a normal prediction.

## 5. Model architecture

V1 uses two active modality tokens inside the existing three-token model:

```text
IMU window ──► IMU temporal encoder ──► IMU token ──┐
                                                        ├─► masked attention
Pose window ─► pose temporal encoder ─► vision token ─┘          + health prior

Radar zeros ─► radar encoder ─► radar token ─► masked out

masked attention ─► pooled representation ─► classifier
                 └─► camera/IMU trust values for the dashboard
```

The validity mask is mandatory: radar must contribute exactly zero in V1.
Health-conditioned pooling can use IMU clipping/connectivity and image
quality/pose-detection signals. The dashboard should not display radar trust as
an active health reading while radar is disabled.

## 6. Training strategy

1. Use the simulator only for contract, masking, and training-pipeline checks.
2. Pretrain the IMU encoder on SisFall, UCI-HAR, or local MPU-6050 recordings.
3. Use MediaPipe Pose for camera features; do not train a raw image model.
4. Train the fusion attention and classifier on paired camera + IMU data such as
   normalized UP-Fall or local synchronized recordings.
5. Apply modality dropout to the two active modalities so the system learns to
   operate when either camera or IMU is unavailable.
6. Evaluate overall accuracy, macro-F1, fall recall, confidence calibration,
   active-sensor dropout, and laptop runtime latency.

Radar pretraining on RadHAR is optional exploratory work for the later
extension, not a V1 prerequisite.

## 7. Architecture decisions

### ADR-001: V1 modality boundary

**Decision:** V1 is camera + wearable IMU. Radar remains in the contract but is
always invalid.

**Consequence:** the team can complete a real end-to-end system with available
hardware while preserving the final tri-modal direction.

### ADR-002: Compute placement

**Decision:** the ESP32 performs acquisition and transmission only; all pose
extraction, synchronization, ML, and dashboard work runs on the laptop.

**Consequence:** no Raspberry Pi, model export, or edge quantization is needed
for V1.

### ADR-003: Hardware roles

**Decision:** use one wearable ESP32 for the MPU-6050 and a separate ESP32-CAM
for OV2640 streaming.

**Consequence:** sensor sampling and video transport remain isolated; neither
board runs ML. A radar gateway remains deferred.

### ADR-004: Camera representation

**Decision:** use MediaPipe's 33 pose landmarks (`33 × xyz = 99` values) rather
than raw pixels.

**Consequence:** V1 avoids raw-camera training and reduces privacy, storage, and
compute costs.

### ADR-005: Window definition

**Decision:** begin with two-second windows and a one-second stride.

**Consequence:** the model receives motion context with roughly one-second
prediction updates; shorten the stride only after measuring live behavior.

## 8. V1 implementation roadmap

| Phase | Deliverable |
|-------|-------------|
| 1 | ESP32 MPU-6050 sampler and timestamped laptop receiver |
| 2 | ESP32-CAM CameraWebServer + OpenCV + MediaPipe Pose with image/pose health |
| 3 | Two-stream synchronizer emitting valid `FusionWindow`s with radar disabled |
| 4 | Camera–IMU training and evaluation path |
| 5 | Live inference producing activity, confidence, trust, and sensor health |
| 6 | Dashboard, fall alerts, recovery, and active-sensor dropout tests |

## 9. Future mmWave extension

The final target is camera + wearable IMU + fixed mmWave radar. The likely
extension is a second ESP32 or radar-capable gateway at a fixed location. Its
timestamped samples enter the laptop synchronizer, populate the existing radar
tensor, and set `radar_valid=True` only for healthy windows.

This extension must preserve:

- the wearable ESP32 and camera pipeline;
- laptop-owned synchronization and inference;
- the `FusionWindow` and model-call interfaces;
- dashboard event and alert behavior; and
- camera–IMU operation when radar is absent.

Tri-modal accuracy and physical radar-degradation claims require paired local
camera + IMU + mmWave recordings. They are future results, not V1 claims.

## 10. Honest contribution framing

For V1, describe FusionSense as a practical, lightweight, interpretable
camera–IMU HAR and fall-alert prototype with explicit sensor validity and health
conditioning. Describe the dormant radar interface as architectural
extensibility, not as an evaluated V1 modality.

The later research contribution can evaluate whether adding mmWave improves
darkness, occlusion, privacy, and missing-modality robustness without breaking
the working camera–IMU system.
