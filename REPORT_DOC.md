# FusionSense: Sensor-Health-Aware Multimodal Human Activity Recognition

## Complete Project Report

| Report field | Value |
|---|---|
| Project | FusionSense |
| Primary application | Posture-transition recognition and fall-alert demonstration |
| General domain | Multimodal Human Activity Recognition (HAR) |
| Current implementation | Laptop-hosted research prototype |
| Active modalities | Waist IMU and third-person ESP32-CAM pose |
| Future modality | mmWave radar, retained as a masked model slot |
| Report date | 16 September 2026 |
| Repository state described | Local `Fusion-Sense-fork`, commit `31ae065` plus the documented notification extension |
| Validation status | Engineering prototype complete; not clinically or deployment validated |

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Problem statement](#2-problem-statement)
3. [Project objectives and completion criteria](#3-project-objectives-and-completion-criteria)
4. [Related work and design motivation](#4-related-work-and-design-motivation)
5. [System architecture](#5-system-architecture)
6. [Hardware acquisition and transport](#6-hardware-acquisition-and-transport)
7. [Timestamp synchronization](#7-timestamp-synchronization)
8. [Preprocessing and model-ready windows](#8-preprocessing-and-model-ready-windows)
9. [Dataset and experimental protocol](#9-dataset-and-experimental-protocol)
10. [Model design](#10-model-design)
11. [Training audit and model selection](#11-training-audit-and-model-selection)
12. [Evaluation results](#12-evaluation-results)
13. [Live inference and dashboard](#13-live-inference-and-dashboard)
14. [Mobile fall notifications](#14-mobile-fall-notifications)
15. [Verification summary](#15-verification-summary)
16. [Operating instructions](#16-operating-instructions)
17. [Repository map](#17-repository-map)
18. [Limitations and safety statement](#18-limitations-and-safety-statement)
19. [Recommended next work](#19-recommended-next-work)
20. [Conclusion](#20-conclusion)
21. [References and project evidence](#21-references-and-project-evidence)

---

## 1. Executive summary

FusionSense is a sensor-health-aware multimodal HAR framework demonstrated
through seven posture transitions, including `stand_to_fall`. The implemented
V1 system combines:

- a waist-mounted, ESP32-controlled six-axis IMU sampled at 50 Hz;
- an AI Thinker ESP32-CAM producing timestamped QVGA JPEG frames at about
  10 FPS;
- MediaPipe full-body pose features extracted on a laptop;
- synchronized two-second IMU and pose windows;
- per-modality temporal encoders and masked cross-modal attention;
- physical sensor-health values and learned sensor-trust weights;
- a strict fall-alert policy, local HTML dashboard, and optional phone
  notification consumer.

Both hardware streams use USB serial in the accepted configuration. The IMU
transmits versioned timestamped rows at 115200 baud, while the camera transmits
CRC-protected JPEG packets at 921600 baud. Independent device clocks are mapped
into laptop monotonic time and corrected using shared visible motion. The
accepted 60-second synchronization session contains 3,000 IMU samples and 601
camera frames, with no malformed IMU rows, capture errors, sequence gaps, or
dropped frames. Its calibrated remaining lag is 0 ms.

The training benchmark is C-MHAD. The currently installed four-subject pilot
contains 363 labelled windows: 257 training windows and 106 windows from held-out
Subject3. The selected frozen-fusion checkpoint achieves 84.91% accuracy and
82.21% macro-F1. At a fall probability threshold of 0.60, it produces 9/9 fall
alerts and 0/97 labelled false positives on this validation subject. The
continuous negative replay contains only 18.17 minutes, so zero observed false
alert episodes still corresponds to a one-sided 95% upper bound of 9.89 false
alerts per hour. These figures are pilot evidence, not a clinical claim.

The final software path is operational: live serial acquisition, unit
conversion, timestamp synchronization, rolling window assembly, inference,
health gating, dashboard updates, and debounced mobile notification logic have
all been implemented. The repository's final Step 9 verification reports 58
passing unit tests and successful connected-device smoke tests. Remaining work
is primarily dataset scale, continuous/background modelling, elderly-domain
evaluation, camera-independent fall detection, and long-duration field
validation.

---

## 2. Problem statement

Fall detection for older adults is safety-sensitive. A single sensing modality
can fail in ways that are difficult to distinguish from normal activity:

- a camera may be dark, blurred, occluded, or unable to see the complete body;
- an IMU may saturate, disconnect, shift orientation, or be mounted poorly;
- network or serial transport may stall or corrupt data;
- clocks on separate embedded devices drift and do not share a common epoch.

FusionSense addresses the engineering side of this problem by combining motion
observed externally through body pose with motion measured directly at the
waist. Each modality carries validity and health information alongside its
features. The model learns a trust distribution over available sensors, while
the runtime applies a conservative safety gate: if either required V1 modality
is invalid, the system reports `DEGRADED` and withholds the activity prediction
instead of silently reporting a normal state.

The project is best described as a reusable multimodal HAR framework whose V1
demonstration is posture-transition and fall recognition. It is not yet an
emergency-response product.

---

## 3. Project objectives and completion criteria

The implemented objectives were:

1. Acquire timestamped IMU data at 50 Hz from an ESP32-controlled waist sensor.
2. Acquire timestamped ESP32-CAM JPEG frames at approximately 10 FPS.
3. Transfer both streams to the laptop without relying on Wi-Fi.
4. Map both device clocks to a shared laptop time domain and validate alignment.
5. Convert live IMU acceleration from `g` to the model's `m/s²` contract.
6. Assemble synchronized two-second windows with fixed tensor shapes.
7. Train IMU-only, vision-only, and masked IMU+camera fusion models.
8. Evaluate activity metrics, fall metrics, false-alert behaviour, modality
   dropout, synchronization sensitivity, latency, and hardware input health.
9. Run live inference and expose alert state, confidence, time, and sensor
   health through an HTML dashboard.
10. Deliver one minimal mobile notification per validated fall episode.

Steps 1–9 and the notification extension are implemented as a local research
prototype. Final safety acceptance is intentionally not claimed.

---

## 4. Related work and design motivation

The project considered the approach reported by Nakabayashi and Saito in
*Multimodal Human Activity Recognition on Edge Devices* (ISMAR-Adjunct 2024,
DOI `10.1109/ISMAR-Adjunct64951.2024.00037`). That work reports approximately
97% precision for five activities using RGB, acceleration, and angular-velocity
streams with attention-based weighting.

That number is not directly comparable to the FusionSense pilot:

| Comparison | Nakabayashi–Saito | FusionSense pilot |
|---|---|---|
| Primary reported quantity | Approximately 97% precision | Accuracy, macro-F1, and thresholded fall metrics |
| Activity scope | Five activities | Seven posture transitions |
| Data described | Approximately 6,522 samples, 10 people | 363 windows, 4 installed subjects |
| Camera representation | RGB images | 33 MediaPipe landmarks × 3 coordinates |
| Inertial representation | Separate acceleration and gyro branches | One combined six-axis branch |
| Split | Per-class 7:2:1 sample split | Complete Subject3 held out |

The paper supports three lessons adopted or retained as future work:

- attention is useful for multimodal weighting;
- acceleration and angular velocity may benefit from separate processing;
- data scale and participant diversity matter at least as much as architecture.

It does not demonstrate that the same 97% will transfer to elderly people,
staged falls, C-MHAD labels, an ESP32-CAM viewpoint, or a waist-mounted
MPU-6500-compatible sensor.

---

## 5. System architecture

### 5.1 End-to-end flow

```text
Waist IMU
  -> ESP32 timestamped samples (50 Hz, USB serial)
  -> unit conversion and clock mapping
  -> 100 x 6 IMU window
                                      +------------------------------+
ESP32-CAM                            | health-aware masked fusion   |
  -> timestamped QVGA JPEG (10 FPS)  | CNN + GRU modality encoders |
  -> USB serial + CRC validation     | 2-layer Transformer         |
  -> MediaPipe Pose                  | learned sensor trust        |
  -> 20 x 99 pose window             +--------------+---------------+
                                                    |
                                             seven activities
                                                    |
                    +-------------------------------+----------------+
                    |                                                |
             HTML dashboard                              mobile fall alert
       MONITORING / FALL_ALERT / DEGRADED               debounced ntfy push
```

### 5.2 Hardware components

| Component | Role | Accepted configuration |
|---|---|---|
| ESP32 development board | Reads and timestamps waist IMU | CP210x USB-UART, normally COM16 in the latest assignment |
| Six-axis IMU | Linear acceleration and angular velocity | `WHO_AM_I=0x70`, MPU-6500-compatible register layout; ±2 g and ±250 dps |
| AI Thinker ESP32-CAM | Captures third-person images | OV2640, QVGA 320×240 JPEG, PSRAM enabled |
| ESP32-CAM-MB | Programs/powers camera and provides serial transport | CH340 USB serial, normally COM14 |
| Windows laptop | Synchronization, pose extraction, inference, dashboard | Python 3.11 project runtime; CPU or CUDA inference |
| Mobile phone | Receives optional fall alert | ntfy Android/iOS application or compatible client |

COM assignments are not treated as identities. The live runner can recover a
unique Windows reassignment using the USB VID/PID of each adapter.

### 5.3 Software data contract

All loaders and runtime paths emit `FusionWindow`, defined in
[`fusionsense/contract.py`](fusionsense/contract.py). With the current
configuration, each window contains:

| Input | Shape | Meaning |
|---|---:|---|
| IMU | `100 × 6` | 2 s at 50 Hz: `ax, ay, az, gx, gy, gz` |
| Radar | `40 × 8` | Reserved future slot; zero-filled and invalid in V1 |
| Vision | `20 × 99` | 2 s at 10 FPS: 33 pose landmarks × XYZ |
| Validity | 3 Boolean values | IMU, radar, and camera availability |
| Health | 3 values in `[0,1]` | IMU health, radar energy, and image quality |

The seven class labels are:

1. `stand_to_sit`
2. `sit_to_stand`
3. `sit_to_lie`
4. `lie_to_sit`
5. `lie_to_stand`
6. `stand_to_lie`
7. `stand_to_fall`

These are transition labels, not persistent `standing`, `sitting`, or `lying`
states.

---

## 6. Hardware acquisition and transport

### 6.1 IMU firmware and verification

The wearable ESP32 emits the versioned packet:

```text
IMU,1,device_id,session_id,seq,t_device_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

The device samples at 50 Hz and includes a sequence number and 64-bit device
capture timestamp. It accepts `SESSION`, `SYNC`, and `INFO` commands over the
same persistent USB serial connection. Firmware calibration estimates
accelerometer and gyroscope bias while the sensor remains stationary.

Standalone verification result:

| Metric | Result |
|---|---:|
| Duration | 59.98 s |
| Samples | 3,000 |
| Effective rate | 50.0 Hz |
| Mean sample interval | 20.0 ms |
| Invalid rows | 0 |
| Missed slots | 0 |
| Sequence gaps | 0 |
| Timestamp regressions | 0 |
| Mean stationary acceleration magnitude | 0.99962 g |
| Mean gyroscope magnitude | 0.12035 dps |
| Gate result | PASS |

A transient earlier run accumulated I²C read failures despite correct
scheduling. The firmware therefore uses a conservative 100 kHz I²C clock and
reports both interval and cumulative read errors. Stable soldered power, SDA,
SCL, and ground connections remain necessary.

### 6.2 Camera firmware and verification

The accepted camera path does not use Wi-Fi. Firmware in
[`hardware/esp32_cam/fusionsense_camera_serial`](hardware/esp32_cam/fusionsense_camera_serial)
captures QVGA JPEG frames and transmits binary `FSC1` packets over the
ESP32-CAM-MB at 921600 baud. Each packet preserves:

- schema/header version;
- device and session identity;
- frame sequence;
- device capture timestamp in microseconds;
- JPEG dimensions and payload size;
- JPEG CRC for corruption detection.

Camera-only USB verification result:

| Metric | Result |
|---|---:|
| Duration | 30.0029 s |
| Frames | 302 |
| Effective rate | 10.0324 FPS |
| Resolution | 320×240 |
| Capture interval p95 / maximum | 108.095 / 108.153 ms |
| Capture errors | 0 |
| CRC errors | 0 |
| Sequence gaps / dropped frames | 0 / 0 |
| Gate result | PASS |

The Wi-Fi MJPEG implementation remains in the repository as a fallback and
previously passed at 9.32 FPS, but variable delivery latency and stalls made
USB serial the authoritative V1 transport.

### 6.3 Transport resilience

The live laptop runner adds defensive handling around the physical streams:

- stale COM assignments are resolved through unique USB VID/PID matches;
- isolated camera CRC failures are logged and discarded;
- five consecutive corrupt camera packets stop the run as a transport fault;
- malformed IMU rows are counted and rejected;
- dashboard JSON uses atomic replacement with retries for temporary Windows or
  OneDrive locks;
- Arduino Serial Monitor must be closed so the Python process can own each COM
  port.

---

## 7. Timestamp synchronization

Each ESP32 clock begins from its own boot-relative microsecond counter. Arrival
order cannot be used as capture time because USB buffering and JPEG transfer
add variable delay. FusionSense uses an affine mapping for each device:

```text
host_ns = scale × device_us × 1000 + offset_ns
```

The IMU mapping uses request/response clock probes. The camera mapping uses a
lower envelope of serial header arrival observations because camera transfer is
one-way. Three sharp side-to-side movements visible to the camera and measured
by the IMU provide a shared-motion signal that resolves the remaining absolute
camera offset through cross-correlation.

The first Wi-Fi combined attempt was correctly rejected because it contained a
malformed IMU row, a 44.18 ms camera clock-fit residual p95, and a 2.74 s camera
stall. The collector was then hardened and the USB-only session
`data/recordings/fusion_usb_step4_20260819` became the accepted artifact.

### 7.1 Accepted synchronization result

| Metric | IMU | Camera |
|---|---:|---:|
| Samples/frames | 3,000 | 601 |
| Device rate | 50.0 Hz | 10.005 FPS |
| Clock-fit residual p95 | 7.61 ms | 14.89 ms |
| Transport latency median | 7.95 ms | 297.41 ms |
| Transport latency p95 | 14.88 ms | 315.93 ms |
| Maximum window resampling skew | 2.751 ms | 53.993 ms |

Additional alignment evidence:

- shared-motion correlation: `0.59719`;
- applied camera offset correction: `-200 ms`;
- remaining calibrated alignment lag: `0 ms`;
- malformed rows, capture errors, drops, and sequence gaps: `0`;
- top-level synchronization result: `PASS`.

The JPEG transport delay is much larger than the model compute time, which is
why device capture timestamps—not receive time—drive window construction.

---

## 8. Preprocessing and model-ready windows

### 8.1 Unit conversion

C-MHAD acceleration is represented in `m/s²`, while the live ESP32 firmware
reports acceleration in `g`. FusionSense converts only the three acceleration
channels:

```text
acceleration_m_s2 = acceleration_g × 9.80665
```

Gyroscope channels remain in `deg/s`. The pipeline checks column names, finite
values, and physical magnitude before applying the training-set normalization.
This prevents an omitted or duplicated unit conversion from silently reaching
the model.

### 8.2 Window assembly

The accepted 60-second hardware session produces two-second windows with a
one-second stride. IMU and pose samples are resampled on the mapped capture-time
grid, not by index or arrival order.

Verified Step 5 result:

| Metric | Result |
|---|---:|
| Total model-ready windows | 58 |
| IMU-valid windows | 58 |
| Windows with at least one usable modality | 58 |
| Strictly bimodal/full-pose-valid windows | 24 |
| IMU coverage | 100% |
| Camera coverage | 100% |
| Converted acceleration median | 9.797 m/s² |
| Validation result | PASS |

Thirty-four windows are withheld from strict live inference because they do not
contain enough valid full-body camera poses. The current safety gate requires
valid IMU input and at least 50% valid pose frames in the camera window.

---

## 9. Dataset and experimental protocol

FusionSense V1 uses the C-MHAD Transition Movements Application dataset. The
full transition subset is approximately 47 GB and contains 12 subjects, ten
roughly two-minute recordings per subject, 640×480 camera video at 15 FPS,
50 Hz IMU streams, and transition annotations.

Only Subject1–Subject4 are installed for the current pilot. Preparation converts
video to MediaPipe pose and synchronized IMU/vision windows, saving the compact
cache at `data/processed/cmhad_windows.npz`.

The pilot protocol is subject-grouped:

| Partition | Windows | Subjects |
|---|---:|---|
| Training | 257 | Installed training subjects |
| Validation | 106 | Entire Subject3 held out |
| Total | 363 | Four installed subjects |

Normalization is fitted only on training subjects. IMU and camera encoders are
trained from random initialization for C-MHAD; an older SisFall checkpoint is
not reused. Holding out a complete subject reduces direct window leakage, but
Subject3 has been observed during iterative development, so it is a validation
set rather than a final untouched test set.

---

## 10. Model design

### 10.1 Modality encoders

Each modality encoder transforms a temporal tensor into one 128-dimensional
token:

```text
Input sequence
 -> Conv1D(in_channels -> 64, kernel 5) + ReLU
 -> Conv1D(64 -> 128, kernel 5) + ReLU
 -> GRU(128 -> 128)
 -> final hidden state token
```

The active IMU encoder consumes six channels. The vision encoder consumes 99
pose coordinates. The radar encoder remains instantiated for forward
compatibility but receives zeros and `radar_valid=False` in V1.

### 10.2 Cross-modal fusion

The three tokens receive learned modality embeddings and enter a two-layer,
four-head Transformer encoder with model dimension 128 and feed-forward
dimension 256. Invalid modalities are masked from both attention and pooling.

Each transformed token receives a learned pool score. When health conditioning
is active, the log of the corresponding physical health value is added to this
score. A masked softmax produces the interpretable sensor-trust distribution:

```text
trust = softmax(learned_score + log(sensor_health))
```

The trust-weighted token sum enters a layer-normalized seven-class linear head.
This design separates two ideas:

- **validity** decides whether a modality may participate at all;
- **health/trust** decides how strongly a participating modality contributes.

### 10.3 Training stages

1. Train complete IMU-only and vision-only classifiers on the same paired
   C-MHAD split.
2. Load the temporal encoders into the fusion network.
3. Train masked cross-modal attention and the fusion classifier.
4. Apply training-only modality dropout without ever dropping all valid inputs.

The original fusion used 30% modality dropout. A later controlled end-to-end
fine-tuning candidate used 15% dropout.

---

## 11. Training audit and model selection

### 11.1 Baselines and original fusion

| Model | Accuracy | Macro-F1 | Fall recall |
|---|---:|---:|---:|
| IMU only | 61.32% | 43.56% | 0% |
| Vision only | 87.74% | 86.49% | 100% |
| Original masked fusion | 83.96% | 80.91% | 100% |

The vision-only baseline is stronger than the original fusion and remains the
best pilot model by overall accuracy. This negative result is retained rather
than hidden.

### 11.2 Corrected training bug

The initial modality-dropout implementation converted NumPy arrays with
`torch.from_numpy` and then zeroed tensors in place. The tensors shared storage
with the source `FusionWindow`, so augmentation progressively erased training
data. The dataset adapter now clones tensors before applying dropout, and a
regression test protects the invariant.

### 11.3 Corrected candidates

| Variant | Accuracy | Macro-F1 | Fall precision | Fall recall |
|---|---:|---:|---:|---:|
| Corrected frozen encoders | 84.91% | **82.21%** | 100% | 100% |
| End-to-end fine-tuned, 15% dropout | **86.79%** | 82.01% | 100% | 100% by argmax |
| Vision only | **87.74%** | **86.49%** | 100% | 100% |
| Engineered 14-channel IMU | 62.26% | 43.95% | 0% | 0% |
| Pose plus temporal velocity | 85.85% | 86.50% | 100% | 100% |

Although end-to-end fine-tuning improves accuracy, its calibrated fall
probability is too low: at the deployment alert threshold of 0.60 it alerts on
only 4/9 falls. The selected Step 9 checkpoint is therefore the corrected
frozen model:

```text
checkpoints/cmhad_step7_bimodal_dropout_fixed/fusionsense_cmhad.pt
```

Model selection prioritizes thresholded fall recall over a small improvement in
general accuracy.

---

## 12. Evaluation results

### 12.1 Labelled validation and continuous negatives

All alert results use a `stand_to_fall` probability threshold of 0.60.

| Model | Accuracy | Macro-F1 | Fall precision | Fall recall | Continuous false-alert episodes |
|---|---:|---:|---:|---:|---:|
| IMU-only baseline | 61.32% | 43.56% | Undefined; no alerts | 0/9 | 0 |
| Vision-only baseline | 87.74% | 86.49% | 9/9 | 9/9 | 0 |
| **Selected corrected frozen fusion** | **84.91%** | **82.21%** | **9/9** | **9/9** | **0** |
| End-to-end fine-tuned fusion | 86.79% | 82.01% | 4/4 | 4/9 | 0 |

The selected model has 0/97 labelled false positives. Its 9/9 precision and
recall estimates both have an approximate Wilson 95% interval of 70.09%–100%,
which reflects the very small number of falls.

The continuous background evaluation contains 545 non-overlapping negative
windows, or 0.30278 hours. It excludes windows from one second before a fall
through the end of the recording, preventing post-fall lying from being
incorrectly counted as normal negative time. Zero false-alert episodes were
observed, but the one-sided 95% upper bound remains 9.89 alerts/hour. Roughly
29.96 zero-event hours would be needed merely to place that bound below
0.1 alerts/hour.

### 12.2 Modality dropout

| Selected-model input | Accuracy | Macro-F1 | Fall alerts at 0.60 |
|---|---:|---:|---:|
| IMU + camera | 84.91% | 82.21% | 9/9 |
| IMU only | 29.25% | 25.03% | 0/9 |
| Camera only | 86.79% | 85.99% | 9/9 |

The selected model is camera-dependent and does not achieve safe graceful
degradation. Therefore, the runtime does not issue an IMU-only fall decision;
camera loss becomes `DEGRADED`.

### 12.3 Synchronization sensitivity

A synthetic perturbation shifts IMU data from `-500 ms` to `+500 ms` within
each labelled two-second window. Across that range, the selected model retains
9/9 fall alerts with no labelled false positives. Worst overall accuracy is
83.96% and worst macro-F1 is 80.92%. This robustness experiment supplements but
does not replace measured hardware clock synchronization.

### 12.4 Latency

| Operation | Median | p95 |
|---|---:|---:|
| Fusion model, CUDA batch 1 | 3.31 ms | 4.22 ms |
| Fusion model, CPU batch 1 | 6.59 ms | 8.08 ms |
| JPEG decode + MediaPipe pose | 14.35 ms | 15.67 ms |

The two-second observation window, one-second live stride, and camera transport
delay dominate model computation. Only 38% of 100 sampled hardware frames
contained a valid full-body pose, reinforcing the need for placement guidance
and a visible health state.

### 12.5 Recorded hardware replay

The accepted hardware session yields 58 windows, of which 24 satisfy the strict
bimodal gate. The selected checkpoint produces zero alerts on those 24 eligible
windows, with maximum fall probability approximately 0.00110. This recording is
unlabelled and verifies the execution path, not activity accuracy or fall
recall.

---

## 13. Live inference and dashboard

[`scripts/run_live_fall_pipeline.py`](scripts/run_live_fall_pipeline.py) opens
both serial devices, establishes a live session, maps device timestamps into
laptop time, assembles one two-second window per second, performs unit and
health checks, runs the selected checkpoint, and atomically publishes
`dashboard/live_output.json`.

Every event contains:

- predicted transition and all seven probabilities;
- prediction confidence and `stand_to_fall` probability;
- UTC event timestamp and inference latency;
- IMU and camera validity;
- physical health and learned sensor trust;
- sampling coverage, skew, gaps, and transport diagnostics;
- the model/checkpoint and configured alert threshold.

The dashboard exposes three states:

| State | Meaning |
|---|---|
| `MONITORING` | Both required inputs are valid and no fall threshold is crossed |
| `FALL_ALERT` | Both inputs are valid and fall probability is at least 0.60 |
| `DEGRADED` | Required input or full-body pose is missing; prediction is withheld |

It polls the JSON feed once per second, displays current and historical events,
uses red for fall alerts and amber for degraded input, and clearly distinguishes
dataset replay, recorded hardware replay, and live ESP32 operation.

Connected Step 9 smoke testing received 290 valid IMU rows and 58 camera frames
in 15 seconds and emitted four timestamped windows. With no complete person in
view, all four correctly became `DEGRADED`. A later resilience test intentionally
requested stale COM17, auto-selected COM16, discarded one real CRC-damaged
camera frame, generated three windows, and exited without serial or dashboard
write failure.

---

## 14. Mobile fall notifications

[`scripts/send_mobile_notifications.py`](scripts/send_mobile_notifications.py)
is a separate consumer of the dashboard JSON. It cannot modify the model or
bypass the runtime health gate. A phone notification is eligible only when:

- the source is the actively running `live_esp32` pipeline;
- the event is no more than 15 seconds old;
- the class is `stand_to_fall` and probability is at least the configured
  threshold;
- the event state is `FALL_ALERT`;
- IMU and camera pose are both valid.

One fall episode produces one urgent ntfy notification containing confidence
and UTC time. Repeated fall windows are deduplicated. Two subsequent valid
`MONITORING` windows re-arm delivery, while `DEGRADED` windows do not. A
30-second cooldown and deterministic provider sequence ID provide additional
duplicate protection. State persists in
`dashboard/notification_state.json`.

The notification component has automated and offline feed tests. End-to-end
phone delivery additionally depends on an Internet connection, ntfy service,
phone configuration, laptop power, and a private unguessable topic. It is not
a guaranteed emergency channel.

---

## 15. Verification summary

| Gate | Evidence | Status |
|---|---|---|
| IMU sampling | 3,000 samples, 50 Hz, no gaps or invalid rows | PASS |
| Camera USB transport | 302 QVGA frames, 10.0324 FPS, no CRC/drop errors | PASS |
| Concurrent synchronization | 3,000 IMU + 601 camera; 0 ms remaining lag | PASS |
| Unit conversion | `g × 9.80665`; accepted median 9.797 m/s² | PASS |
| Model-ready windowing | 58 windows with expected shapes and coverage | PASS |
| Recorded fused inference | 24 eligible bimodal windows, no runtime errors | PASS |
| Step 7 training audit | destructive dropout bug fixed and regression tested | PASS |
| Step 8 pilot evaluation | metrics, dropout, timing, latency, and negatives reported | PASS for research demonstration |
| Live dashboard | connected smoke tests and HTTP/JSON checks | PASS |
| Notification logic | strict gate, deduplication, re-arm, provider request tests | PASS offline |
| Automated tests | 58-unit-test Step 9 suite reported passing | PASS |
| Clinical/deployment validation | insufficient subjects, falls, negative hours, and elderly evidence | NOT COMPLETE |

---

## 16. Operating instructions

Run commands from:

```powershell
cd "C:\Users\SHIRDITHAN\OneDrive\Desktop\fusionsense\Fusion-Sense-fork"
```

Close both Arduino Serial Monitor windows before accessing the devices.

### 16.1 Live inference

```powershell
.\.runtime\python311\python.exe -u .\scripts\run_live_fall_pipeline.py
```

The current defaults are IMU COM16 and camera COM14. The runner attempts USB
identity-based recovery if Windows reassigns a missing port.

### 16.2 Dashboard

In a second terminal:

```powershell
.\.runtime\python311\python.exe -u .\scripts\serve_dashboard.py `
  --data live_output.json --open
```

Keep the monitored person's complete body visible in the 320×240 camera image.

### 16.3 Mobile notification consumer

Install ntfy on the receiving phone, subscribe to a long unguessable topic, and
set the same topic only in the notification terminal:

```powershell
$env:FUSIONSENSE_NTFY_TOPIC = "your-long-random-topic"
.\.runtime\python311\python.exe .\scripts\send_mobile_notifications.py --test-notification
.\.runtime\python311\python.exe -u .\scripts\send_mobile_notifications.py
```

The dashboard server is optional for phone delivery, but the live inference
pipeline must be running.

### 16.4 Mentor dataset replay

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode mentor
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py `
  --data mentor_demo.json --open
```

### 16.5 Accepted hardware replay

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode recorded `
  --session-dir .\data\recordings\fusion_usb_step4_20260819
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py `
  --data session_output.json --open
```

### 16.6 Full automated test suite

```powershell
.\.runtime\python311\python.exe -m unittest discover -s tests -v
```

---

## 17. Repository map

| Path | Purpose |
|---|---|
| `fusionsense/contract.py` | Canonical `FusionWindow` and seven labels |
| `fusionsense/config.py` | Window rates, dimensions, model size, and dropout |
| `fusionsense/data/imu_units.py` | Unit conversion and physical-scale validation |
| `fusionsense/data/camera_serial.py` | Timestamped CRC-protected camera packet reader |
| `fusionsense/data/clock_sync.py` | Affine clock mapping and timestamp alignment |
| `fusionsense/data/recorded_session.py` | Recorded hardware to synchronized model windows |
| `fusionsense/data/live.py` | Rolling live window assembly |
| `fusionsense/data/vision_extractor.py` | MediaPipe pose extraction and quality |
| `fusionsense/models/encoders.py` | CNN+GRU modality encoders |
| `fusionsense/models/fusion.py` | Masked Transformer and health-conditioned trust |
| `fusionsense/inference.py` | Checkpoint loading, normalization, probabilities, alert gate |
| `scripts/record_imu_serial.py` | Timestamped IMU recording and validation |
| `scripts/validate_esp32_camera_serial.py` | Camera serial transport gate |
| `scripts/record_fusion_session.py` | Concurrent recording and synchronization |
| `scripts/build_recorded_windows.py` | Step 5 model-ready hardware windows |
| `scripts/train_bimodal_step7.py` | IMU, vision, and masked-fusion training |
| `scripts/evaluate_step8.py` | Full pilot evaluation report |
| `scripts/run_fall_pipeline.py` | Mentor and recorded-session inference |
| `scripts/run_live_fall_pipeline.py` | Live synchronized inference and dashboard feed |
| `scripts/serve_dashboard.py` | Local dashboard web server |
| `scripts/send_mobile_notifications.py` | Debounced ntfy fall notification consumer |
| `dashboard/` | HTML dashboard and generated event feeds |
| `hardware/esp32_firmware/` | Wearable IMU firmware |
| `hardware/esp32_cam/` | ESP32-CAM firmware and instructions |
| `docs/` | Detailed datasets, audits, evaluations, and operation notes |

---

## 18. Limitations and safety statement

The current prototype has important limitations:

1. **Small dataset pilot.** Only four of twelve C-MHAD subjects are installed,
   and only one subject provides the reported validation results.
2. **Validation subject reuse.** Subject3 has been examined during development;
   it is not a locked final test subject.
3. **Nine labelled falls.** Perfect observed fall precision/recall has a wide
   confidence interval and cannot establish reliability.
4. **Short negative duration.** Eighteen minutes cannot establish a low
   false-alert rate for day-long monitoring.
5. **Camera dependence.** IMU-only dropout produces 0/9 alerts. The model must
   abstain when camera pose is invalid.
6. **Pose visibility.** Only 38% of sampled accepted-session frames had valid
   full-body pose. Camera placement and occlusion are major operational risks.
7. **Domain mismatch.** Training uses C-MHAD hardware and participant
   demographics; deployment uses an MPU-6500-compatible sensor, ESP32-CAM, and
   may target older adults.
8. **Transition-only labels.** The model does not directly classify persistent
   standing, sitting, or lying states.
9. **No clinical cohort.** C-MHAD does not validate elderly transfer or actual
   unstaged falls in homes.
10. **Infrastructure dependency.** Laptop, USB, power, Internet, and ntfy may
    fail; mobile notifications are not an emergency service.

FusionSense must therefore be described as a research prototype and mentor
demonstration. It must not be presented as a certified medical device, clinical
monitor, or guaranteed emergency-response system.

---

## 19. Recommended next work

1. Obtain all 12 C-MHAD subjects and rebuild the cache from scratch.
2. Lock a subject-grouped test split before further model selection.
3. Train a continuous detector with background and an `other`/uncertain state,
   rather than forcing every window into a transition.
4. Collect at least approximately 30 hours of representative negative data if
   targeting a zero-event 95% bound below 0.1 false alerts/hour.
5. Safely collect labelled local transitions and staged falls with trained
   assistance, mats, and no uncontrolled falls.
6. Improve the inertial branch through orientation/gain augmentation, timing
   jitter, sensor noise, separate acceleration/gyro encoders, and possibly a
   fall-specific anomaly head.
7. Evaluate RGB, pose, and combined RGB+pose representations on the complete
   subject set.
8. Add probability calibration, uncertainty, temporal confirmation, and
   explicit post-fall state handling.
9. Evaluate elderly activities of daily living and difficult non-falls without
   asking vulnerable participants to perform dangerous falls.
10. Run long-duration hardware soak tests, power-loss tests, camera-occlusion
    tests, notification outage tests, and recovery tests.
11. Add authenticated/self-hosted notification delivery or a managed emergency
    workflow before considering personal data or real monitoring.
12. Add the planned mmWave modality only after collecting synchronized
    tri-modal data; do not claim radar performance from the currently masked
    placeholder.

---

## 20. Conclusion

FusionSense has reached the intended local research-prototype milestone. It
successfully integrates two independently clocked ESP32 sensing nodes, converts
and validates their data, constructs synchronized model inputs, runs a
health-conditioned multimodal network, withholds unsafe degraded predictions,
and exposes results through both a dashboard and a debounced mobile-alert path.

The strongest contribution is not a claim of flawless fall detection. It is an
auditable end-to-end architecture in which timestamps, units, validity, sensor
health, model trust, transport faults, and uncertainty boundaries remain
visible from embedded acquisition to the user interface. The next phase should
focus on evidence: more subjects, continuous negatives, elderly-domain
evaluation, camera-independent fall sensitivity, calibration, and long-duration
field testing.

---

## 21. References and project evidence

### External resources

1. T. Nakabayashi and H. Saito, “Multimodal Human Activity Recognition on Edge
   Devices,” *2024 IEEE International Symposium on Mixed and Augmented Reality
   Adjunct (ISMAR-Adjunct)*, DOI:
   <https://doi.org/10.1109/ISMAR-Adjunct64951.2024.00037>.
2. C-MHAD Transition Movements Application:
   <https://personal.utdallas.edu/~kehtar/C-MHAD.html>.
3. C-MHAD Python reader:
   <https://github.com/HaoranWeiUTD/C-MHAD-Python>.
4. ntfy documentation: <https://docs.ntfy.sh/>.

### Repository evidence

- [Current project plan](docs/CURRENT_PROJECT_PLAN.md)
- [Dataset instructions](docs/DATASETS.md)
- [Step 7 accuracy and safety audit](docs/STEP7_ACCURACY_AUDIT.md)
- [Step 8 pilot evaluation](docs/STEP8_EVALUATION.md)
- [Step 9 live dashboard report](docs/STEP9_LIVE_DASHBOARD.md)
- [Mobile notification contract](docs/MOBILE_FALL_NOTIFICATIONS.md)
- `checkpoints/cmhad_step7_bimodal/step7_metrics.json`
- `checkpoints/step8_pilot_evaluation_v3.json`
- `data/recordings/fusion_usb_step4_20260819/session.json`
- `data/recordings/fusion_usb_step4_20260819/step5_validation.json`
