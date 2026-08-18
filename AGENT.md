# AGENT.md

This file provides guidance to any coding agent which reades the codebase 

## What this is

FusionSense is a **general HAR (Human Activity Recognition) framework**, demonstrated via **posture-transition
and fall detection**. V1 fuses third-person camera pose + waist IMU with a merged-token cross-modal attention model,
and dynamically trusts whichever sensor is currently reliable. Write about it as a reusable framework first,
fall-detection demo second.

## Commands

```bash
pip install -r requirements.txt          # install a CUDA torch build for the target GPU from pytorch.org

# Sanity-check data / pipeline without torch
python scripts/viz_windows.py
python tests/test_pipeline.py            # numpy-only checks, no test framework/runner — just run the file

# Smoke-test the full pretraining pipeline on the simulator (needs torch, no real data)
python scripts/pretrain_imu.py --sim
python scripts/pretrain_radar.py --sim
python scripts/pretrain_vision.py --sim
python scripts/train_fusion.py --sim

# Real training (after downloading datasets per docs/DATASETS.md into data/raw/)
python scripts/check_cmhad.py
python scripts/prepare_cmhad.py
python scripts/train_cmhad.py --stage all

# Torch-free baseline + robustness figure (useful before the GPU model is trained)
python scripts/baseline_numpy.py
```

There is no linter/formatter config and no pytest suite — `tests/test_pipeline.py` is a plain script run
directly with `python`.

## Architecture

### The one interface that matters: `FusionWindow`

`fusionsense/contract.py` defines `FusionWindow`, a synchronized fixed-size snapshot of all three sensors
for one time window. Every data source (simulator, real dataset loaders, and eventually real hardware
capture) emits exactly this object, so everything downstream — model, training, inference — is
hardware-agnostic and source-agnostic.

Per default 2s window config: `imu (100,6)`, `radar (40,K)`, `vision (20,D_v)` — vision is a per-frame
*embedding* (e.g. MediaPipe pose), never raw pixels. Each modality also carries a `*_valid` bool (for
graceful degradation when a sensor is dropped/masked) and a health scalar in `[0,1]`:
`imu_health`, `radar_energy`, `image_quality`. These health scalars are the project's differentiator — see
"Sensor-health conditioning" below.

All shapes/rates/dims are centralized in `fusionsense/config.py` (`CFG`) — change window length, sample
rates, channel dims, model dims, or dataset directory names there, not by hardcoding elsewhere.

### Two data sources, one contract

1. **Simulator** (`fusionsense/data/simulator.py`) — fake `FusionWindow`s for testing pipeline plumbing,
   shapes, masking, and robustness logic only. Its "video" is random numbers — never use it for accuracy
   claims.
2. **Real datasets** — the actual training path, routed through `fusionsense/data/registry.py` (unified
   access with optional simulator fallback). See `docs/DATASETS.md` for downloads/expected layouts before
   attempting real training.

### Training: two independent stages (modular pretraining)

```
Stage 1  Train the two V1 encoders separately on C-MHAD training subjects
              enc_imu   <- C-MHAD waist IMU
              enc_vis   <- C-MHAD MediaPipe pose

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data (sensors time-aligned)
              C-MHAD (third-person camera + waist IMU) (scripts/train_cmhad.py)
```

Stage 2 requires paired (time-aligned) data specifically because attention learns relationships *between*
modalities at the same instant (e.g. "dark → trust radar"); disjoint single-modality datasets never show all
sensors describing one moment. `fusion.py`'s `load_pretrained_encoders` is how Stage 2 picks up Stage 1's
trained encoder weights.

### Sensor-health conditioning (the core research contribution)

Each sensor reports its own reliability alongside its data: radar **gate energy**, image **quality**
(brightness × sharpness), IMU **clipping** fraction. These ride in every `FusionWindow`
(`health_vector()`, ordered `[imu, radar, vision]` to match token order) and bias the fusion model's trust
weights, so it leans on physically healthy sensors rather than just statistically-attended ones. Trust
weights are exported as an interpretable output. Toggle via `CFG.use_health_conditioning` to run the
ablation. The headline experiment (`train_fusion.py`) prints a **robustness table**: accuracy when each
modality is dropped at inference (e.g. "no vision" = a dark room) — graceful degradation there vs. a
collapsing naive baseline is the core empirical result.

### Layout

```
fusionsense/
  config.py            # CFG — window/rate/dim knobs + dataset dir names
  contract.py           # FusionWindow — the data contract
  device.py
  data/
    simulator.py        # fake FusionWindows (plumbing/smoke test only)
    windowing.py         # resample/segment real streams -> fixed windows
    imu_loader.py         # SisFall / UCI-HAR      -> IMU windows
    radar_loader.py        # RadHAR                 -> radar windows
    vision_extractor.py     # video -> MediaPipe pose -> vision windows
    cmhad_loader.py          # C-MHAD annotations/video/IMU -> FusionWindows
    registry.py               # unified access + optional simulator fallback
    dataset.py                 # torch Dataset + modality-dropout augmentation
  models/
    encoders.py          # ModalityEncoder (pretrainable) + EncoderClassifier
    fusion.py              # cross-modal attention + health conditioning + load_pretrained_encoders
  train/
    pretrain.py           # Stage 1 engine (trains one encoder)
    loop.py                 # Stage 2 fusion training/eval loop
    metrics.py                # accuracy, fall recall, robustness_report
scripts/
  pretrain_{imu,radar,vision}.py   # Stage 1 entry points
  train_fusion.py                  # Stage 2 entry point
  viz_windows.py, make_figures.py, make_diagrams.py, baseline_numpy.py
hardware/
  esp32_firmware/        # ESP32 gateway firmware (I2C IMU + UART radar) (.ino)
  wokwi/                 # in-browser circuit simulation
docs/
  DATASETS.md            # dataset downloads + expected on-disk layouts
tests/test_pipeline.py   # numpy-only sanity checks (run directly, not via pytest)
```

## Roadmap context

- Current focus: prepare C-MHAD, then train IMU, pose, and fusion with a subject-isolated split.
- Planned: ESP32 firmware (I2C IMU + UART radar) + Raspberry Pi pose extractor emitting the same
  `FusionWindow`; quantize to ONNX/TFLite; measure Pi latency; collect a small real tri-modal dataset (UP-Fall
  lacks radar).
- Paper V2: real sensor-degradation study + health-conditioned ablation.

## Verified live hardware state (2026-08-17)

- Windows identifies COM16 as the physical CP210x ESP32 port; COM6 is a
  Bluetooth serial link and must not be used for this board.
- The connected IMU reports `WHO_AM_I=0x70`, identifying it as
  MPU-6500-compatible. The Step 1 sketch accepts MPU-6050/6500/9250/9255
  identities that share the required motion-register layout.
- The updated sketch under
  `hardware/esp32_firmware/mpu6050_usb_test/mpu6050_usb_test.ino` produces
  the versioned timestamped IMU packet at 50 Hz. The live recorder acceptance
  run passed with 3,000 samples over 59.98 device seconds at 50.0 Hz, 20.0 ms
  mean intervals, and zero invalid rows, missed slots, sequence gaps, or
  non-monotonic timestamps. Mean stationary acceleration magnitude was
  0.99962 g and mean gyroscope magnitude was 0.12035 dps. Hardware Step 1 and
  the persistent IMU transport are verified.
- A live attempt accumulated 190 I2C read failures in a transient burst while
  scheduler misses remained zero. The firmware was reduced to 100 kHz I2C and
  now reports interval and cumulative error counts. Treat any nonzero read-error
  interval as a physical connection/power problem and rerun the full test.
- ESP32-CAM timestamp synchronization and the live model/dashboard alert path
  remain pending.
- ESP32-CAM Step 2 software is implemented under
  `hardware/esp32_cam/fusionsense_camera/`. It emits QVGA JPEG multipart frames
  at a 10 FPS target with `X-Frame-Sequence` and camera-driver
  `X-Capture-Timestamp-Us` headers. `TimestampedMjpegStream` preserves these on
  the laptop. Compilation and parser tests pass. The live hardware test also
  passed: 283 QVGA frames over 30.255 seconds (9.32 device FPS), monotonic
  sequence/capture timestamps, zero dropped frames, and zero capture errors.
  One stream disconnect is expected when the validator closes the connection.
- `scripts/record_esp32_camera.py` finishes the camera-only transport path. It
  keeps one HTTP/TCP stream open and writes the original JPEG payloads plus a
  manifest containing ESP32 capture time, laptop monotonic receive time,
  sequence, dimensions, size, and path. Validate one recorded session before
  adding concurrent IMU ingestion.
- The camera recorder has now passed live: 581 original QVGA JPEGs over 60.078
  device seconds at 9.65 FPS, with zero drops and zero capture errors.
- The IMU firmware now emits
  `IMU,1,device_id,session_id,seq,t_device_us,ax,ay,az,gx,gy,gz` and accepts
  `SESSION`, `SYNC`, and `INFO` commands over the same persistent USB serial
  connection. `scripts/record_imu_serial.py` saves `imu.csv`, device status, and
  session validation. Compilation, 11 offline tests, and the live stationary
  recording all pass. The verified artifact is
  `data/recordings/imu_20260817T174209Z/`.
- Step 4 software is implemented in `scripts/record_fusion_session.py` and
  `fusionsense/data/clock_sync.py`. Three concurrent workers read serial IMU
  samples, persistent MJPEG frames, and camera clock probes. Each device gets a
  low-RTT affine map into laptop monotonic nanoseconds; raw and mapped capture
  timestamps, receive timestamps, transport latency, sequence health, and
  nearest cross-device skew are preserved and reported.
- The current camera sketch adds a separate port-80 `/session`, `/sync`, and
  `/health` control server plus device/session headers on the port-81 stream.
  It compiles for AI Thinker ESP32-CAM at 32% flash and 17% RAM. Offline parser,
  mapping, recording, and synthetic combined-session tests pass. The firmware
  has been uploaded, but concurrent synchronization still needs one live
  combined recorder PASS before it is hardware-verified.
- The Step 4 camera firmware was subsequently uploaded and one combined run was
  recorded at `data/recordings/fusion_20260817T182717Z/`. It proved both live
  inputs can be captured together, but the authoritative result is `FAIL`: one
  malformed IMU row, camera clock-fit p95 residual 44.18 ms, and a 2.74-second
  camera stall. Do not use this session as synchronized training evidence.
- The laptop collector now addresses those findings without another firmware
  change: warm camera then flush serial to a line boundary, save exact malformed
  rows, reuse a persistent HTTP control connection, enforce camera gap and
  transport-latency gates, and validate alignment through shared physical
  motion cross-correlation. Nearest-sample distance remains diagnostic only.
  The revised suite has 28 passing unit tests. Live acceptance must use
  `--motion-check`, include three visible sharp movements, and return top-level
  `PASS` before Step 5 begins.
