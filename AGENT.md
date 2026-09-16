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

- Current focus: feed the passing USB combined-hardware recording through the
  completed recorded inference and dashboard path, then collect safe local
  fall/non-fall sessions.
- Planned: ESP32 firmware (I2C IMU + UART radar) + Raspberry Pi pose extractor emitting the same
  `FusionWindow`; quantize to ONNX/TFLite; measure Pi latency; collect a small real tri-modal dataset (UP-Fall
  lacks radar).
- Paper V2: real sensor-degradation study + health-conditioned ablation.

## Verified live hardware state (2026-08-17)

- Windows currently identifies COM17 as the physical CP210x ESP32 port; COM6 is a
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
- ESP32-CAM/IMU timestamp synchronization is hardware-verified over two USB
  serial links. Live replay through the model/dashboard is the next gate.
- A Wi-Fi-independent camera transport is now available under
  `hardware/esp32_cam/fusionsense_camera_serial/`. It sends CRC-protected QVGA
  JPEG packets with sequence and device capture timestamps over the
  ESP32-CAM-MB USB serial link at 921600 baud. The matching laptop gate is
  `scripts/validate_esp32_camera_serial.py`. The live COM14 gate passed with
  302 complete QVGA JPEGs over 30.0029 device seconds at 10.0324 FPS. Capture
  interval p95/max were 108.095/108.153 ms, with zero capture errors, CRC
  errors, dropped frames, sequence gaps, or non-monotonic timestamps. This is
  now the hardware-verified camera transport; retain Wi-Fi as a fallback.
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
  `fusionsense/data/clock_sync.py`. The verified mode concurrently reads the
  IMU on COM17 and binary JPEG packets on COM14. IMU request/response probes and
  the camera serial-header lower envelope produce affine laptop clock maps;
  shared visible motion resolves the one-way camera offset. Raw/mapped capture
  timestamps, receive timestamps, transport latency, sequence health, and the
  raw/applied motion offset are preserved and reported. Wi-Fi remains a
  supported fallback.
- The current camera sketch adds a separate port-80 `/session`, `/sync`, and
  `/health` control server plus device/session headers on the port-8080 stream.
  It compiles for AI Thinker ESP32-CAM at 32% flash and 17% RAM. Offline parser,
  mapping, recording, and synthetic combined-session tests pass. The firmware
  was uploaded and used for the earlier Wi-Fi experiments; the later USB path
  supersedes it as the verified transport.
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
  This was the pre-USB acceptance procedure; retain it as historical context
  for the failed Wi-Fi artifacts below.
- Step 4 subsequently passed on the USB-only path. The authoritative artifact
  is `data/recordings/fusion_usb_step4_20260819/`: 3,000 IMU samples over 59.98
  device seconds at 50.0 Hz and 601 QVGA JPEGs over 59.97 seconds at 10.005
  FPS, with zero malformed rows, missed slots, capture errors, drops, sequence
  gaps, or timestamp regressions. IMU/camera affine residual p95 values are
  7.61/14.89 ms, drift is -46.7/+105.8 ppm, shared-motion correlation is
  0.597, the recorded camera offset correction is -200 ms, and remaining lag
  is 0 ms. `session.json` reports top-level `PASS`; Step 4 is complete.

## Laptop inference and dashboard state (2026-08-18)

- The model contract is acceleration in `m/s²` and angular velocity in `deg/s`.
  C-MHAD already has those units. ESP32 acceleration is converted from `g` by
  multiplying only `ax/ay/az` by 9.80665; `gx/gy/gz` remain in `deg/s`.
  `fusionsense/data/imu_units.py` validates header units, finite values, and
  physical scale so an omitted conversion fails early.
- `fusionsense/data/recorded_session.py` assembles two-second/one-second-stride
  windows from mapped laptop capture timestamps. It resamples IMU to `100x6`,
  camera pose to `20x99`, masks radar, and records coverage/skew diagnostics.
- Step 5 is verified on `fusion_usb_step4_20260819`. The assembler produced 58
  model-ready windows with shapes IMU `100x6`, vision `20x99`, and masked radar
  `40x8`. All 58 have valid IMU and a usable modality; 24 pass the vision gate.
  IMU/camera coverage is 100%, maximum resampling skew is 2.751/53.993 ms, and
  the acceleration median after `g × 9.80665` is 9.797 m/s². The accepted
  artifacts are `pose_cache.npz`, `step5_windows.npz`, and
  `step5_validation.json` in the session directory.
- The repository's original `.venv` targets a removed Python 3.11 install. An
  ignored project-local official Python 3.11.9 embeddable runtime now exists at
  `.runtime/python311/python.exe` and loads the existing PyTorch 2.12/CUDA
  packages and checkpoint without modifying model weights.
- Step 6 recorded inference is verified `PASS`. It requires both IMU and camera
  pose, rejects radar-valid input, and withholds non-bimodal windows. The
  passing session yielded 24 fused predictions and withheld 34 pose-invalid
  windows. Regenerated with the Step 8-selected checkpoint, median learned
  trust is 85.31% IMU and 14.69% camera; radar trust is zero. All predictions
  are `lie_to_stand`, maximum fall probability is 0.001097, and no alert fires.
  This unlabelled capture validates the execution path, not classification
  accuracy. Output: `dashboard/session_output.json`.
- Step 7 is complete under `checkpoints/cmhad_step7_bimodal/`. The new trainer
  uses the same 257/106 window held-out-Subject3 split for all models and saves
  complete `baseline_imu.pt`, `baseline_vision.pt`, and
  `fusionsense_cmhad.pt` artifacts. Held-out accuracy/macro-F1/fall-recall are
  0.6132/0.4356/0.0 for IMU-only, 0.8774/0.8649/1.0 for vision-only, and
  0.8396/0.8091/1.0 for masked fusion. Radar is permanently invalid/zero-trust;
  fusion uses 30% modality dropout without ever dropping all inputs. Fusion is
  worse than vision-only on this pilot, so Step 8 must not claim an accuracy
  gain. See `step7_metrics.json` and `training.log`.
- A post-Step-7 safety audit found and fixed destructive modality dropout:
  `FusionDataset` tensors shared NumPy storage with their source windows and
  were zeroed permanently across epochs. The adapter now clones before
  augmentation and `tests/test_dataset_dropout.py` guards the invariant.
  Corrected fusion reached 0.8491 accuracy / 0.8221 macro-F1; end-to-end
  fine-tuning with 15% dropout reached 0.8679 / 0.8201, with 9/9 held-out falls
  by class argmax and no fall false positives. At the Step 8 probability alert
  threshold of 0.60, fine-tuning retains only 4/9 fall alerts. Camera-only
  remains stronger at 0.8774 / 0.8649. See
  `docs/STEP7_ACCURACY_AUDIT.md`. Do not describe 9/9 falls from one subject as
  clinical, elderly, or flawless validation.
- Step 8 research-pilot evaluation is complete. It uses 106 labelled Subject3
  windows (9 falls), a corrected 545-window/0.30278-hour continuous negative
  replay, synthetic IMU timing shifts, and the accepted USB hardware session.
  Corrected frozen fusion at threshold 0.60 has 9/9 fall recall and precision,
  0/97 labelled false positives, and zero observed continuous false-alert
  episodes. The zero-event 95% upper bound remains 9.89 alerts/hour. Under
  camera dropout it produces 0/9 fall alerts, so this is not a graceful or safe
  degraded mode. Fine-tuned fusion is rejected because it alerts on 4/9 falls.
  Step 9's pilot checkpoint is
  `checkpoints/cmhad_step7_bimodal_dropout_fixed/fusionsense_cmhad.pt`, with a
  strict both-modalities-valid gate. See `docs/STEP8_EVALUATION.md` and
  `checkpoints/step8_pilot_evaluation_v3.json`.
- Step 9 is complete as a local research prototype.
  `fusionsense/inference.py` defaults to the Step 8-selected
  `cmhad_step7_bimodal_dropout_fixed` checkpoint and requires valid IMU plus
  full-body camera pose. `scripts/run_live_fall_pipeline.py` reads the two COM
  ports concurrently, assembles timestamped two-second windows every second,
  and atomically updates `dashboard/live_output.json`. Missing input emits
  `DEGRADED` and withholds prediction. `scripts/run_fall_pipeline.py` retains
  mentor and accepted-recording replays. The live runner tolerates transient
  OneDrive target-file locks, auto-recovers a unique reassigned USB device by
  VID/PID, and discards isolated CRC-damaged camera frames. Five consecutive
  corrupt camera frames remain a fatal transport-health failure.
- `dashboard/index.html` is the simple local alert UI. Generate and run it with:

  ```powershell
  .\.runtime\python311\python.exe -u .\scripts\run_live_fall_pipeline.py
  .\.runtime\python311\python.exe -u .\scripts\serve_dashboard.py --data live_output.json --open
  ```

- The selected-checkpoint mentor JSON uses 106 Subject3 windows: accuracy
  0.8491, macro-F1 0.8221, fall recall 1.0, and selected fall probability
  0.994536. It is explicitly labelled as dataset replay.
- The accepted `fusion_usb_step4_20260819` replay produced 24 strictly bimodal
  predictions from 58 windows, withheld 34 pose-invalid windows, and returned
  `PASS` with no alert; maximum fall probability is 0.001097 under the selected
  checkpoint. A connected 15-second live run produced four correctly
  `DEGRADED` events because no body pose was visible. See
  `docs/STEP9_LIVE_DASHBOARD.md`.
- Mobile notification delivery is implemented as a separate consumer in
  `scripts/send_mobile_notifications.py`. It publishes an urgent ntfy message
  only for a fresh, actively-running live event whose class is `stand_to_fall`,
  whose probability exceeds the configured threshold, and whose IMU and camera
  are both valid. It collapses consecutive fall windows into one episode,
  re-arms after two valid safe windows, persists state, and uses a deterministic
  provider sequence ID to prevent duplicate phone notifications on retry. See
  `docs/MOBILE_FALL_NOTIFICATIONS.md`.
