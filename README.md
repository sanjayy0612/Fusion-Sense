# FusionSense

Lightweight, **sensor-health-aware** multimodal Human Activity Recognition (HAR)
for edge devices. The practical V1 fuses **third-person MediaPipe pose + a
waist-worn IMU** with merged-token cross-modal attention. It recognizes seven
C-MHAD posture transitions, including stand-to-fall. mmWave radar is a future
third modality, not a V1 dependency.

> **Framework vs. application:** FusionSense is a *general HAR framework*; fall
> detection is the *demo*. Write it that way — it's reusable and stronger.

For the current v1 plan, see **[docs/CURRENT_PROJECT_PLAN.md](docs/CURRENT_PROJECT_PLAN.md)**.

## Two ways to run: simulator (plumbing) vs. real data (results)

There is **one data contract** — `FusionWindow` (`fusionsense/contract.py`) — and
two sources that both emit it:

1. **Simulator** — fake data for **testing the pipeline only**. Great for
   validating shapes, masking, and the robustness logic with zero hardware.
   **Not** for real accuracy claims (its "video" is random numbers).
2. **Real datasets** — the actual training path. Each encoder is pretrained on a
   real single-modality benchmark, then the cross-modal attention is trained on
   paired data. See **[docs/DATASETS.md](docs/DATASETS.md)**.

## Training is two stages (modular pretraining)

```
Stage 1  Train both V1 encoders on C-MHAD training subjects
              enc_imu   <- waist IMU windows
              enc_vis   <- MediaPipe pose sequences

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data (sensors time-aligned)
              C-MHAD (camera + waist IMU)          (scripts/train_cmhad.py)
```

Why paired data for Stage 2: attention learns *relationships between modalities
at the same instant*. Separate datasets never show both sensors describing one
moment, so the cross-modal layer needs aligned camera and IMU data.

## V1 hardware and runtime

V1 uses a wearable ESP32 to read the waist-mounted MPU-6050 and transmit
timestamped samples. A fixed external laptop/USB camera observes the complete
person. MediaPipe Pose extracts landmarks, and the laptop synchronizes both streams
into two-second `FusionWindow`s, runs the model, and drives the dashboard and
fall alerts. Radar is zero-filled with `radar_valid=False`; no Raspberry Pi is
involved. See
**[docs/CURRENT_PROJECT_PLAN.md](docs/CURRENT_PROJECT_PLAN.md)**.

### Live IMU status

The wearable ESP32 is currently detected on **COM16** through a CP210x USB-UART
bridge. Its connected IMU reports `WHO_AM_I=0x70` (MPU-6500-compatible), and the
updated hardware test sketch produces scaled seven-field rows at the required
50 Hz (`20 ms` timestamp increments). The timestamped persistent-serial
recorder is now hardware-verified: a 59.98-second stationary run captured 3,000
samples at exactly 50.0 Hz, with zero invalid rows, missed slots, sequence gaps,
or non-monotonic timestamps. Mean acceleration magnitude was `0.99962 g` and
mean gyroscope magnitude was `0.12035 dps`, so hardware Step 1 passes.

A later live run showed a transient burst of 190 I2C read errors even though the
50 Hz scheduler itself missed no slots. The firmware now uses a conservative
100 kHz I2C clock and reports per-window as well as cumulative errors. Stable,
soldered or firmly seated power/SDA/SCL connections are required; recovery after
a burst does not count as a passing stability test.

The timestamped ESP32-CAM Step 2 firmware and laptop multipart reader are
implemented and hardware-verified. A 30.255-second live run delivered 283 QVGA
JPEG frames at 9.32 device FPS, with zero sequence drops and zero capture
errors. See
[`hardware/esp32_cam/README.md`](hardware/esp32_cam/README.md); set Wi-Fi values
only in its ignored `fusionsense_camera/secrets.h` file.

The camera-only recorder keeps that stream on one persistent Wi-Fi TCP
connection and stores original JPEGs plus capture/receive metadata:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_esp32_camera.py --host <esp32-ip> --duration 60
```

The camera recorder is hardware-verified: 581 original QVGA JPEGs over 60.078
device seconds at 9.65 FPS, zero drops, and zero capture errors.

The IMU persistent USB-serial contract and laptop recorder are implemented and
hardware-verified:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_imu_serial.py --port <imu-com-port> --duration 60 --stationary
```

The verified session is under `data/recordings/imu_20260817T174209Z/`; its
`session.json` records the full validation report. Both standalone transports
are therefore ready for concurrent collection and laptop clock mapping.

### Step 4: synchronized bimodal recording

The combined laptop collector is implemented at
`scripts/record_fusion_session.py`. It reads the IMU and camera independently,
performs repeated request/response clock probes, fits
`host_ns = scale * device_us * 1000 + offset_ns` for each ESP32, and writes both
raw device timestamps and mapped laptop capture timestamps. It never aligns
samples by arrival order.

The Step 4 camera firmware is already running: the first combined attempt
proved that its port-80 `/session`, `/sync`, and `/health` endpoints work. The
remaining corrections are laptop-only, so neither ESP32 needs another upload.
After closing Arduino Serial Monitor:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM16 --camera-host <esp32-ip> --duration 60 --motion-check
```

After about ten seconds, perform three distinct, sharp side-to-side movements
while the IMU is clearly visible to the camera. The collector now warms the
camera before flushing the IMU to a complete serial-line boundary, reuses one
persistent camera control connection, saves malformed input in
`malformed_imu.csv`, rejects multi-second camera stalls/excessive latency, and
uses shared-motion cross-correlation for the 50 ms alignment gate. Nearest-IMU
sample distance is diagnostic only. All 28 unit tests pass. Do not mark Step 4
hardware-verified until the revised command produces a live `session.json` with
`"result": "PASS"`.

Close Arduino Serial Monitor, then run the validator from this repository:

```powershell
cd "C:\Users\SHIRDITHAN\OneDrive\Desktop\fusionsense\Fusion-Sense-fork"
.\.venv\Scripts\python.exe .\scripts\validate_imu_stream.py --port COM16 --duration 300
```

## Quick start

```bash
pip install -r requirements.txt          # CUDA torch build for your 4060

# See / sanity-check the data (no torch)
python scripts/viz_windows.py
python tests/test_pipeline.py
python tests/test_camera_stream.py
python -m unittest tests.test_validate_imu_stream -v

# Download the official local MediaPipe model once
python scripts/download_pose_model.py

# After flashing CameraWebServer and finding the ESP32-CAM IP
python scripts/test_esp32_camera.py --host 192.168.1.42

# Smoke-test the training pipeline on the simulator (needs torch):
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim

# Real C-MHAD V1
python scripts/check_cmhad.py --raw-root data/raw/cmhad --expected-subjects 4
python scripts/prepare_cmhad.py --raw-root data/raw/cmhad
python scripts/check_cmhad.py --expected-subjects 4 --require-cache
python scripts/train_cmhad.py --stage all --expected-subjects 4 \
  --output-dir checkpoints/cmhad_pilot4
```

`scripts/pretrain_radar.py` and the radar encoder remain available for the
later camera + IMU + mmWave extension; they are not required to complete V1.

`scripts/baseline_numpy.py` gives a torch-free baseline + the robustness figure
(useful for reviews before the GPU model is trained).

## The "unique angle"

Not the attention (that's everywhere). The differentiator is **sensor-health
conditioning**: each sensor reports its own reliability — radar gate **energy**,
image **quality**, IMU **clipping**. These scalars ride in every `FusionWindow`
and bias the fusion's trust weights, so the model leans on *physically healthy*
sensors. Trust weights are exported as an interpretable output. Toggle with
`CFG.use_health_conditioning` for the ablation.

## The headline experiment

`train_fusion.py` prints a **robustness table** — accuracy when each modality is
dropped at inference ("no vision" = a dark room). Graceful degradation there,
vs. a collapsing naive baseline, is the core result.

## Layout

```
fusionsense/
  config.py            # knobs + dataset dir names
  contract.py          # FusionWindow — the one interface that matters
  data/
    simulator.py       # fake FusionWindows (plumbing/smoke test only)
    windowing.py       # resample/segment real streams -> fixed windows
    imu_loader.py      # SisFall / UCI-HAR      -> IMU windows
    radar_loader.py    # future mmWave extension
    camera_stream.py   # ESP32-CAM URL/local-camera adapter
    vision_extractor.py# MediaPipe Tasks -> 99-value pose frames/windows
    cmhad_loader.py    # C-MHAD camera+waist IMU -> FusionWindows
    registry.py        # unified access + optional simulator fallback
    dataset.py         # torch Dataset + modality-dropout augmentation
  models/
    encoders.py        # ModalityEncoder (pretrainable) + EncoderClassifier
    fusion.py          # attention + health conditioning + load_pretrained_encoders
  train/
    pretrain.py        # Stage 1 engine (one encoder)
    loop.py            # Stage 2 fusion training/eval
    metrics.py         # accuracy, fall recall, robustness_report
scripts/
  prepare_cmhad.py                 # aligned pose+IMU cache
  check_cmhad.py                   # raw/cache validation
  train_cmhad.py                   # IMU -> vision -> fusion training
  download_pose_model.py           # install official pose model asset
  test_esp32_camera.py             # live camera/pose verification
  viz_windows.py, make_figures.py, make_diagrams.py, baseline_numpy.py
hardware/
  esp32_firmware/      # wearable ESP32 + MPU-6050 gateway (.ino)
  esp32_cam/           # ESP32-CAM CameraWebServer setup
  wokwi/               # in-browser circuit simulation
docs/
  CURRENT_PROJECT_PLAN.md # authoritative camera+wearable-IMU V1 plan
  DATASETS.md          # downloads + expected layouts (read this before real training)
tests/test_pipeline.py # numpy-only checks
```

## Roadmap

- **V1 now:** waist MPU-6050 → ESP32 plus a fixed third-person camera;
  run MediaPipe Pose and fusion on the laptop, then show transition confidence
  and stand-to-fall alerts.
- **V1 model rule:** keep radar zeroed and masked with `radar_valid=False`.
- **Later extension:** add a fixed mmWave node, enable the existing radar slot,
  and collect a paired tri-modal dataset without replacing the V1 pipeline.
- **Paper extension:** real sensor-degradation study + health-conditioned ablation.
