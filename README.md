# FusionSense

Lightweight, **sensor-health-aware** multimodal Human Activity Recognition (HAR)
for edge devices. The practical V1 fuses **ESP32-CAM pose + wearable IMU**
with a merged-token cross-modal attention model and demonstrates **elderly fall
detection**. mmWave radar is the planned third modality, not a V1 dependency.

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
Stage 1  Prepare the two V1 branches
              enc_imu   <- SisFall / UCI-HAR      (scripts/pretrain_imu.py)
              camera    <- MediaPipe Pose landmarks (no raw-camera training)

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data (sensors time-aligned)
              UP-Fall (camera + IMU)              (scripts/train_fusion.py)
```

Why paired data for Stage 2: attention learns *relationships between modalities
at the same instant*. Separate datasets never show both sensors describing one
moment, so the cross-modal layer needs aligned camera and IMU data.

## V1 hardware and runtime

V1 uses two purpose-specific ESP32 nodes. A wearable ESP32 reads the MPU-6050
and transmits timestamped IMU samples. An ESP32-CAM captures OV2640 JPEG frames
and exposes a Wi-Fi MJPEG stream. OpenCV receives that stream on the laptop,
where MediaPipe Pose extracts landmarks. The laptop synchronizes both streams
into two-second `FusionWindow`s, runs the model, and drives the dashboard and
fall alerts. Radar is zero-filled with `radar_valid=False`; no Raspberry Pi is
involved. See
**[docs/CURRENT_PROJECT_PLAN.md](docs/CURRENT_PROJECT_PLAN.md)**.

## Quick start

```bash
pip install -r requirements.txt          # CUDA torch build for your 4060

# See / sanity-check the data (no torch)
python scripts/viz_windows.py
python tests/test_pipeline.py
python tests/test_camera_stream.py

# Download the official local MediaPipe model once
python scripts/download_pose_model.py

# After flashing CameraWebServer and finding the ESP32-CAM IP
python scripts/test_esp32_camera.py --host 192.168.1.42

# Smoke-test the training pipeline on the simulator (needs torch):
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim

# Real V1: wearable IMU + MediaPipe camera pose
python scripts/pretrain_imu.py
python scripts/train_fusion.py           # paired camera+IMU; radar_valid=False
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
    paired_loader.py   # UP-Fall (camera+IMU)   -> FusionWindows
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
  pretrain_imu.py                  # required V1 encoder pretraining
  pretrain_{radar,vision}.py       # optional/future experiments
  train_fusion.py                  # Stage 2
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

- **V1 now:** wearable MPU-6050 → ESP32 plus OV2640 → ESP32-CAM stream;
  run MediaPipe Pose and fusion on the laptop, then show health and fall alerts.
- **V1 model rule:** keep radar zeroed and masked with `radar_valid=False`.
- **Later extension:** add a fixed mmWave node, enable the existing radar slot,
  and collect a paired tri-modal dataset without replacing the V1 pipeline.
- **Paper extension:** real sensor-degradation study + health-conditioned ablation.
