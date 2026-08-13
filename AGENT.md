# AGENT.md

This file provides guidance to any coding agent which reades the codebase 

## What this is

FusionSense is a **general HAR (Human Activity Recognition) framework**, demonstrated via **elderly fall
detection**. It fuses camera + mmWave radar + wearable IMU with a merged-token cross-modal attention model,
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
python scripts/pretrain_imu.py
python scripts/pretrain_radar.py
python scripts/pretrain_vision.py        # set CFG.vision_dv = 99 first (real MediaPipe pose dim)
python scripts/train_fusion.py

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
Stage 1  Pretrain each encoder SEPARATELY on real single-modality data
              enc_imu   <- SisFall / UCI-HAR      (scripts/pretrain_imu.py, fusionsense/train/pretrain.py)
              enc_radar <- RadHAR                 (scripts/pretrain_radar.py)
              enc_vis   <- MediaPipe pose / video (scripts/pretrain_vision.py)

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data (sensors time-aligned)
              UP-Fall (camera + IMU)              (scripts/train_fusion.py, fusionsense/train/loop.py)
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
    paired_loader.py         # UP-Fall (camera+IMU)   -> FusionWindows
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

- Current focus: pretrain encoders on real benchmarks, then train fusion on UP-Fall.
- Planned: ESP32 firmware (I2C IMU + UART radar) + Raspberry Pi pose extractor emitting the same
  `FusionWindow`; quantize to ONNX/TFLite; measure Pi latency; collect a small real tri-modal dataset (UP-Fall
  lacks radar).
- Paper V2: real sensor-degradation study + health-conditioned ablation.
