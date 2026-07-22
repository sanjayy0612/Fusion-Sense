# FusionSense

Lightweight, **sensor-health-aware** multimodal Human Activity Recognition (HAR)
for edge devices. Fuses **camera + mmWave radar + wearable IMU** with a
merged-token cross-modal attention model, and dynamically trusts the most
reliable sensor per window. Demonstrated on **elderly fall detection**.

> **Framework vs. application:** FusionSense is a *general HAR framework*; fall
> detection is the *demo*. Write it that way — it's reusable and stronger.

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
Stage 1  Pretrain each encoder SEPARATELY on real single-modality data
              enc_imu   <- SisFall / UCI-HAR      (scripts/pretrain_imu.py)
              enc_radar <- RadHAR                 (scripts/pretrain_radar.py)
              enc_vis   <- MediaPipe pose / video (scripts/pretrain_vision.py)

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data (sensors time-aligned)
              UP-Fall (camera + IMU)              (scripts/train_fusion.py)
```

Why paired data for Stage 2: attention learns *relationships between modalities
at the same instant* ("dark → trust radar"). Separate datasets never show all
sensors describing one moment, so the cross-modal layer needs aligned data.

## Quick start

```bash
pip install -r requirements.txt          # CUDA torch build for your 4060

# See / sanity-check the data (no torch)
python scripts/viz_windows.py
python tests/test_pipeline.py

# Smoke-test the WHOLE pretraining pipeline on the simulator (needs torch):
python scripts/pretrain_imu.py --sim
python scripts/pretrain_radar.py --sim
python scripts/pretrain_vision.py --sim
python scripts/train_fusion.py --sim

# Real training: download datasets (docs/DATASETS.md) into data/raw/, then:
python scripts/pretrain_imu.py
python scripts/pretrain_radar.py
python scripts/pretrain_vision.py        # set CFG.vision_dv = 99 first
python scripts/train_fusion.py
```

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
    radar_loader.py    # RadHAR                 -> radar windows
    vision_extractor.py# video -> MediaPipe pose -> vision windows
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
  pretrain_{imu,radar,vision}.py   # Stage 1
  train_fusion.py                  # Stage 2
  viz_windows.py, make_figures.py, make_diagrams.py, baseline_numpy.py
hardware/
  esp32_firmware/      # ESP32 gateway (.ino)
  wokwi/               # in-browser circuit simulation
docs/
  DATASETS.md          # downloads + expected layouts (read this before real training)
tests/test_pipeline.py # numpy-only checks
```

## Roadmap

- **Now:** pretrain encoders on real benchmarks → train fusion on UP-Fall.
- **Hardware:** ESP32 firmware (I²C IMU + UART radar) + Pi pose extractor, all
  emitting the same `FusionWindow`; quantize to ONNX/TFLite; measure Pi latency;
  collect a small real **tri-modal** set (the one thing UP-Fall lacks: radar).
- **Paper (V2):** real sensor-degradation study + health-conditioned ablation.
```
