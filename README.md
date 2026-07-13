# FusionSense

Lightweight, **sensor-health-aware** multimodal Human Activity Recognition (HAR)
for edge devices. Fuses **camera + mmWave radar + wearable IMU** with a
merged-token cross-modal attention model, and dynamically trusts the most
reliable sensor per window. Demonstrated on **elderly fall detection**.

> **Framework vs. application:** FusionSense is a *general HAR framework*; fall
> detection is the *demo*. Write it that way — it's reusable and stronger.

## Why this repo runs with no hardware

Everything is built behind one data contract — `FusionWindow` (see
`fusionsense/contract.py`). A **simulator** produces `FusionWindow`s directly, so
you can build, train, and validate the entire model today. When real sensors
arrive, you swap only the *data source*; the model, training, and inference code
don't change.

## Quick start

```bash
pip install -r requirements.txt          # install the CUDA torch build for your 4060

# 1. See the data (no torch needed) -> writes window_preview.png
python scripts/viz_windows.py

# 2. Sanity-check the pipeline (no torch needed)
python tests/test_pipeline.py

# 3. Train on simulated data + run the robustness study (needs torch)
python scripts/train.py
```

## What's the "unique angle"?

Not the attention (that's everywhere). The differentiator is **sensor-health
conditioning**: each sensor reports its own reliability for free —
radar gate **energy**, image **quality** (brightness × sharpness), IMU
**clipping/health**. These scalars are carried in every `FusionWindow` and bias
the fusion's trust weights, so the model leans on *physically healthy* sensors.
The trust weights are exported as an interpretable output (great for a dashboard
and for a paper figure). See `models/fusion.py` and the review doc.

Toggle it with `CFG.use_health_conditioning` to run the ablation
(health-conditioned vs. plain attention).

## The headline experiment

`scripts/train.py` prints a **robustness table** — accuracy when each modality is
force-dropped at inference (e.g., "no vision" = a dark room). A model trained
with modality-dropout + health conditioning should degrade *gracefully* where a
naive fusion baseline collapses. That table is your core result.

## Layout

```
fusionsense/
  config.py            # all knobs (window size, rates, dims)
  contract.py          # FusionWindow — the one interface that matters
  data/
    simulator.py       # generates FusionWindows (no hardware)
    dataset.py         # torch Dataset + modality-dropout augmentation
  models/
    fusion.py          # encoders + merged-token attention + health conditioning
  train/
    loop.py            # training/eval
    metrics.py         # accuracy, fall recall, robustness_report
scripts/
  viz_windows.py       # visualize walking vs falling
  train.py             # full run + robustness study
tests/
  test_pipeline.py     # numpy-only pipeline checks
```

## Roadmap

- **V1 (this repo):** working tri-modal edge prototype on simulated data.
- **Next:** pretrain each encoder on public single-modality datasets
  (UCI-HAR / PAMAP2 for IMU; public mmWave HAR; pose/action for vision), then
  fine-tune fusion.
- **Then (hardware):** ESP32 firmware (I²C IMU + UART radar) + Pi camera
  embedding extractor, all emitting the same `FusionWindow`; quantize to
  ONNX/TFLite; measure on-device latency; collect a small real tri-modal set.
- **Paper (V2):** real sensor-degradation study + health-conditioned ablation.
```
```
