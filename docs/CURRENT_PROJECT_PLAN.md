# FusionSense — Current Project Plan

This is the single current plan. Older Pi-first and train-everything plans are no
longer the v1 direction.

## 1. V1 priorities

1. **Train radar branch** with RadHAR.
2. **Train IMU branch** with SisFall, UCI-HAR, or local MPU-6050 captures.
3. **Use an open-source camera model** for pose/keypoint extraction; do not train
   a raw camera model in v1.
4. **Train FusionSense fusion attention/head** after radar and IMU checkpoints are
   available.
5. **Set up Arduino/ESP32 hardware after model branches are understood.**

## 2. Final v1 architecture

```text
LD2410 radar ─┐
              ├─ Arduino/ESP32 gateway ── USB serial/WiFi ── laptop
MPU-6050 IMU ─┘                                                │
                                                               │
Laptop webcam/camera ── MediaPipe Pose or MoveNet ─────────────┤
                                                               ▼
                                                   FusionWindow builder
                                                               ▼
                                               FusionSense fusion model/API
```

The microcontroller is only a sensor gateway. The laptop handles pose extraction,
windowing, model training, inference, and API/dashboard output.

## 3. Camera decision

The camera path uses an open-source pose model:

- **Recommended:** MediaPipe Pose Landmarker, because the repo already supports
  33 landmarks × `(x, y, z)` = 99-dimensional pose frames.
- **Alternative:** MoveNet Lightning, if a TensorFlow/TFLite path is preferred.

This is still AI: the open-source model extracts human pose, while FusionSense
learns how to fuse camera pose with radar and IMU signals. We skip raw camera
model training because it adds dataset size, compute, and labeling work without
being the core contribution.

## 4. Training plan

```bash
# Smoke tests first
python tests/test_pipeline.py
python scripts/pretrain_radar.py --sim
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim

# Real v1 branch training
python scripts/pretrain_radar.py
python scripts/pretrain_imu.py

# Fusion after branch checkpoints exist
python scripts/train_fusion.py
```

For v1, do not run `scripts/pretrain_vision.py` as a required step. Keep it only
as an optional experiment for training a small temporal pose head later.

## 5. Dataset plan

| Modality | Dataset/source | Needed now? | Notes |
|----------|----------------|-------------|-------|
| Radar | RadHAR | Yes | Main radar pretraining source. Adapt later for LD2410. |
| IMU | SisFall or UCI-HAR | Yes | SisFall is better for fall detection; UCI-HAR is easier for a first pass. |
| Camera | MediaPipe/MoveNet on webcam/video | Yes, but no raw camera training | Keep only pose/keypoint features where possible. |
| Fusion | UP-Fall or local paired captures | Later | Use after branch checkpoints exist. |

Do not commit raw datasets to git. Keep datasets under `data/raw/` locally and
commit only code, docs, preprocessing utilities, and small placeholders.

## 6. What to claim honestly

- FusionSense is a lightweight, sensor-health-aware multimodal fusion framework.
- V1 trains radar and IMU branches, uses an open-source pose model for camera,
  and trains the fusion attention/head.
- The Arduino/ESP32 gateway is for live sensor acquisition after model training
  is understood.
- Simulator results prove plumbing only, not final real-world accuracy.
- A small local tri-modal capture is still needed for final hardware validation.
