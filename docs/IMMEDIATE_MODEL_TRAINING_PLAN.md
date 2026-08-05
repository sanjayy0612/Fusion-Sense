# FusionSense — Immediate Individual Model Training Plan

Current priority: **train the model branches individually first, then set up the
hardware**. This avoids getting blocked by Arduino wiring or a huge video dataset
before the AI pipeline is understood.

## 1. Training order for now

| Priority | Branch | What to train first | Why |
|----------|--------|---------------------|-----|
| 1 | Radar | `scripts/pretrain_radar.py` on RadHAR | You chose RadHAR as a trainable branch; do it before hardware radar integration. |
| 2 | IMU | `scripts/pretrain_imu.py` on SisFall or UCI-HAR/local MPU-6050 export | Numeric signal is cheap to train and useful for falls. |
| 3 | Camera | Open-source pose model: MediaPipe Pose or MoveNet | Skip camera-model training for v1; use pose/keypoints as features. |
| 4 | Fusion | `scripts/train_fusion.py` after radar + IMU checkpoints exist | Fusion learns how to combine trainable sensors with camera pose features. |

Do not wait for hardware before learning the model. Hardware should come after
you can train/load checkpoints and explain what each branch is doing.

## 2. Camera setup decision: use an open-source model for v1

Current decision: **do not train the camera model in v1**. Use an open-source
pose model and spend training effort on RadHAR + IMU + fusion. The camera branch
is still AI, but it should be **pose-based AI**, not raw-pixel video
classification:

1. Capture short videos from laptop webcam or phone.
2. Run MediaPipe Pose to extract 33 body landmarks per frame.
3. Convert each frame to a 99-dimensional vector: `(x, y, z) * 33`.
4. Segment those pose vectors into `FusionWindow` vision windows.
5. Feed those pose windows into FusionSense. If time allows later, train only a
   tiny temporal head on top of pose features.

This is still AI: the open-source camera model extracts human pose, while your
project learns multimodal fusion, robustness, and fall/activity classification.
The difference is that v1 does not spend time training a raw camera model.

## 3. How much data is enough for the first training pass?

These are practical targets, not final research limits:

| Stage | Minimum to start | Better target for review/demo | Notes |
|-------|------------------|-------------------------------|-------|
| Simulator smoke test | Built-in simulator | Built-in simulator | Only proves plumbing, not accuracy. |
| IMU encoder | 200–500 windows | 2,000+ windows | Public IMU datasets can provide this easily. |
| Camera pose features | 10–20 short clips total for API testing | 20–40 clips per class if training a tiny pose head later | Use MediaPipe/MoveNet; no v1 camera-model training required. |
| Radar encoder | 100–300 windows | 1,000+ windows | Start with available radar dataset or collect repeated LD2410 motions. |
| Fusion | 500 paired windows | 2,000+ paired windows | Start with camera+IMU; add radar later through local tri-modal collection. |

For your first working branch checkpoints, prioritize **clean labels and correct
window shapes** over a massive dataset.

## 4. Where to get data

### IMU

- **SisFall**: best fit for fall detection because it includes falls and ADLs.
  Expected repo layout: `data/raw/sisfall/**/*.txt`.
- **UCI-HAR**: smaller and easier to download, but it has no fall class. Use it
  only to learn the IMU training path.

### Vision

Avoid huge video downloads at first. Use one of these:

1. Your own short webcam/phone clips under
   `data/raw/vision_videos/<label>/<clip>.mp4`.
2. A small subset copied from a public action/fall video dataset.
3. UP-Fall video only if you have storage and only download the parts you need.

Use labels that match the project classes: `walking`, `sitting`, `standing`,
`lying`, `fall`.

### Radar

- **RadHAR** is the current public radar pretraining target, but it can be large
  and its mmWave format may not exactly match the LD2410.
- If RadHAR is too heavy, collect small repeated LD2410 recordings later and use
  simulator/IMU/vision training first.

### Fusion

- Use **UP-Fall** when you are ready for paired camera+IMU fusion.
- UP-Fall is large, so do not make it the first download unless you already have
  storage and time.

## 5. Why raw datasets should not be committed to git

Do **not** push downloaded datasets into the branch:

- public datasets can be hundreds of MB to hundreds of GB,
- some sources require terms, accounts, or citation rules,
- git history becomes huge and hard to clone,
- this repo already expects data under `data/raw/`, which is intentionally not
  committed.

Instead, commit:

1. download/preprocessing instructions,
2. small scripts if needed,
3. `.gitkeep` placeholders,
4. trained checkpoints only if your project rules allow them and file size is
   acceptable.

## 6. Commands to run now

```bash
# Always begin with a smoke test
python tests/test_pipeline.py
python scripts/pretrain_imu.py --sim
python scripts/pretrain_vision.py --sim
python scripts/train_fusion.py --sim

# Then train the branches you chose for v1
python scripts/pretrain_radar.py
python scripts/pretrain_imu.py
# Camera: use MediaPipe Pose or MoveNet features; skip pretrain_vision.py for v1

# Then train fusion
python scripts/train_fusion.py
```
