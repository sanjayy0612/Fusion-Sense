# FusionSense — Open-Source Model Shortlist to Reduce Training Work

Goal: reduce training workload without breaking the project idea. The best
practice for this project is **not** to replace everything with one generic
black-box activity model. Instead, use open-source pretrained models where they
are strong, then train only the small FusionSense classifier/fusion head that is
specific to our sensors and labels.

## 1. Current project decision

You chose to **train RadHAR/radar and IMU**, while using an open-source model for
camera. That is a good workload split for v1: radar and IMU stay project-specific,
and the camera uses a proven low-compute pose model.

## 2. Recommended approach

Use this hybrid stack:

| Modality | Use pretrained? | Recommended model/tool | What we still train |
|----------|-----------------|------------------------|---------------------|
| Vision | Yes | MediaPipe Pose Landmarker or MoveNet Lightning | No camera-model training for v1; optional tiny temporal pose head later. |
| Radar | Train | RadHAR dataset/model code | Train the radar branch now, then adapt/calibrate later for LD2410 features. |
| IMU | Train | SisFall / UCI-HAR / local MPU-6050 data | Train the tiny IMU encoder locally. |
| Fusion | Yes, keep small | FusionSense cross-modal attention already small | Train only fusion head/attention after feature extraction. |

This keeps FusionSense as an AI project while avoiding expensive raw-video or
full tri-modal training from scratch.

## 3. Vision options

### Option A — MediaPipe Pose Landmarker (recommended)

Use MediaPipe Pose as the camera model. It outputs body landmarks from images or
video, including 3D-style landmark coordinates. This is the best fit for the
current repo because `vision_extractor.py` already converts video into pose
vectors.

Why it fits:

- low compute on laptop,
- no raw-pixel model training,
- privacy-preserving features,
- directly useful for fall/posture detection,
- compatible with the existing `vision_dv = 99` pose-feature path.

### Option B — MoveNet Lightning

MoveNet Lightning is another good pose model when latency matters. It detects 17
body keypoints and is designed for real-time use on ordinary devices. Use this if
MediaPipe is difficult to install or if a TensorFlow/TFLite path is preferred.

Tradeoff: MoveNet has fewer keypoints than MediaPipe Pose, so it may require a
small adapter in the repo because the current vision path expects 99 dimensions
from 33 landmarks.

### Option C — pretrained video action recognition models

Models such as TSM/TSN, MMAction2 model-zoo networks, or TorchVision video
models can classify RGB actions, but they are less aligned with this project:

- more compute than pose features,
- more storage/bandwidth,
- labels may not include fall/lying in the way we need,
- less privacy-preserving,
- harder to fuse cleanly with IMU/radar health signals.

Use them only as a baseline, not as the main project path.

## 4. Radar options

### RadHAR pretrained classifiers

RadHAR provides mmWave HAR code and pretrained classifiers, but its expected
input is a preprocessed mmWave point-cloud format and the preprocessed dataset is
large. It is useful as a reference or if we use compatible TI mmWave-style point
clouds.

Caution for our hardware: LD2410 is a presence/range/energy radar, not the same
as a full point-cloud radar. Therefore, a RadHAR model cannot be treated as a
plug-and-play replacement for LD2410 without an adapter or local calibration.

Recommended use:

1. Keep FusionSense radar branch simple for LD2410: distance, energy, motion
   state, temporal changes.
2. Use RadHAR as literature/reference or optional pretrained radar baseline.
3. Collect small LD2410 recordings for the exact hardware later.

## 5. IMU options

Open-source IMU HAR repositories exist, but pretrained weights are usually not a
clean replacement because IMU signals depend heavily on:

- sensor placement,
- sampling rate,
- device calibration,
- class labels,
- body orientation,
- accelerometer/gyroscope scaling.

Recommended use:

- Train the tiny IMU encoder locally on SisFall/UCI-HAR/local MPU-6050 windows.
- If time is short, use a classical baseline or shallow neural model first.
- Do not depend on a random pretrained IMU checkpoint unless its sensor placement
  and labels match our project.

## 6. Final recommendation for FusionSense

Use this workload-reduced plan:

1. **Radar:** train the radar branch using RadHAR first.
2. **IMU:** train the small existing IMU encoder on SisFall/UCI-HAR/local data.
3. **Camera:** use MediaPipe Pose or MoveNet as the open-source camera model;
   skip camera model training for v1.
4. **Fusion:** train only the FusionSense attention/head on windows. This is the
   real project contribution and is already lightweight.

This reduces training effort while preserving the project claim: robust,
interpretable multimodal fusion rather than just importing a single black-box
activity classifier.
