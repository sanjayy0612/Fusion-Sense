# FusionSense — Project Alignment and Learning-by-Doing Plan

This file is the practical north star for the project. It keeps the prototype,
training plan, hardware decisions, and report claims aligned with the original
idea: **sensor-health-aware multimodal human activity recognition** for fall
monitoring, built step by step instead of as a one-shot perfect system.

## 1. The project idea, in one sentence

FusionSense combines **IMU + mmWave radar + camera-derived pose features** into a
single windowed model that predicts human activity and exposes interpretable
per-sensor trust weights, so the system can keep working when one sensor is weak,
missing, noisy, or degraded.

## 2. What changed and what did not change

| Area | Still the same | Changed for feasibility |
|------|----------------|-------------------------|
| Research idea | Tri-modal HAR with sensor-health-aware fusion and robustness to modality dropout. | Nothing. This remains the core contribution. |
| Model boundary | Every source must produce the same `FusionWindow`. | Nothing. This is still the strongest engineering decision. |
| Hardware | Dedicated sensor gateway for IMU/radar. | Raspberry Pi compute is replaced by laptop inference/API for v1. |
| Vision | Use privacy-preserving features instead of raw pixels. | Use short clips + cached MediaPipe pose arrays to avoid huge video datasets. |
| Training | Pretrain modality encoders, then train fusion attention on paired windows. | Be honest that public paired radar+IMU+camera data is missing; collect a small local tri-modal set later. |

## 3. The correct build order

Do not try to solve hardware, dataset, model training, and dashboard all at the
same time. Build one learning layer at a time:

1. **Pipeline proof:** run simulator and tests to understand `FusionWindow`,
   masking, health values, and robustness reports.
2. **IMU path:** connect/record MPU-6050 data, then compare it with SisFall or
   UCI-HAR windows.
3. **Radar path:** stream LD2410 distance/energy, then map it into the radar
   feature shape expected by the model.
4. **Vision path:** use webcam/phone clips, extract MediaPipe pose, and train on
   cached keypoint arrays instead of storing huge videos.
5. **Model path:** pretrain encoders one by one; train fusion after at least IMU
   and vision are working.
6. **Laptop API path:** wrap inference behind a small local API after the model
   can load a checkpoint and predict on recorded windows.
7. **Demo path:** show live or replayed windows, predicted activity, and trust
   weights changing when one modality is hidden or unplugged.

## 4. What to learn from each phase

| Phase | What you learn | What you can show |
|-------|----------------|-------------------|
| Simulator | Tensor shapes, validity masks, modality dropout, health-conditioned trust. | Robustness table and pipeline sanity. |
| IMU | Sampling rates, accelerometer/gyroscope preprocessing, windowing. | Motion windows and IMU encoder checkpoint. |
| Radar | UART parsing, range/energy features, missing-modality behavior. | Radar stream and radar-valid/radar-invalid windows. |
| Vision | Pose extraction, storage-light video preprocessing, privacy-preserving features. | Keypoint overlays/features and vision encoder checkpoint. |
| Fusion | Why paired data matters, why attention needs synchronized modalities. | Activity prediction plus trust weights. |
| Hardware demo | End-to-end integration and failure/degradation cases. | Arduino/ESP32 gateway feeding laptop inference. |

## 5. Honest report framing

Use this framing in the report or presentation:

- **Contribution:** a lightweight, sensor-health-aware fusion framework for HAR
  that uses interpretable trust weights and modality masking.
- **Prototype:** Arduino/ESP32 streams IMU/radar; laptop handles camera features,
  windowing, model inference, and API output.
- **Training strategy:** separate real datasets for encoder pretraining, paired
  camera+IMU data for fusion, and a small local tri-modal dataset for validation.
- **Dataset limitation:** no public dataset perfectly combines camera + mmWave +
  wearable IMU; this is handled through modular pretraining and honest ablations.
- **Learning-by-doing value:** each phase produces a working artifact even if the
  final tri-modal dataset is small.

## 6. The promise I should keep while helping you

Every future change should protect these principles:

1. Keep the project aligned with the original sensor-fusion idea.
2. Prefer small working steps over unrealistic big-bang training.
3. Avoid fake claims from simulator accuracy.
4. Keep the Arduino/ESP32 as a sensor gateway, not a model device.
5. Keep the laptop API path practical for your RTX 4060 setup.
6. Make the video branch pose-based and storage-light.
7. Make every code/doc change useful for your final demo or report.
