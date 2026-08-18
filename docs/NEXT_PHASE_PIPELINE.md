# FusionSense live pipeline build and acceptance plan

This document tracks the laptop-hosted IMU + ESP32-CAM fall-detection MVP. A
software path is not hardware-verified until its recorded acceptance metrics
pass on the physical devices.

## Completed standalone gates

- IMU: 3,000 samples over 59.98 seconds at 50.0 Hz, with zero invalid rows,
  missed slots, sequence gaps, or non-monotonic timestamps.
- Camera: 581 original QVGA JPEGs over 60.078 device seconds at 9.65 FPS, with
  zero sequence drops and zero capture errors.

## Step 4 — concurrent capture and clock mapping

Status: revised software complete; live combined hardware rerun pending.

1. The Step 4 camera firmware is already uploaded. The current corrections are
   laptop-only; no firmware upload is required.
2. Power the camera normally and note the IP address. Connect the IMU ESP32 to
   the laptop over USB and close all Arduino Serial Monitor windows.
3. From the `Fusion-Sense-fork` repository run:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM16 --camera-host <esp32-ip> --duration 60 --motion-check
   ```

4. After about ten seconds, perform three distinct sharp side-to-side movements
   while the IMU is clearly visible to the camera.
5. Inspect the generated `data/recordings/fusion_<UTC>/session.json`.

Acceptance requires:

- IMU approximately 50 Hz and camera approximately 10 FPS, with valid monotonic
  timestamps and no sequence gaps.
- At least four round-trip sync observations for each device, spread over at
  least two seconds.
- Absolute fitted drift no greater than 2,000 ppm and p95 fit residual no
  greater than 20 ms.
- Camera p95 capture interval no greater than two target intervals and maximum
  interval no greater than five target intervals.
- IMU latency p95/maximum no greater than 50/250 ms and camera latency
  p95/maximum no greater than 500/1,000 ms.
- Shared motion detected in both modalities, correlation at least 0.25, and
  absolute cross-correlation lag no greater than 50 ms. Nearest-IMU sample
  distance is diagnostic only because a 50 Hz stream nearly guarantees a
  sample within 10 ms.
- Zero malformed IMU rows; if one occurs its exact input and parser reason must
  be present in `malformed_imu.csv`.
- Original timestamps and mapped laptop timestamps present in both manifests;
  missing inputs are reported rather than silently filled.

## Step 5 — time-aligned window assembler

After Step 4 passes, replay the recorded manifests through one deterministic
assembler:

- Two-second windows with one-second stride.
- IMU resampled to `100 x 6` at 50 Hz.
- Camera selected/resampled to 20 frames at 10 FPS, then converted to pose
  landmarks or lightweight visual embeddings on the laptop.
- Every window carries modality-valid flags, coverage, maximum gap, and
  capture-time skew.
- The same assembler must serve both recorded replay and live inference.

## Steps 6–8 — data, model, and alert

Record safe labeled fall/non-fall sessions, split by subject/session, and train
IMU-only, vision-only, then fused baselines. Evaluate fall recall, fall
precision, macro-F1, false alerts per hour, and alert latency. Finally run live
inference and publish fall confidence, time, and sensor health to a local HTML
dashboard with threshold, debounce, cooldown, and a visible research-prototype
warning.
