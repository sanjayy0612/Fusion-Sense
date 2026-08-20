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

Status: hardware-verified `PASS` on 2026-08-19.

The authoritative artifact is
`data/recordings/fusion_usb_step4_20260819/session.json`. It contains 3,000 IMU
samples at 50.0 Hz and 601 QVGA camera frames at 10.005 FPS with no malformed
rows, camera errors, drops, or sequence gaps. Both affine clock maps pass;
shared-motion correlation is 0.597 and remaining lag after the recorded
one-way camera offset calibration is 0 ms.

1. Upload the USB-serial camera firmware only if it is not already running.
2. Connect both boards over USB and close all Arduino Serial Monitor windows.
3. From the `Fusion-Sense-fork` repository run:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM17 --camera-port COM14 --duration 60 --motion-check
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

Status: hardware-session window assembly verified `PASS` on 2026-08-19.

`fusionsense/data/recorded_session.py` now replays the recorded manifests
through one deterministic assembler:

- Two-second windows with one-second stride.
- IMU resampled to `100 x 6` at 50 Hz.
- Camera selected/resampled to 20 frames at 10 FPS, then converted to pose
  landmarks or lightweight visual embeddings on the laptop.
- Every window carries modality-valid flags, coverage, pose-valid ratio, and
  capture-time skew. Windows with no trustworthy modality are withheld from
  model inference.
- ESP32 accelerometer channels are explicitly converted from `g` to the
  C-MHAD contract in `m/s²` (`× 9.80665`) before resampling and model
  normalization. Gyroscope `deg/s` channels are unchanged.
- Dataset headers are validated as `m/s²` + `degrees/s`, and converted live
  input must pass a broad physical-scale check that catches an omitted `g`
  conversion.

Run/repeat with:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_recorded_windows.py `
  --session-dir .\data\recordings\fusion_usb_step4_20260819
```

The accepted output has 58 windows. All 58 have valid IMU and at least one
usable modality; 24 also have valid full-body pose. Shapes are IMU `100 x 6`,
vision `20 x 99`, and masked radar `40 x 8`. IMU/camera timestamp coverage is
100%, maximum sampling skew is 2.751/53.993 ms, and converted acceleration has
a physically valid 9.797 m/s² median. The artifacts are `step5_windows.npz`,
`pose_cache.npz`, and `step5_validation.json` under the passing session.

## Step 7 — baselines and masked fusion

Status: complete on 2026-08-19.

`scripts/train_bimodal_step7.py` trains three explicit classifiers on the same
257-window training set and 106-window held-out Subject3 validation set. Every
source window has both IMU and camera pose; radar remains invalid. Masked fusion
uses 30% modality dropout but never removes all valid inputs.

| Model | Accuracy | Macro-F1 | Fall recall |
|---|---:|---:|---:|
| IMU only | 0.6132 | 0.4356 | 0.0000 |
| Vision only | 0.8774 | 0.8649 | 1.0000 |
| Masked IMU + camera fusion | 0.8396 | 0.8091 | 1.0000 |

Artifacts are under `checkpoints/cmhad_step7_bimodal/`: complete baseline
checkpoints, masked fusion, training-only normalization, labels, training log,
confusion matrices, and `step7_metrics.json`. Fusion did not outperform the
vision baseline on this pilot; that negative result is retained.

A subsequent safety audit fixed destructive in-place modality dropout and
retrained controlled variants. Corrected frozen fusion reaches 0.8491 accuracy
/ 0.8221 macro-F1; end-to-end fine-tuning reaches 0.8679 / 0.8201. Both have
9/9 held-out fall recall by class argmax and no fall false positives, but their
thresholded alert recall differs in Step 8 and camera-only remains stronger
overall. Full analysis and the required data/task changes are in
`docs/STEP7_ACCURACY_AUDIT.md`.

## Step 8 — evaluation

Status: research-pilot evaluation complete on 2026-08-19; clinical/deployment
acceptance remains blocked.

`scripts/evaluate_step8.py` now evaluates fall precision/recall and confidence,
continuous false alerts/hour, modality dropout, synthetic synchronization
shifts, measured hardware synchronization, preprocessing/model latency, and an
unlabelled accepted-hardware replay. `fusionsense/evaluation.py` provides the
tested metric helpers. The corrected continuous-negative cache contains 545
non-overlapping windows (18.17 minutes) and excludes fall plus post-fall time.

At threshold 0.60, corrected frozen fusion produces 9/9 labelled fall alerts,
0/97 labelled false positives, and zero observed continuous false-alert
episodes. Its zero-count 95% upper bound is nevertheless 9.89 alerts/hour due
to the short monitoring duration. It misses all 9 falls under camera dropout,
so Step 9 must require both valid modalities. The fine-tuned candidate is
rejected because it detects only 4/9 falls at the same threshold.

The Step 9 prototype checkpoint is
`checkpoints/cmhad_step7_bimodal_dropout_fixed/fusionsense_cmhad.pt`. See
`docs/STEP8_EVALUATION.md` and
`checkpoints/step8_pilot_evaluation_v3.json`. Subject3 has been observed during
tuning; final selection still requires an untouched subject-grouped test set,
preferably after installing all 12 C-MHAD subjects.

## Step 9 — live inference and dashboard

Status: local research-prototype implementation and connected-hardware
acceptance complete on 2026-08-19.

`scripts/run_live_fall_pipeline.py` reads the IMU and camera COM ports
concurrently, builds synchronized two-second windows every second, converts
acceleration from `g` to `m/s²`, and runs the Step 8-selected corrected frozen
fusion checkpoint. Every event exports alert state, activity confidence, fall
probability, UTC timestamp, inference time, sensor validity/health, sampling
diagnostics, and learned trust to `dashboard/live_output.json`.

The live policy requires valid IMU and full-body camera pose. Missing input is
reported as `DEGRADED` and no model prediction is emitted. The dashboard polls
once per second and distinguishes `MONITORING`, `FALL_ALERT`, and `DEGRADED`.

Mentor demonstration:

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode mentor
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py --open
```

This selected-checkpoint replay evaluates 106 Subject3 windows at 84.91%
accuracy, 0.8221 macro-F1, and 9/9 fall recall. The displayed fall example has
99.4536% fall probability. It is labelled as dataset replay, not live detection.

Recorded ESP32 replay:

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode recorded `
  --session-dir .\data\recordings\fusion_usb_step4_20260819
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py --data session_output.json --open
```

Of 58 assembled accepted-session windows, 24 have both valid modalities and 34
are withheld. The selected checkpoint emits zero alerts with maximum fall
probability 0.001097. A connected 15-second live run received 290 valid IMU
rows and 58 camera frames and created four dashboard events. With no person in
view, all four correctly reported `DEGRADED` because pose validity was zero.

Live operation currently uses COM16/COM14. These are the defaults, and the
runner can recover a unique Windows reassignment by USB VID/PID:

```powershell
.\.runtime\python311\python.exe -u .\scripts\run_live_fall_pipeline.py
.\.runtime\python311\python.exe -u .\scripts\serve_dashboard.py `
  --data live_output.json --open
```

See `docs/STEP9_LIVE_DASHBOARD.md`. The separate debounced mobile-notification
consumer is now implemented in `scripts/send_mobile_notifications.py`. It uses
the same strict validity gate, emits one urgent ntfy message per fall episode,
and does not alter inference or dashboard state. Setup and limitations are in
`docs/MOBILE_FALL_NOTIFICATIONS.md`.
