# FusionSense — Current C-MHAD Project Plan

## Objective

FusionSense recognizes seven posture transitions using a third-person camera
and a waist-worn six-axis IMU. C-MHAD is the training benchmark; the live
MPU-6050 and external camera reproduce the same sensor topology.

```text
External camera -> MediaPipe pose -> temporal encoder (128-d) --+
                                                               +-> Transformer -> transition
Waist MPU-6050 -> ESP32 -> IMU temporal encoder (128-d) --------+
```

The seven outputs are `stand_to_sit`, `sit_to_stand`, `sit_to_lie`,
`lie_to_sit`, `lie_to_stand`, `stand_to_lie`, and `stand_to_fall`.

## Fixed data contract

- IMU: 100 samples × 6 channels, representing two seconds at 50 Hz.
- Vision: 20 frames × 99 MediaPipe coordinates, representing the same two
  seconds at 10 FPS.
- Radar: retained as a future slot but zero-filled and marked invalid.
- Each C-MHAD annotation becomes one window centred on the transition.
- Train/validation splitting holds out complete subjects.

## Real training

```bash
python scripts/download_pose_model.py
python scripts/check_cmhad.py --raw-root data/raw/cmhad --expected-subjects 4
python scripts/prepare_cmhad.py --raw-root data/raw/cmhad
python scripts/check_cmhad.py --expected-subjects 4 --require-cache
python scripts/train_cmhad.py --stage all --encoder-epochs 20 --fusion-epochs 20 \
  --expected-subjects 4 --output-dir checkpoints/cmhad_pilot4
```

The stages save `enc_imu.pt`, `enc_vis.pt`, and `fusionsense_cmhad.pt` under
the ignored `checkpoints/cmhad_pilot4/` directory. Normalization statistics and the label
order are saved beside them for live inference.

Step 7 adds explicit, directly evaluable baselines rather than discarding their
classifier heads:

```powershell
.\.runtime\python311\python.exe .\scripts\train_bimodal_step7.py `
  --output-dir .\checkpoints\cmhad_step7_bimodal `
  --baseline-epochs 20 --fusion-epochs 20
```

All three models use 257 training windows and the same 106-window held-out
Subject3 split. IMU-only achieved 0.6132 accuracy / 0.4356 macro-F1 / 0.0 fall
recall; vision-only achieved 0.8774 / 0.8649 / 1.0; masked fusion achieved
0.8396 / 0.8091 / 1.0. The fusion model was trained with 30% modality dropout,
radar permanently invalid, and never drops all valid inputs. It does not beat
vision-only on this pilot, which must remain visible in Step 8 reporting.

## Completion criteria

- C-MHAD cache contains all selected subjects and seven classes.
- MediaPipe-invalid windows are measured and investigated if excessive.
- IMU and camera encoders are trained on training subjects only.
- Both encoders start from random initialization; the old SisFall IMU
  checkpoint is not used for C-MHAD training.
- Fusion is evaluated on held-out subjects.
- Validation accuracy, macro-F1, stand-to-fall recall, and modality-dropout
  results are reported.
- The laptop camera and waist MPU-6050 produce the same window shapes during
  the live demonstration.

The model recognizes transitions, not indefinitely persistent posture states.
Simulator results remain plumbing tests and must not be reported as real
performance.

## Live hardware progress

Hardware Step 1 is complete for the standalone IMU transport. The compiled 50
Hz acquisition sketch is compatible with the connected `WHO_AM_I=0x70`
MPU-6500-class device. A live 59.98-second recorder run captured 3,000 samples
at exactly 50.0 Hz with 20.0 ms mean intervals, zero invalid rows, zero missed
slots, zero sequence gaps, and monotonic timestamps. Mean stationary
acceleration magnitude was 0.99962 g and mean gyroscope magnitude was 0.12035
dps. The validator returned `PASS`.

The same acceptance run can be repeated with:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_imu_serial.py --port COM17 --duration 60 --stationary
```

The validator measures effective rate, timestamp monotonicity/gaps,
accelerometer magnitude, gyroscope bias, malformed rows, and returns an explicit
PASS/FAIL report. ESP32-CAM clock alignment is the next phase after this gate.

Camera Step 2 is hardware-validated. The AI Thinker sketch
captures QVGA JPEG frames at a 10 FPS target and embeds a monotonically
increasing sequence plus the ESP32 camera framebuffer's microsecond capture
timestamp in every MJPEG part. The laptop timestamp-aware reader preserves both
fields. The acceptance run delivered 283 frames over 30.255 seconds at 9.32
device FPS, with zero sequence drops and zero capture errors. This establishes
device-local capture timing; mapping the camera clock to the laptop/IMU clock
remains the following synchronization step.

The camera-only recorder is also implemented. It uses one persistent HTTP/TCP
stream and saves the original JPEG payloads with a CSV manifest containing
frame sequence, device capture microseconds, laptop monotonic receive
nanoseconds, dimensions, payload size, and path. A live 60-second recorded
session is the final camera-only gate before concurrent IMU ingestion.

That camera recording gate passed with 581 original QVGA JPEGs over 60.078
device seconds at 9.65 FPS, zero drops, and zero capture errors. The IMU side now
has a versioned persistent serial packet with device/session IDs, scheduled-slot
sequence, and 64-bit microsecond capture time. Its laptop recorder writes the
corresponding laptop monotonic receive time and preserves device health/status.
That IMU-only recording gate passed with 3,000 samples over 59.98 seconds at
50.0 Hz and no invalid rows, missed slots, sequence gaps, or non-monotonic
timestamps. Both standalone transports are verified; concurrent collection and
per-device affine clock mapping are now the next implementation phase.

Step 4 is implemented and hardware-verified using two USB serial links. The IMU
firmware answers `SESSION` and `SYNC`; the camera firmware sends CRC-protected
timestamped JPEG packets. `scripts/record_fusion_session.py` runs independent
camera and IMU readers, fits the IMU request/response map and a camera
lower-envelope affine map, then uses shared visible motion to resolve the
camera's one-way absolute offset. It records raw device time, mapped laptop
capture time, receive time, sequence health, transport latency, and the raw and
applied motion offset.

Close both Serial Monitor windows and run:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM17 --camera-port COM14 --duration 60 --motion-check
```

The first combined hardware attempt proved concurrent acquisition but failed
acceptance: one IMU row was malformed, camera clock-fit residual p95 was 44.18
ms, and camera delivery/capture stalled for 2.74 seconds. The laptop collector
now opens and warms the camera before flushing serial to a complete line,
preserves malformed rows, reuses a persistent camera control connection, and
fails on excessive capture gaps or latency. It also replaces nearest-sample
alignment acceptance with shared-motion cross-correlation. All 28 unit tests
pass; neither ESP32 needs reflashing for these laptop changes.

During the run, perform three sharp side-to-side movements while the IMU is
visible to the camera. The authoritative artifact,
`data/recordings/fusion_usb_step4_20260819/`, passes: 3,000 IMU samples at 50.0
Hz, 601 QVGA frames at 10.005 FPS, no invalid rows/errors/gaps, clock residual
p95 of 7.61/14.89 ms (IMU/camera), shared-motion correlation 0.597, and 0 ms
remaining calibrated lag. Step 4 is complete.

## Laptop inference MVP (2026-08-18)

The laptop pipeline was completed before the passing USB hardware run and is
now ready to consume its synchronized artifacts:

- `fusionsense/data/imu_units.py` defines canonical model units as acceleration
  in `m/s²` and angular velocity in `deg/s`. ESP32 `g` values are multiplied by
  9.80665; gyro values are unchanged. Dataset headers and live physical scale
  are validated before normalization.
- `fusionsense/data/recorded_session.py` uses mapped laptop capture timestamps
  to create two-second windows with one-second stride, 100 IMU samples, 20
  camera-pose frames, modality masks, health, coverage, and skew diagnostics.
- `fusionsense/inference.py` loads the trained C-MHAD checkpoint and
  training-only normalization, then exports seven class probabilities,
  `stand_to_fall` probability, and health-conditioned sensor trust.
- `scripts/run_fall_pipeline.py` generates either a labeled held-out mentor
  replay or a real ESP32 recording replay. `dashboard/` displays the alert,
  confidence, model metrics, sensor validity, and event history.

Step 5 is hardware-session verified. Running `scripts/build_recorded_windows.py`
on `fusion_usb_step4_20260819` generated 58 two-second windows: all 58 have
valid IMU and a usable modality, 24 also pass the vision pose-validity gate,
IMU/camera coverage is 100%, maximum sampling skew is 2.751/53.993 ms, and the
converted acceleration median is 9.797 m/s². `step5_validation.json` reports
`PASS`; `step5_windows.npz` contains the model-ready tensors.

The current selected-checkpoint Subject3 replay reports 84.91% accuracy, 0.8221
macro-F1, and 100% fall recall across 106 windows. The dashboard's selected
fall example scores 99.4536%. This is explicitly labelled a dataset replay and
the four-subject pilot is not a clinical/safety claim.

The accepted recording has also been regenerated with the selected frozen
checkpoint. It evaluates only the 24 windows with valid IMU and camera pose;
34 non-bimodal windows are withheld. Median IMU/camera trust is 85.31%/14.69%
and radar trust is zero. All 24 are predicted `lie_to_stand`, with maximum fall
probability 0.001097 and no alert. This unlabelled recording proves the fused
execution path, not activity accuracy.

Step 7 is also complete. Full baseline classifier checkpoints and the newly
trained masked-fusion checkpoint are under `checkpoints/cmhad_step7_bimodal/`,
with validation metrics and confusion matrices in `step7_metrics.json`.

The Step 7 accuracy audit subsequently fixed a destructive modality-dropout
storage-sharing bug and retrained two controlled candidates. Corrected frozen
fusion reaches 0.8491 accuracy / 0.8221 macro-F1; end-to-end fine-tuning reaches
0.8679 / 0.8201. Both classify all 9 held-out falls correctly with no fall false
positives. These are pilot figures from one observed validation subject, not an
elderly or clinical claim. See `docs/STEP7_ACCURACY_AUDIT.md` before Step 8.

Step 8 research-pilot evaluation is complete. The corrected negative replay
contains 545 non-overlapping windows (18.17 minutes) and now excludes fall and
all post-fall time. The corrected frozen-fusion candidate produces 9/9 fall
alerts, 0/97 labelled false positives, and zero observed continuous false-alert
episodes at threshold 0.60. The short replay still gives a one-sided 95% upper
bound of 9.89 false alerts/hour, so this is not deployment evidence.

The frozen candidate is selected for the Step 9 prototype over end-to-end
fine-tuning, which alerts on only 4/9 labelled falls at threshold 0.60. The
selected model remains camera-dependent: IMU-only dropout yields 0/9 fall
alerts, while camera-only yields 9/9. Measured hardware alignment is 0 ms after
calibration, clock residual p95 is 7.61/14.89 ms for IMU/camera, and selected
model CPU p95 inference is 8.08 ms. Only 38% of 100 hardware frames sampled
across the accepted recording contain a full-body pose. Step 9 must therefore
require both valid modalities, display `DEGRADED` on camera/IMU loss, and avoid
a safety claim. Full results are in `docs/STEP8_EVALUATION.md` and
`checkpoints/step8_pilot_evaluation_v3.json`.

Step 9 is complete as a local research prototype. Inference now defaults to
the Step 8-selected frozen checkpoint and enforces a strict IMU+camera gate.
`scripts/run_live_fall_pipeline.py` concurrently reads the versioned IMU USB
stream and FSC1 camera JPEG stream, maps device timestamps into laptop time,
creates a synchronized two-second window every second, converts acceleration
to m/s², and atomically updates `dashboard/live_output.json`.

The dashboard polls that feed once per second and exposes alert state,
confidence, event timestamp, inference latency, input validity, physical health
scores, sampling diagnostics, and learned sensor trust. It distinguishes
`MONITORING`, `FALL_ALERT`, and `DEGRADED`; missing camera pose or IMU data
withholds the prediction instead of showing a false safe state.

Final connected acceptance used IMU COM17 and camera COM14. A 15-second run
received 290 valid IMU rows and 58 frames, produced four timestamped windows,
and correctly marked all four `DEGRADED` because no body was visible. IMU scale
remained approximately 9.81 m/s², skew was 0.002 ms, and there were no malformed
rows or IMU sequence gaps. Separately, the accepted recording produced 24
strictly bimodal predictions with zero alerts and maximum fall probability
0.001097. See `docs/STEP9_LIVE_DASHBOARD.md`. Mobile notifications are the next
separate feature and must consume debounced state transitions without changing
the model health gate. That separate consumer is now implemented in
`scripts/send_mobile_notifications.py`: fresh strict-bimodal fall transitions
send one urgent ntfy notification, repeated fall windows are deduplicated, and
two valid monitoring windows re-arm delivery for a later fall. Phone pairing
and delivery verification are documented in `docs/MOBILE_FALL_NOTIFICATIONS.md`.

Windows later reassigned the same CP210x IMU to COM16. The live runner now
falls back from a missing requested port to the unique matching USB VID/PID,
retries transient OneDrive locks while atomically publishing dashboard JSON,
and discards isolated CRC-damaged camera packets. A follow-up COM16/COM14 run
verified all three resilience changes and exited normally.
