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
.\.venv\Scripts\python.exe .\scripts\record_imu_serial.py --port COM16 --duration 60 --stationary
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

Step 4 is now implemented in software. The ESP32-CAM has a separate port-80
control server for `/session`, `/sync`, and `/health`, so clock probes remain
available while the persistent port-81 MJPEG stream is open. The existing IMU
firmware already answers `SYNC` over the persistent serial connection.
`scripts/record_fusion_session.py` runs independent camera, IMU, and camera-sync
workers, retains low-round-trip observations across the session, fits an affine
mapping for each device, and records raw device time, mapped laptop capture
time, receive time, sequence health, transport latency, and inter-device skew.

The revised camera firmware compiles and the offline clock/parser/recording
tests pass. Hardware acceptance remains pending. Re-upload the current camera
sketch, close Serial Monitor, and run:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_fusion_session.py --imu-port COM16 --camera-host <esp32-ip> --duration 60 --motion-check
```

The first combined hardware attempt proved concurrent acquisition but failed
acceptance: one IMU row was malformed, camera clock-fit residual p95 was 44.18
ms, and camera delivery/capture stalled for 2.74 seconds. The laptop collector
now opens and warms the camera before flushing serial to a complete line,
preserves malformed rows, reuses a persistent camera control connection, and
fails on excessive capture gaps or latency. It also replaces nearest-sample
alignment acceptance with shared-motion cross-correlation. All 28 unit tests
pass; neither ESP32 needs reflashing for these laptop changes.

During the revised run, perform three sharp side-to-side movements after ten
seconds while the IMU is visible to the camera. Do not begin the 2-second window
assembler until the combined `session.json` reports `PASS`, both clock fits
pass, and the measured shared-motion lag is at most 50 ms.
