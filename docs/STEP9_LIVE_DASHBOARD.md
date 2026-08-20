# Step 9 live inference and dashboard

Status: complete as a local research prototype on 2026-08-19.

## Implemented contract

The laptop now consumes the timestamped IMU USB stream and timestamped
ESP32-CAM JPEG USB stream concurrently, creates a new synchronized two-second
window every second, and runs the Step 8-selected checkpoint:

`checkpoints/cmhad_step7_bimodal_dropout_fixed/fusionsense_cmhad.pt`

Every dashboard event contains:

- alert state: `MONITORING`, `FALL_ALERT`, or `DEGRADED`;
- predicted activity, confidence, and `stand_to_fall` probability;
- UTC window timestamp and inference latency;
- IMU/camera validity, physical health score, sampling diagnostics, and learned
  model trust;
- the strict 0.60 fall-alert threshold and required modality policy.

Live ESP32 acceleration is converted from `g` to `m/s²` with
`m/s² = g × 9.80665` before training normalization. Device timestamps drive
the rolling resampling grid; Step 8 median USB delays are used to place both
device clocks in the laptop time domain.

The safety gate requires valid IMU and at least 50% valid full-body pose frames
in the camera window. If either fails, the model does not issue a prediction or
a false safe state: the event becomes `DEGRADED` with an explanation.

## Dashboard behavior

The existing local dashboard now:

- polls `live_output.json` once per second without caching;
- follows the newest event while allowing the operator to inspect history;
- renders fall alerts in red and degraded input in amber;
- shows event time, confidence, class probabilities, inference time, sensor
  health, modality validity, and learned sensor trust;
- clearly labels mentor replay, recorded ESP32 replay, and live ESP32 modes;
- retains the research-prototype/non-clinical disclosure.

## Run live

Close both Arduino Serial Monitor windows first. The current Windows assignment
is COM16 for the IMU and COM14 for the camera. These are also the runner's
defaults, so no port flags are required. If Windows reassigns a port, the
runner automatically uses the unique matching USB VID/PID and reports the
change in the terminal.

Terminal 1:

```powershell
.\.runtime\python311\python.exe -u .\scripts\run_live_fall_pipeline.py
```

Terminal 2:

```powershell
.\.runtime\python311\python.exe -u .\scripts\serve_dashboard.py `
  --data live_output.json --open
```

Keep the monitored person's complete body visible in the 320×240 camera image.
Press `Ctrl+C` in both terminals to stop. A single CRC-damaged camera packet is
logged and discarded without stopping inference; five consecutive corrupt
packets stop the run. Dashboard JSON publication also retries brief Windows or
OneDrive file locks while preserving atomic reads.

Mentor replay and accepted-recording replay remain available:

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode mentor
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py --data mentor_demo.json --open
```

```powershell
.\.runtime\python311\python.exe .\scripts\run_fall_pipeline.py --mode recorded `
  --session-dir .\data\recordings\fusion_usb_step4_20260819
.\.runtime\python311\python.exe .\scripts\serve_dashboard.py --data session_output.json --open
```

## Acceptance evidence

- The selected checkpoint's mentor feed evaluates all 106 Subject3 windows at
  84.91% accuracy, 82.21% macro-F1, and 9/9 fall recall. Its selected fall
  example has 99.4536% fall probability.
- The accepted hardware recording produces 24 strictly bimodal predictions,
  zero alerts, and maximum fall probability 0.001097. It remains unlabelled.
- A final 15-second live run opened both devices, received 290 valid IMU rows
  and 58 camera frames, and emitted four timestamped dashboard windows.
- IMU sampling skew was 0.002 ms, sequence gaps and malformed rows were zero,
  and acceleration scale remained approximately 9.81 m/s².
- No person was visible during that short live run, so pose validity was 0%.
  All four events correctly became `DEGRADED`; no ungrounded prediction or
  alert was emitted.
- A follow-up 15-second live resilience run on COM16/COM14 deliberately used a
  stale COM17 request. It auto-selected COM16, discarded one real CRC-damaged
  camera frame, generated three windows, and exited normally with no write or
  serial error.
- The dashboard page and live JSON both returned HTTP 200, all 58 unit tests
  passed, all pipeline regression scripts passed, and dashboard JavaScript
  syntax validation passed.

This completes Step 9 implementation and verifies both the successful
bimodal-inference path through recorded hardware and the live degraded-health
path through connected hardware. It does not provide clinical validation.

## Mobile notification extension

Mobile delivery remains outside the Step 9 dashboard process but is now
implemented as the separate `scripts/send_mobile_notifications.py` consumer.
It reacts only to fresh, validated, debounced fall transitions and sends a
minimal urgent message containing confidence and UTC time. See
`docs/MOBILE_FALL_NOTIFICATIONS.md`.
