# Step 8 pilot evaluation

Status: complete as a research-pilot evaluation on 2026-08-19. It is not a
clinical or deployment validation.

## Decision

Use `checkpoints/cmhad_step7_bimodal_dropout_fixed/fusionsense_cmhad.pt` for
the Step 9 prototype, with a `stand_to_fall` alert threshold of 0.60 and a
strict both-modalities-valid gate. Do not use the fine-tuned candidate for
alerts: at the same threshold it misses 5 of the 9 labelled falls.

The selected candidate is suitable for demonstrating the complete laptop
pipeline. It is not suitable for unattended elderly-safety deployment because
the labelled fall sample is small, Subject3 has already been observed during
tuning, and the model misses every labelled fall when the camera is removed.

## Evaluation data

- Labelled validation: 106 synchronized Subject3 windows, including 9
  `stand_to_fall` windows and 97 non-fall windows.
- Continuous negative replay: 545 non-overlapping two-second windows, or
  0.30278 hours, from all ten Subject3 transition recordings.
- Negative replay excludes every window from one second before an annotated
  fall onset through the end of that recording. A post-fall subject remaining
  on the floor is not counted as negative time.
- Hardware replay: the accepted
  `data/recordings/fusion_usb_step4_20260819` USB session. It is unlabelled and
  therefore verifies execution, synchronization, and input health only.
- ESP32 acceleration in `g` is converted with `m/s² = g × 9.80665` before the
  model contract is validated, matching the prepared C-MHAD acceleration unit.

Subject3 has been used during iterative development, so these are validation
figures rather than an untouched final test.

## Labelled and continuous results

All fall-alert figures use probability threshold 0.60.

| Model | Accuracy | Macro-F1 | Fall precision | Fall recall | Continuous false-alert episodes |
|---|---:|---:|---:|---:|---:|
| IMU-only baseline | 61.32% | 43.56% | undefined (no alerts) | 0/9, 0% | 0 |
| Vision-only baseline | 87.74% | 86.49% | 9/9, 100% | 9/9, 100% | 0 |
| Corrected frozen fusion | 84.91% | 82.21% | 9/9, 100% | 9/9, 100% | 0 |
| End-to-end fine-tuned fusion | 86.79% | 82.01% | 4/4, 100% | 4/9, 44.44% | 0 |

The selected frozen fusion has 0/97 labelled false positives and 0 observed
false-alert episodes in 18.17 minutes of continuous negative replay. That
observation is encouraging but not statistically strong: with zero observed
events, the one-sided 95% Poisson upper bound is still 9.89 alerts/hour. At
least 29.96 zero-event negative hours would be required merely to place that
upper bound below 0.1 alerts/hour.

For the selected model, the Wilson 95% interval for both fall recall and fall
precision is approximately 70.09% to 100%, reflecting that there are only nine
falls. Vision-only remains more accurate overall, so the present experiment
does not demonstrate an accuracy gain from fusion.

## Modality dropout

| Selected-model input | Accuracy | Macro-F1 | Fall alerts at 0.60 |
|---|---:|---:|---:|
| IMU + camera | 84.91% | 82.21% | 9/9 |
| IMU only | 29.25% | 25.03% | 0/9 |
| Camera only | 86.79% | 85.99% | 9/9 |

This is a failed graceful-degradation result. The fusion path technically
supports masks, but its fall decision is camera-dependent. Step 9 must report
camera loss as `DEGRADED` and withhold a fused activity/alert rather than claim
that IMU-only operation is safe.

## Synchronization

Measured on the accepted USB hardware session:

- post-calibration alignment lag: 0 ms;
- shared-motion correlation: 0.59719;
- clock-fit residual p95: 7.61 ms IMU and 14.89 ms camera;
- resampling skew maximum: 2.751 ms IMU and 53.993 ms camera;
- transport latency p95: 14.88 ms IMU and 315.93 ms camera.

The synthetic sensitivity test shifts the IMU within each labelled two-second
window from -500 to +500 ms. The selected model keeps 9/9 fall alerts with no
labelled false positives throughout that range. Worst overall accuracy is
83.96% and worst macro-F1 is 80.92%. This is a perturbation test, not a
replacement for measured clock synchronization.

## Latency and hardware input health

For the selected fusion checkpoint, batch-one model latency is:

| Runtime | Median | p95 |
|---|---:|---:|
| CUDA | 3.31 ms | 4.22 ms |
| CPU | 6.59 ms | 8.08 ms |

JPEG decode plus pose extraction has 14.35 ms median and 15.67 ms p95 over 100
frames sampled across the accepted 601-frame recording. Only 38% of those
sampled frames contain a valid full-body pose; similarly, only 24 of the 58
assembled hardware windows passed the camera-pose gate. The camera transport
delay, two-second observation window, and one-second live stride dominate the
small neural-network compute time.

The selected checkpoint produces zero alerts on the 24 eligible unlabelled
hardware windows, with maximum fall probability 0.00110. This does not measure
fall recall because no hardware ground-truth labels exist.

## Acceptance and remaining safety work

Step 8 is complete for a mentor/research demonstration. Final safety acceptance
is blocked until all of the following are available:

1. An untouched subject-grouped test set, preferably all 12 C-MHAD subjects.
2. At least about 30 hours of representative negative monitoring if the target
   is a 95% upper bound below 0.1 false alerts/hour with zero observed events.
3. Safely staged and labelled local falls and difficult non-falls using the
   actual ESP32-CAM/IMU placement.
4. IMU-branch improvement until camera dropout retains acceptable fall recall.
5. Hard-negative training for lying down, rapid sitting, occlusion, pose loss,
   and postural transitions, followed by calibration on a separate set.

## Reproduction

Build the corrected continuous cache:

```powershell
.\.runtime\python311\python.exe -u .\scripts\build_step8_background.py `
  --subject 3 --stride-seconds 2 --fall-guard-seconds 1 `
  --output .\data\processed\cmhad_step8_subject3_background_v2.npz
```

Run the full evaluation:

```powershell
.\.runtime\python311\python.exe -u .\scripts\evaluate_step8.py `
  --checkpoint .\checkpoints\cmhad_step7_bimodal_dropout_fixed `
  --checkpoint .\checkpoints\cmhad_step7_bimodal_finetuned `
  --background-cache .\data\processed\cmhad_step8_subject3_background_v2.npz `
  --background-metadata .\data\processed\cmhad_step8_subject3_background_v2.json `
  --hardware-session .\data\recordings\fusion_usb_step4_20260819 `
  --output .\checkpoints\step8_pilot_evaluation_v3.json
```

The machine-readable report is
`checkpoints/step8_pilot_evaluation_v3.json`.
