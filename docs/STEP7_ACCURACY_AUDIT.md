# Step 7 accuracy and safety audit

Date: 2026-08-19

## Conclusion

The current four-subject result cannot be made equivalent to a clinical fall
detector by changing only the network. A verified training-data mutation bug
was fixed and the fusion models were retrained. The best pilot fusion accuracy
increased from 83.96% to 86.79%, and its held-out fall precision/recall changed
from 90%/100% to 100%/100%. Those fall values cover only nine falls from one
held-out subject and are therefore preliminary, not evidence of flawless or
clinical performance.

## What the Nakabayashi--Saito result establishes

The cited paper is T. Nakabayashi and H. Saito, *Multimodal Human Activity
Recognition on Edge Devices*, ISMAR-Adjunct 2024,
DOI `10.1109/ISMAR-Adjunct64951.2024.00037`.

The authors report approximately 97% **precision** for five activities. Their
method uses three input streams--RGB images, acceleration, and angular
velocity--and dynamically weights their features with attention. It was tested
on a subset of UESTC-MMEA-CL. The official UESTC-MMEA-CL release contains 6,522
samples, 32 activities, 10 participants, and about 30.4 hours of synchronized
egocentric video/accelerometer/gyroscope data. Its published split divides the
samples of each class 7:2:1 into train/validation/test sets.

That 97% is not directly comparable with FusionSense's current result:

| Item | Nakabayashi--Saito | FusionSense pilot |
|---|---|---|
| Reported primary number | Approx. 97% precision | Accuracy and macro-F1 |
| Evaluated classes | Five activities | Seven posture transitions |
| Available source data | Approx. 6,522 samples, 10 people | 363 windows, 4 people |
| Camera representation | RGB images | 33-landmark MediaPipe pose |
| Inertial streams | Acceleration and gyro branches | One combined six-axis branch |
| Current split | UESTC sample split | Entire Subject3 held out |

The paper supports attention-based multimodal fusion, separate acceleration and
gyroscope processing, and more training data. It does not establish that 97%
will transfer to unseen elderly subjects, staged falls, an ESP32-CAM viewpoint,
or a different waist IMU.

## Verified experiments

All comparisons used the same 257-window training set and 106-window held-out
Subject3 set.

| Variant | Accuracy | Macro-F1 | Fall precision | Fall recall |
|---|---:|---:|---:|---:|
| Original masked fusion | 83.96% | 80.91% | 90% | 100% |
| Dropout mutation fixed, frozen encoders | 84.91% | **82.21%** | 100% | 100% |
| Fixed + end-to-end fine-tuning + 15% dropout | **86.79%** | 82.01% | 100% | 100% |
| Vision-only baseline | **87.74%** | **86.49%** | 100% | 100% |
| Engineered 14-channel IMU | 62.26% | 43.95% | 0% | 0% |
| Pose plus temporal velocity | 85.85% | 86.50% | 100% | 100% |

The original modality-dropout implementation used `torch.from_numpy` and then
zeroed tensors in place. PyTorch shared those tensors' storage with the source
`FusionWindow` arrays, so training progressively erased its own inputs. The
dataset adapter now clones values before augmentation and has a regression
test. The corrected loss remains finite and decreases instead of rising toward
2.0 as the source data disappears.

End-to-end fine-tuning improved overall accuracy, but no fusion variant has yet
beaten the camera-only baseline on this single held-out subject. This negative
result must be retained in Step 8.

## Required path to a stronger detector

1. **Use all twelve C-MHAD subjects.** Only Subject1--Subject4 are installed.
   The final model needs the remaining eight subjects, a rebuilt cache, and
   subject-grouped evaluation. Repeated tuning against Subject3 would overfit
   the current validation subject.
2. **Train continuous detection, not only event classification.** The raw
   C-MHAD recordings contain arbitrary non-interest activity between labelled
   transitions, but the current cache discards it. Mine hard negative/background
   windows and add a detection or `other` state before transition
   classification. Otherwise every live window is forced into one of seven
   transitions, making false-alerts-per-hour unreliable.
3. **Align labels with the requested dashboard.** The current labels are
   `stand_to_sit`, `sit_to_stand`, and similar transitions. They are not steady
   `standing`, `sitting`, and `lying` labels. A state head or separately labelled
   pre/post-event windows are needed for those dashboard states.
4. **Implement the paper-aligned three-stream model.** Use separate acceleration
   and gyroscope encoders plus a camera encoder, then learned attention. Keep
   sensor-level masking so both inertial branches disappear together if the IMU
   is unhealthy. Compare RGB/frame features, pose features, and RGB+pose on the
   full subject set before selecting the camera representation.
5. **Add a fall-specific safety head.** A binary fall head and a normal-activity
   anomaly detector can complement the seven-class model. Nakabayashi and Saito's
   2023 fall paper specifically uses a lightweight acceleration autoencoder so
   rare falls do not have to be the sole source of supervision. This should be
   a second signal, not a substitute for multimodal validation.
6. **Reduce sensor/domain mismatch.** Training uses a Shimmer3 at the waist;
   deployment uses an MPU-6500-compatible device. Add training-only noise,
   gain/orientation perturbation, timing jitter, camera occlusion/brightness
   changes, and then calibrate on labelled recordings from the actual hardware.
7. **Treat uncertainty as a safety feature.** Calibrate probabilities, introduce
   an abstain/uncertain state, apply temporal confirmation/hysteresis, and report
   confidence only after calibration. Do not turn low confidence into a normal
   state silently.
8. **Evaluate elderly transfer explicitly.** C-MHAD's published demographics
   are 10 male and 2 female participants and do not establish an elderly test
   cohort. Add elderly activities-of-daily-living data and, where ethically and
   safely possible, real-world or independently collected fall evidence. Never
   ask elderly participants to stage dangerous falls.

## Step 8 gate

Step 8 may run now as a **pilot evaluation** of the corrected checkpoints. A
final safety claim must wait for all-subject training, continuous background
negatives, an elderly-domain evaluation, and a locked subject-level test set.

