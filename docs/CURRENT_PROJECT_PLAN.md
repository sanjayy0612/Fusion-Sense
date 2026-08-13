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
python scripts/check_cmhad.py --raw-root data/raw/cmhad
python scripts/prepare_cmhad.py --raw-root data/raw/cmhad
python scripts/check_cmhad.py
python scripts/train_cmhad.py --stage all --encoder-epochs 20 --fusion-epochs 20
```

The stages save `enc_imu.pt`, `enc_vis.pt`, and `fusionsense_cmhad.pt` under
the ignored `checkpoints/cmhad/` directory. Normalization statistics and the label
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
