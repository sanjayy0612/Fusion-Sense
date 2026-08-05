# FusionSense — Datasets & the Pretraining Pipeline

The simulator is for **plumbing only** — it validates the pipeline and the
robustness logic, not real performance. Real numbers come from the datasets
below. Training happens in **two stages**:

```
Stage 1  Pretrain each encoder SEPARATELY on real single-modality data
         (unpaired is fine — we only want good per-modality features)
              enc_imu   <- SisFall / UCI-HAR
              enc_radar <- RadHAR
              enc_vis   <- pose from real video (MediaPipe)

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data
         (sensors aligned in time — required to learn cross-relationships)
              UP-Fall (camera + wearable IMU)
```

Put downloaded/unzipped data under `data/raw/<name>/`. Nothing is committed
(`.gitignore` excludes it).

---

## Stage 1 — single-modality datasets (for the encoders)

### IMU — SisFall (recommended) or UCI-HAR
- **SisFall** — waist-worn IMU, 19 ADLs + 15 fall types, 38 subjects. Waist
  placement matches our physical design; includes real falls.
  Size: ~200-230 MB zipped, ~500-600 MB unzipped (plain-text CSV, short trials).
  Download (pick whichever mirror responds):
    - Official: https://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
    - GitHub mirror: https://github.com/BIng2325/SisFall/releases
    - Hugging Face mirror: https://huggingface.co/datasets/Trupal7/Sisfall_Dataset/tree/main
  Layout: `data/raw/sisfall/**/*.txt` (filename prefix `D##`=ADL, `F##`=fall).
- **UCI-HAR** — smartphone accel+gyro, 6 activities, 30 subjects, 50 Hz.
  Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
  Set `CFG.imu_dir = "uci_har"`.

Run: `python scripts/pretrain_imu.py`  → `checkpoints/enc_imu.pt`

### Radar — RadHAR
- mmWave point-cloud HAR, 5 activities. ~70 GB preprocessed.
  Repo/data: https://github.com/nesl/RadHAR
  Preprocess into per-sample `.npy` (frames × features) at
  `data/raw/radhar/<activity>/*.npy`. Adapt `_to_frame_features` in
  `radar_loader.py` if your layout differs.

Run: `python scripts/pretrain_radar.py`  → `checkpoints/enc_radar.pt`

### Vision — pose from small video clips (MediaPipe, storage-light path)
- We never train on raw pixels. Real video → MediaPipe Pose → 33 landmarks × 3
  = **99-dim** per-frame vector (privacy-preserving, posture-rich).
- **Set `CFG.vision_dv = 99`** before real vision training.
- Do **not** download a huge full video dataset first. Start with a small, curated
  subset or your own laptop/webcam clips, convert each video to pose windows, and
  cache only the extracted keypoints/features. After extraction, raw `.mp4` files
  can be moved to external storage or deleted if your review only needs the
  feature dataset.
- Labeled clips at `data/raw/vision_videos/<label>/*.mp4`. Keep the labels aligned
  to the 5-class FusionSense mapping: `walking`, `sitting`, `standing`, `lying`,
  `fall`. If a public dataset uses more labels, copy only the classes you need.
- Install: `pip install mediapipe opencv-python`

Run: `python scripts/pretrain_vision.py`  → `checkpoints/enc_vis.pt`

#### If video storage is the blocker

Use this order instead of downloading hundreds of GB:

1. Record **short 5–10 second clips** on your laptop/webcam or phone for each
   safe class. For falls, use safe staged events: mattress/cushion, dummy object,
   or controlled sit-to-lying transitions; do not perform unsafe falls.
2. Extract MediaPipe pose features immediately. The training input becomes small
   numeric arrays instead of raw video. A 10-second clip at 10 FPS with 99 float32
   pose values is roughly 40 KB before metadata, so the feature cache is tiny
   compared with the original videos.
3. Train the vision encoder on pose windows, not raw pixels. This is enough for
   posture/fall cues and matches the deployed laptop API, which also extracts
   pose before inference.
4. For the report, be explicit: “vision encoder trained on pose/keypoint
   features extracted from a curated small video set,” not “trained on a massive
   raw-video dataset.”

---

## Stage 2 — paired dataset (for the cross-modal attention)

### UP-Fall
- Multimodal: **2 cameras + 5 wearable IMUs + ambient**, 17 subjects, 11
  activities + falls. Published in *Sensors* (2019). ~850 GB full.
  Site: http://sites.google.com/up.edu.mx/har-up/ · Code: https://github.com/jpnm561/HAR-UP
- UP-Fall has **no mmWave radar**, so `paired_loader` marks `radar_valid=False`
  per window — the model's masking handles the missing modality natively. You
  add real radar later from your own capture (or lean on the RadHAR-pretrained
  encoder).
- Normalize into:
  `data/raw/up_fall/<subject>/<activity>/<trial>/imu.csv`  (cols `ax,ay,az,gx,gy,gz`)
  `data/raw/up_fall/<subject>/<activity>/<trial>/video.mp4`
- Map activity folder names to our 5 classes in `paired_loader.ACTIVITY_MAP`.

Run: `python scripts/train_fusion.py`  → loads pretrained encoders, freezes
them, trains the attention, prints the robustness table, saves
`checkpoints/fusionsense.pt`.

---

## Full command sequence

```bash
# Stage 1 — encoders (real data in data/raw/…)
python scripts/pretrain_imu.py
python scripts/pretrain_radar.py
python scripts/pretrain_vision.py          # after setting CFG.vision_dv = 99

# Stage 2 — cross-modal attention on paired data
python scripts/train_fusion.py

# No data yet? Smoke-test the whole pipeline on the simulator:
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim
```

## Honesty checklist (for the report/paper)

- ✅ "Simulator validates architecture + robustness mechanism."
- ✅ "Encoders pretrained on real single-modality benchmarks."
- ✅ "Cross-modal attention trained on paired UP-Fall data."
- ❌ Never report simulator accuracy as real-world performance.
- ⚠️ The tri-modal (camera+radar+IMU together) claim needs your own paired
  capture — UP-Fall covers only camera+IMU.
