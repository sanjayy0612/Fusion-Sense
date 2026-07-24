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
  Download: https://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
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

### Vision — pose from real video (MediaPipe)
- We never train on raw pixels. Real video → MediaPipe Pose → 33 landmarks × 3
  = **99-dim** per-frame vector (privacy-preserving, posture-rich).
- **Set `CFG.vision_dv = 99`** before real vision training.
- Labeled clips at `data/raw/vision_videos/<label>/*.mp4`. Any action-recognition
  video source works (e.g., UCF/Kinetics subsets, or your own recordings).
- Install: `pip install mediapipe opencv-python`

Run: `python scripts/pretrain_vision.py`  → `checkpoints/enc_vis.pt`

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
