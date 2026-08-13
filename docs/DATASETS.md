# FusionSense — Datasets & the Pretraining Pipeline

The simulator is for **plumbing only** — it validates the pipeline and the
robustness logic, not real performance. Real numbers come from the datasets
below. Training happens in **two stages**:

```
Stage 1  Prepare the two V1 branches
              enc_imu   <- SisFall / UCI-HAR
              camera    <- MediaPipe Pose, no raw-camera training

Stage 2  Train the CROSS-MODAL ATTENTION on PAIRED data
         (sensors aligned in time — required to learn cross-relationships)
              UP-Fall (camera + wearable IMU)
```

Put downloaded/unzipped data under `data/raw/<name>/`. Nothing is committed
(`.gitignore` excludes it).

---

## Stage 1 — V1 branch preparation

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

### Radar — RadHAR (future extension, not required for V1)
- mmWave point-cloud HAR, 5 activities. ~70 GB preprocessed.
  Repo/data: https://github.com/nesl/RadHAR
  Preprocess into per-sample `.npy` (frames × features) at
  `data/raw/radhar/<activity>/*.npy`. Adapt `_to_frame_features` in
  `radar_loader.py` if your layout differs.

Optional later command: `python scripts/pretrain_radar.py` →
`checkpoints/enc_radar.pt`. Do not block the camera–IMU V1 on this dataset.

### Camera — open-source pose model, not v1 camera training
- Use **MediaPipe Pose Landmarker** first. It extracts 33 landmarks × 3 values =
  **99-dim** per-frame pose features, which match the repo's default vision input.
- The live V1 source is the OV2640 on ESP32-CAM. OpenCV decodes its MJPEG stream
  on the laptop; MediaPipe and FusionSense do not run on the camera board.
- Download the official Lite bundle with `python scripts/download_pose_model.py`
  and test live capture with `python scripts/test_esp32_camera.py --host <ip>`.
- **MoveNet Lightning** is the alternative if you prefer a TensorFlow/TFLite path;
  it has fewer keypoints, so it may need an adapter.
- Do not download a huge raw-video dataset just to train a camera model in v1.
  Capture or reuse short clips only for testing pose extraction and paired fusion.
- If clips are used, place them at `data/raw/vision_videos/<label>/*.mp4`, but
  treat `scripts/pretrain_vision.py` as optional. The required V1 work is IMU
  preparation plus paired camera–IMU fusion.

---

## Stage 2 — paired dataset (for the cross-modal attention)

### Your own paired dataset (recommended first milestone)

You do not need to wait for a perfect public dataset. The repository includes a
collector for your exact hardware topology. It records the ESP32 IMU serial
stream and the ESP32-CAM video during the same labelled trial:

```bash
python scripts/collect_paired.py \
  --port /dev/cu.usbserial-XXXX --camera-host 192.168.1.42 \
  --subject s01 --activity walking --trial 01 --seconds 20
```

For the laptop webcam, replace `--camera-host 192.168.1.42` with
`--camera-source 0`. MediaPipe still runs later on the laptop in both cases.

Repeat this for all five labels (`walking`, `standing`, `sitting`, `lying`,
`falling`), multiple trials, and every available subject. Each trial is saved
under `data/raw/up_fall/<subject>/<activity>/<trial>/` with `imu.csv`,
`video.mp4`, and `metadata.json`, so it is immediately readable by
`scripts/train_fusion.py`. Start with 10–20 trials per class, then expand to
different people, camera distances, lighting, and IMU placement.

Collect at least two separate trials per class before the first smoke training
run; otherwise a class cannot appear in both training and validation without
leaking windows from the same recording. Check readiness without running
MediaPipe:

```bash
python scripts/check_paired_dataset.py
```

The training script holds out complete subjects when possible, and otherwise
keeps complete recordings together in a class-stratified split.

For a fall trial, use a safe controlled setup and have the subject remain still
for about one second at the beginning and end. Do not train on simulator data
as if it were real sensor data; the simulator is only a pipeline smoke test.

### UP-Fall
- Multimodal: **2 cameras + 5 wearable IMUs + ambient**, 17 subjects, 11
  activities + falls. Published in *Sensors* (2019). ~850 GB full.
  Site: http://sites.google.com/up.edu.mx/har-up/ · Code: https://github.com/jpnm561/HAR-UP
- UP-Fall has **no mmWave radar**, which exactly matches the V1 modality set.
  `paired_loader` zero-fills radar and marks `radar_valid=False` per window; the
  model's masking removes that token from fusion.
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
# Stage 1 — V1 IMU branch (real data in data/raw/…)
python scripts/pretrain_imu.py
# Camera uses MediaPipe Pose features; skip raw-camera training in V1.

# Stage 2 — cross-modal attention on paired data
python scripts/train_fusion.py

# No data yet? Smoke-test the whole pipeline on the simulator:
python scripts/pretrain_imu.py --sim
python scripts/train_fusion.py --sim
```

## Honesty checklist (for the report/paper)

- ✅ "Simulator validates architecture + robustness mechanism."
- ✅ "V1 fuses MediaPipe camera-pose features with wearable IMU data."
- ✅ "Radar is zero-filled and masked with `radar_valid=False` in V1."
- ❌ Never report simulator accuracy as real-world performance.
- ⚠️ Any tri-modal claim needs a later paired camera + radar + IMU capture;
  UP-Fall supports only the bi-modal V1.
