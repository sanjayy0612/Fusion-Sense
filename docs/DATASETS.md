# FusionSense dataset: C-MHAD

FusionSense V1 uses the official **C-MHAD Transition Movements Application**:

- Official page: https://personal.utdallas.edu/~kehtar/C-MHAD.html
- Official Box share: https://utdallas.box.com/s/4zwus5h4khsxpullnm45o59xlfpba6a0
- Official reader: https://github.com/HaoranWeiUTD/C-MHAD-Python

Use only `TransitionMovementsApplication`; `TVGestureApplication` is unrelated.
The transition subset is approximately 47 GB: 12 subjects × ten two-minute
recordings, plus waist Shimmer3 acceleration/gyroscope streams and timestamp
annotations. The camera is 640×480 at 15 FPS; IMU is 50 Hz.

Expected extracted layout:

```text
data/raw/cmhad/
  Subject1/
    ActionOfInterestTraSubject1.xlsx
    InertialData/inertial_sub1_tr1.csv ... tr10.csv
    VideoData/video_sub1_tr1.avi ... tr10.avi
  ...
  Subject12/
```

An extra `TransitionMovementsApplication/` directory is accepted. Raw and
processed data are Git-ignored. A symlink at `data/raw/cmhad` may point to a
larger drive.

## Preparation

```bash
python scripts/download_pose_model.py
python scripts/check_cmhad.py --raw-root data/raw/cmhad --expected-subjects 4
python scripts/prepare_cmhad.py --raw-root data/raw/cmhad
python scripts/check_cmhad.py --expected-subjects 4 --require-cache
```

Preparation reads each annotation, centres a two-second interval on it,
restores the documented missing leading IMU samples, extracts the matching
100×6 IMU data, and runs MediaPipe on 20 aligned video frames. It writes
`data/processed/cmhad_windows.npz`, normally only tens of megabytes.

## Training the four-subject pilot

```bash
python scripts/train_cmhad.py --stage all --encoder-epochs 20 --fusion-epochs 20 \
  --expected-subjects 4 --output-dir checkpoints/cmhad_pilot4
```

This uses one subject-level split for all three stages, fits normalization only
on training subjects, and trains IMU and camera-pose from random initialization
before training fusion. For a deadline
pilot, complete Subject1–Subject4 can be used; final reported results should use
all 12 subjects.

When all twelve subjects are available, rebuild the cache and retrain all three
stages from scratch with `--expected-subjects 12 --output-dir
checkpoints/cmhad_full12`. Do not train only on the eight newly added subjects.

The complete interactive GPU handoff is in
[CODEX_CMHAD_TRAINING_PROMPT.md](CODEX_CMHAD_TRAINING_PROMPT.md).
