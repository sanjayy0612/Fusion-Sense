# Copy-paste prompt for the RTX 4060 training laptop

Paste everything inside the block below into Codex after cloning this repository.

```text
You are working in the cloned FusionSense repository. Your goal is to prepare
the four-subject pilot from the official C-MHAD Transition Movements dataset
and train the real camera+IMU FusionSense model on this RTX 4060 laptop. The
scope for this run is fixed to complete Subject1 through Subject4 (about 16 GB).
Do not propose or download the other eight subjects during this run.

Work interactively and persist until training either completes or reaches a
genuine external blocker. Do not use simulator data for results. Do not commit
raw data, processed data, logs, or checkpoints.

Before downloading or installing anything, inspect README.md,
docs/DATASETS.md, requirements.txt, scripts/prepare_cmhad.py,
scripts/check_cmhad.py, and scripts/train_cmhad.py.

Before asking questions, explain the experiment to me briefly in plain English:

- C-MHAD records four selected people separately with a third-person laptop
  camera and a synchronized waist-mounted six-axis IMU.
- Every selected person performs the same seven posture transitions, including
  standing-to-sitting, lying transitions, and standing-to-falling.
- Each example becomes an aligned two-second camera-pose window and 100x6 IMU
  window. MediaPipe itself stays fixed; the IMU temporal encoder and camera
  temporal encoder are trained from random initialization to produce separate
  128-dimensional embeddings. A cross-modal Transformer then learns to fuse
  those embeddings and predict the transition.
- Entire people, not random windows, are held out for validation. With four
  subjects, three are used for training and one unseen subject for validation;
  the training script reports the exact held-out subject.
- This is a pilot model for the seven transitions, not a claim of universal
  activity recognition or continuous walking/posture recognition.

Keep that explanation below 180 words. Then ask me these questions one at a
time and wait for each answer:

1. Where should the large raw dataset live? Recommend a path outside the Git
   repository if the repository drive is small. The selected four-subject
   subset is about 16 GB, and extraction/processing needs additional free space.
2. Confirm that at least 35 GB of temporary free space is available because a
   downloaded archive and extracted files may briefly coexist. If not, help me
   place the archive and extracted dataset on separate drives.
3. Ask permission to install Python dependencies and the correct official
   CUDA-enabled PyTorch build for the detected NVIDIA driver.
4. Immediately before preprocessing, ask permission because MediaPipe processing
   can take a long time.
5. Show the exact training commands and estimated stages, then ask permission
   immediately before starting GPU training.

After approval:

- Inspect nvidia-smi, Python version, available disk space, and the current Git
  status. Preserve all existing files and changes.
- Prefer Python 3.11 for MediaPipe compatibility. Create a project-local virtual
  environment named .venv if one does not exist.
- Install a CUDA PyTorch build from the current official PyTorch instructions,
  then install requirements.txt. Verify torch.cuda.is_available(), GPU name,
  and a small CUDA tensor operation.
- Run: python scripts/download_pose_model.py
- Use only the official C-MHAD source:
  https://personal.utdallas.edu/~kehtar/C-MHAD.html
  Official Box share:
  https://utdallas.box.com/s/4zwus5h4khsxpullnm45o59xlfpba6a0
- Download only TransitionMovementsApplication, never TVGestureApplication.
  If Box requires a browser confirmation or login, open it and ask me to finish
  that step; do not silently switch to an unofficial mirror.
- Download complete Subject1 through Subject4 folders only. Preserve each
  Subject folder with:
    ActionOfInterestTraSubjectN.xlsx
    InertialData/inertial_subN_tr*.csv
    VideoData/video_subN_tr*.avi
- Make the selected extraction visible at data/raw/cmhad. A symlink is allowed
  if the files live outside the repository.
- Run:
    python scripts/check_cmhad.py --raw-root data/raw/cmhad --expected-subjects 4
  Stop to explain any validation failure involving subjects, annotations,
  videos, or IMU files.
- Run python scripts/prepare_cmhad.py --raw-root data/raw/cmhad. Do not use
  --max-subjects for the final selected run. This creates the ignored compact
  cache data/processed/cmhad_windows.npz.
- Run the same check again with --require-cache. Report per-class counts and
  the number of MediaPipe-invalid windows. The checker must exit successfully;
  if more than 20% are invalid, inspect sample frames and diagnose before training.
- Run the repository tests:
    python tests/test_pipeline.py
    python tests/test_camera_stream.py
- Start all real training stages with logging:
    python scripts/train_cmhad.py --stage all --encoder-epochs 20 \
      --fusion-epochs 20 --batch-size 64 --expected-subjects 4 \
      --output-dir checkpoints/cmhad_pilot4
  If CUDA runs out of memory, retry only by lowering --batch-size to 32 and then
  16. Do not change labels, windows, subject split, or architecture without my
  approval.
- The script must train in this order:
    1) C-MHAD waist-IMU encoder FROM RANDOM INITIALIZATION (do not load
       SisFall or any existing checkpoint) -> <output-dir>/enc_imu.pt
    2) MediaPipe camera-pose encoder FROM RANDOM INITIALIZATION ->
       <output-dir>/enc_vis.pt
    3) cross-modal Transformer -> <output-dir>/fusionsense_cmhad.pt
- Confirm that validation holds out entire subjects and report the held-out
  subject names. Never use a random window-level split.
- At completion, summarize final validation accuracy, macro-F1,
  stand-to-fall recall, modality-dropout results, elapsed time, and every
  generated checkpoint. Clearly state that this was the four-subject pilot.
```
