# Copy-paste prompt for the RTX 4060 training laptop

Paste everything inside the block below into Codex after cloning this repository.

```text
You are working in the cloned FusionSense repository. Your goal is to prepare
the official C-MHAD Transition Movements dataset and train the real camera+IMU
FusionSense model on this RTX 4060 laptop.

Work interactively and persist until training either completes or reaches a
genuine external blocker. Do not use simulator data for results. Do not commit
raw data, processed data, logs, or checkpoints.

Before downloading or installing anything, inspect README.md,
docs/DATASETS.md, requirements.txt, scripts/prepare_cmhad.py,
scripts/check_cmhad.py, and scripts/train_cmhad.py. Then ask me these questions
one at a time and wait for each answer:

1. Where should the large raw dataset live? Recommend a path outside the Git
   repository if the repository drive is small. The transition subset is about
   47 GB, and extraction/processing needs additional free space.
2. Do I want a four-subject pilot download (approximately 16 GB, recommended
   for a deadline smoke run) or all twelve subjects (approximately 47 GB,
   recommended for final reported results)? Explain that every selected subject
   contains all seven transition classes.
3. Ask permission to install Python dependencies and the correct official
   CUDA-enabled PyTorch build for the detected NVIDIA driver.
4. Immediately before preprocessing, ask permission because MediaPipe processing
   can take a long time.
5. Show the exact training commands and estimated stages, then ask permission
   immediately before starting GPU training.

After approval:

- Inspect nvidia-smi, Python version, available disk space, and the current Git
  status. Preserve all existing files and changes.
- Create a project-local virtual environment named .venv if one does not exist.
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
- For a pilot, download complete Subject1 through Subject4 folders. For the full
  run, download Subject1 through Subject12. Preserve each Subject folder with:
    ActionOfInterestTraSubjectN.xlsx
    InertialData/inertial_subN_tr*.csv
    VideoData/video_subN_tr*.avi
- Make the selected extraction visible at data/raw/cmhad. A symlink is allowed
  if the files live outside the repository.
- Run python scripts/check_cmhad.py --raw-root data/raw/cmhad and stop to explain
  any missing subjects, annotations, videos, or IMU files.
- Run python scripts/prepare_cmhad.py --raw-root data/raw/cmhad. Do not use
  --max-subjects for the final selected run. This creates the ignored compact
  cache data/processed/cmhad_windows.npz.
- Run python scripts/check_cmhad.py again. Report the per-class counts and the
  number of MediaPipe-invalid windows. If more than 20% are invalid, inspect
  sample frames and diagnose before training.
- Run the repository tests:
    python tests/test_pipeline.py
    python tests/test_camera_stream.py
- Start all real training stages with logging:
    python scripts/train_cmhad.py --stage all --encoder-epochs 20 \
      --fusion-epochs 20 --batch-size 64
  If CUDA runs out of memory, retry only by lowering --batch-size to 32 and then
  16. Do not change labels, windows, subject split, or architecture without my
  approval.
- The script must train in this order:
    1) C-MHAD waist-IMU encoder FROM RANDOM INITIALIZATION (do not load
       SisFall or any existing checkpoint) -> checkpoints/cmhad/enc_imu.pt
    2) MediaPipe camera-pose encoder FROM RANDOM INITIALIZATION ->
       checkpoints/cmhad/enc_vis.pt
    3) cross-modal Transformer -> checkpoints/cmhad/fusionsense_cmhad.pt
- Confirm that validation holds out entire subjects and report the held-out
  subject names. Never use a random window-level split.
- At completion, summarize final validation accuracy, macro-F1,
  stand-to-fall recall, modality-dropout results, elapsed time, and every
  generated checkpoint. Clearly state whether this was the four-subject pilot
  or full twelve-subject experiment.
```
