# Nakabayashi–Saito paper: clues for FusionSense

## Paper identified

Takuya Nakabayashi and Hideo Saito, “Multimodal Human Activity Recognition on Edge Devices,” IEEE ISMAR-Adjunct 2024, pp. 136–140, DOI: [10.1109/ISMAR-Adjunct64951.2024.00037](https://doi.org/10.1109/ISMAR-Adjunct64951.2024.00037).

The authors' official publication page and Keio University record say that the method takes images, acceleration, and angular velocity, uses an attention mechanism to dynamically weight sensor features, classifies five actions, and reports approximately 97% precision on UESTC-MMEA-CL:

- [Author publication page](https://nakabayashitakuya.github.io/)
- [Keio University publication record](https://keio.elsevierpure.com/en/publications/multimodal-human-activity-recognition-on-edge-devices/)

The complete paper is paywalled in IEEE Computer Society in the environment used for this review. Therefore, details not exposed by the authors' page or dataset sources—including the exact selected five classes, tensor sizes, and full layer configuration—must not be asserted as verified.

## Dataset clue

The paper uses **UESTC-MMEA-CL**, rather than a newly collected private dataset. Its official project page documents:

- 30.4 hours of fully synchronized first-person RGB video, accelerometer, and gyroscope data.
- 32 activity classes, approximately 200 paired samples per class.
- 10 participants (documented in the dataset paper).
- Official 70/20/10 train/validation/test split.
- Each video and sensor CSV pair shares the same filename prefix.
- Sensor CSV columns are `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`.
- Academic use only; commercial use prohibited.

Sources:

- [Official UESTC-MMEA-CL project/download page](https://ivipclab.github.io/publication_uestc-mmea-cl/mmea-cl/)
- [Official dataset code](https://github.com/Xu-Linfeng/UESTC_MMEA_CL_main)
- [Dataset paper](https://arxiv.org/abs/2301.10931)

The official training manifest contains these relevant labels: `fall`, `standing`, `walking`, and `sit_stand`. It does **not** contain separate `sitting` and `lying` classes. The five actions selected by Nakabayashi and Saito could not be verified from accessible primary-source text, so they should not be guessed.

## Architecture interpretation

The verified architectural idea is attention-based feature fusion: modality-specific networks first convert the visual, acceleration, and gyroscope inputs into learned features; attention then assigns input-dependent importance to those features; a classifier predicts the activity from the weighted representation. This is conceptually similar to FusionSense, but it is not evidence that the paper uses MediaPipe, a CNN–GRU camera encoder, 128-dimensional tokens, or a cross-modal Transformer.

## Applicability to FusionSense

The dataset is a credible route to train a synchronized camera–IMU fusion model without collecting a new dataset. However, it is not an exact domain match:

- Its camera is first-person and integrated into smart glasses; FusionSense plans a third-person external camera.
- Its IMU is located in the glasses/head coordinate frame; FusionSense uses a body-mounted MPU6050.
- It lacks separate `sitting` and `lying` labels required by the current five-class FusionSense contract.

Consequently, a model trained on it can support a proof-of-concept and demonstrate learned multimodal fusion, but should not be claimed to generalize directly to FusionSense's third-person camera and body IMU hardware.

## Practical deadline recommendation

Use a subset of UESTC-MMEA-CL for a small proof-of-concept fusion experiment, preserving the dataset's native labels. The simplest defensible experiment is to select five available classes, train vision and IMU feature encoders plus a lightweight attention-gating layer end-to-end on paired samples, and compare camera-only, IMU-only, and fused accuracy. Do not force-map `sit_stand` to `sitting`, and do not invent a `lying` label.

The official baseline code is old (Python 3.6/PyTorch 1.7, BN-Inception, eight segments) and has no released Nakabayashi–Saito checkpoint. Reusing the synchronized data and experimental idea is more practical than reproducing their exact environment.
