# Pretrained camera–IMU models for FusionSense

Research date: 2026-08-13. Only checkpoints whose files were verifiably published by the model/repository owner are counted. Code-only repositories are not counted as pretrained models.

## Conclusion

There is **no verified plug-and-play checkpoint** for FusionSense's exact contract:

- third-person RGB camera processed as MediaPipe Pose `20 x 99`;
- one waist-mounted MPU6050 window `100 x 6` (accelerometer + gyroscope);
- two `128`-dimensional tokens;
- a cross-modal fusion classifier for `walking`, `standing`, `sitting`, `lying`, and `falling`.

The closest real pretrained multimodal checkpoint is **EVI-MAE / IMU-Video-MAE**, but it accepts raw egocentric video and acceleration from four limb-mounted IMUs, not MediaPipe landmarks and one MPU6050. Its weights can initialize a substantially different model; they cannot be loaded into the current `FusionSense.enc_vis`, `enc_imu`, or 128-dimensional fusion Transformer.

For the deadline, the least risky option is to keep the already-trained FusionSense IMU encoder and use a published **camera-only fall model as an independent branch**, combining the two branches by probability/decision-level fusion. This is a legitimate pretrained-model demonstration, but it must not be described as a pretrained cross-modal Transformer.

## Verified candidates

### 1. EVI-MAE / IMU-Video-MAE — closest research match, not drop-in

- **Weights:** The authors publish a downloadable `checkpoints.zip` and separate adapted VideoMAE initialization through the official repository's checkpoint section: [official repository and checkpoint links](https://github.com/mf-zhang/IMU-Video-MAE#checkpoints), [checkpoint archive](https://drive.google.com/file/d/1N0U-PR8ydHx-BtWz_v1QUrCNkZGWQ1KV/view?usp=share_link).
- **Architecture:** video ViT with joint space-time attention, IMU spectrogram encoder, modality-unified Transformer encoder, and a graph encoder across IMU devices. Fine-tuning pools unified and graph features and applies a linear action classifier. [Paper, Sections 3.2–3.5](https://arxiv.org/pdf/2407.06628).
- **Inputs:** synchronized raw egocentric RGB video plus **3-axis acceleration** from multiple body-worn IMUs. The published experiments use four limb IMUs; CMU-MMAC is approximately 60 Hz. Gyroscope and orientation are explicitly discarded. [Paper, pp. 4 and 8](https://arxiv.org/pdf/2407.06628).
- **Data/labels:** pretrained and fine-tuned on CMU-MMAC (32 cooking/action categories in the paper's processing) and WEAR (sports/activity recognition), not fall detection. [Official repository](https://github.com/mf-zhang/IMU-Video-MAE#data-preparation), [paper](https://arxiv.org/pdf/2407.06628).
- **License:** BSD-2-Clause for the released repository code. Dataset/checkpoint reuse must also respect the source datasets' terms. [Repository license](https://github.com/mf-zhang/IMU-Video-MAE/blob/main/LICENSE).
- **FusionSense compatibility:** low. It uses raw video rather than MediaPipe `33 x xyz`, four accelerometers rather than one six-axis MPU6050, and its learned dimensionality/state-dict keys do not match FusionSense. Adapting it would mean replacing the existing branches and preprocessing, then fine-tuning on compatible labeled data.

### 2. ImageBind — genuine image/IMU shared embeddings, too mismatched/heavy

- **Weights:** Meta's official implementation automatically downloads `imagebind_huge.pth` from Meta's public file host when instantiated with `pretrained=True`. [Official repository](https://github.com/facebookresearch/ImageBind), [checkpoint-loading source](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L506-L529).
- **Architecture/output:** separate modality Transformers project vision and IMU into a shared **1024-dimensional normalized embedding** for the released Huge model. It is a representation model rather than a fall/activity classifier. [Official model source](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L54-L120), [official repository](https://github.com/facebookresearch/ImageBind).
- **Inputs:** raw RGB image/video tensors and an IMU tensor configured as `6 x 2000`; the official vision preprocessor uses 224-pixel crops. [Official model source](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L254-L355), [official data preprocessing](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py#L123-L162).
- **Data/labels:** multimodal pretraining; the official table reports Ego4D IMU zero-shot evaluation, but it does not supply FusionSense's five-class head. [Official repository model table](https://github.com/facebookresearch/ImageBind#imagebind-model).
- **License:** code and model weights are CC BY-NC 4.0. [Official repository license statement](https://github.com/facebookresearch/ImageBind#license).
- **FusionSense compatibility:** low-to-medium as a research alternative, not as a drop-in. It would replace MediaPipe with a raw-video encoder, requires resampling/padding the MPU6050 stream from `100 x 6` to its expected representation, outputs 1024 rather than 128 dimensions, and still needs a trained classifier. The ~4.5 GB checkpoint is also awkward for a laptop RTX 4060 and unnecessary for the current compact architecture.

### 3. SAM-MM-HAR — real files, but not practically reusable

- **Weights:** the Hugging Face repository contains actual `best.pt` and `latest.pt` files, each about 202 MB. [Verified file listing](https://huggingface.co/AMFORGE/sam-mm-har-checkpoints/tree/main).
- **Claimed architecture/input/output:** a roughly 18M-parameter sparse multimodal Transformer over depth, IR, thermal, COCO-17 skeleton, radar, and IMU from **five body sensors x nine features**, producing one of 40 CUHK-X activity IDs (including a claimed `Fall_down` class). RGB is explicitly not used. [Model card](https://huggingface.co/AMFORGE/sam-mm-har-checkpoints/blob/main/README.md).
- **License:** the Hub metadata says Apache-2.0, while the same card says the architecture is proprietary and intentionally undisclosed. [Model card](https://huggingface.co/AMFORGE/sam-mm-har-checkpoints/blob/main/README.md).
- **FusionSense compatibility:** effectively none. The architecture/inference source needed to reconstruct the state dict is not present in the checkpoint repository, the card's usage example refers to a different repository ID, and its modalities and sensor layout differ. Do not base the submission on this checkpoint unless its author publishes working inference code and an exact schema.

### 4. YOLOv11 LE2I fall detector — usable camera-only fallback

- **Weights:** the research repository commits an actual 15.3 MB `Model/weights/best.pt`. [Checkpoint file](https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO/blob/main/Model/weights/best.pt).
- **Architecture/input/output:** YOLO11n object detector trained at 640-pixel image size on a subset of LE2I; it detects/localizes fall examples from RGB frames. The repository describes temporal filtering outside the detector. [Training arguments](https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO/blob/main/Model/args.yaml), [official research repository](https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO).
- **Data/labels:** LE2I fall-detection subset; this is a camera fall detector, not a five-activity temporal pose classifier. [Repository dataset/model description](https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO#datasets).
- **License:** MIT for the repository; Ultralytics and dataset terms must also be checked for the intended distribution/deployment. [Repository license](https://github.com/SyedBurhanAhmed/Real-Time-Fall-Detection-using-YOLO/blob/main/LICENSE).
- **FusionSense compatibility:** useful only as an independent camera branch. It does not emit a 128-dimensional MediaPipe motion token and cannot be loaded into the current camera CNN+GRU. Its fall confidence can be combined with the existing IMU classifier by late fusion.

## Excluded after verification

- **IMU2CLIP:** the official work is highly relevant (joint IMU/video/text embeddings), but published benchmark documentation explicitly notes that the authors did not release pretrained weights. The public code therefore does not meet the checkpoint requirement. [Official IMU2CLIP repository](https://github.com/facebookresearch/imu2clip), [UniMTS evaluation paper noting weights were not released](https://openreview.net/pdf?id=DpByqSbdhI).
- **LiteRehab Fusion:** its README describes MediaPipe + MPU6050 fusion and named local `.pt` files, but also states `python/models/` is gitignored; the public repository does not expose those claimed checkpoints. It is code/documentation, not a downloadable pretrained model for this review. [Repository README](https://github.com/ydh0411/lite-rehab-mvp#models-and-data).
- **IMU-Video-OOD-HAR and COMODO:** both are relevant official IMU/video cross-modal research implementations, but neither official repository publishes a trained cross-modal checkpoint; their instructions require external/base-model or user-trained checkpoint paths. [IMU-Video-OOD-HAR](https://github.com/scheshmi/IMU-Video-OOD-HAR), [COMODO](https://github.com/cruiseresearchgroup/COMODO).
- **Exact MediaPipe + waist-IMU fall paper:** the published system is conceptually close (binary fall/normal), but it does not release executable code or weights and therefore cannot supply FusionSense with a pretrained model. [Sensors paper](https://www.mdpi.com/1424-8220/25/19/6035).

## Recommendation

1. Do **not** promise that a pretrained camera+MPU6050 five-class fusion Transformer was found; it was not.
2. For the fastest real demo, use the existing pretrained IMU branch and the verified YOLOv11 fall checkpoint as two independent predictors. Fuse their fall probabilities with a fixed or validation-tuned weighted average and present this accurately as **late decision fusion**.
3. Keep the MediaPipe CNN+GRU/cross-modal Transformer path as the project's proposed trainable architecture. It can be shown with synthetic smoke-test data, but a real learned checkpoint still requires compatible paired labeled data.
4. If research fidelity matters more than the deadline, start a separate EVI-MAE adaptation branch; do not attempt to force its checkpoint into FusionSense's 128-dimensional modules.
