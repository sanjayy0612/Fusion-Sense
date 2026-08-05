# FusionSense — Arduino Gateway, Laptop Inference API, and Training Plan

This plan replaces the earlier Raspberry Pi deployment target with a lower-cost
setup: an Arduino-compatible microcontroller acquires sensor data, while your
laptop runs the model and exposes a local API. The model architecture and data
contract stay the same, so the project can still move to a Raspberry Pi or other
edge computer later without retraining from scratch.

## 1. Updated deployment architecture

```
MPU-6050 IMU ──I2C──┐
                    ├── Arduino/ESP32 gateway ──USB serial/WiFi── Laptop API
LD2410 radar ─UART──┘                                      │
                                                           ▼
Camera / webcam ───────────────────────────────► vision feature extractor
                                                           │
                                                           ▼
                                         time-sync + FusionWindow builder
                                                           │
                                                           ▼
                                         PyTorch FusionSense inference
                                                           │
                                                           ▼
                                      JSON result / dashboard / alert
```

### Responsibilities

| Component | Responsibility | Reason |
|-----------|----------------|--------|
| Arduino-compatible gateway | Read MPU-6050 and LD2410, timestamp samples, stream CSV or JSON frames. | Cheap, deterministic sensor acquisition. |
| Laptop API | Parse serial/WiFi frames, extract camera pose features, build `FusionWindow`s, run PyTorch model, return activity + trust weights. | Uses available RTX 4060/CUDA for development and fast inference. |
| Dashboard/client | Calls the laptop API or subscribes to output events. | Keeps visualization separate from model code. |

Use an **ESP32 programmed through the Arduino IDE** if possible. It is still an
Arduino-style workflow, but it gives WiFi, more RAM, and a second UART for the
LD2410. An Arduino Uno can stream an MPU-6050, but it is tight for simultaneous
radar parsing and network communication.

## 2. What changes from the Raspberry Pi plan

| Earlier plan | New cost-constrained plan |
|--------------|---------------------------|
| Pi owns camera, windowing, inference, MQTT. | Laptop owns camera, windowing, inference, and API. |
| Pi runs quantized model. | Laptop runs the full PyTorch checkpoint during development. |
| ESP32 is only a sensor gateway. | Arduino/ESP32 remains only a sensor gateway. |
| Deployment metric: Pi CPU latency. | Development metric: laptop API latency; later optional edge-port metric. |

The important design decision is that the microcontroller **does not run the
model**. It only produces timestamped sensor samples that the laptop converts
into the same `FusionWindow` contract used by training and simulation.

## 3. Laptop API shape

A minimal first version can be a local HTTP service:

- `POST /predict/window`: accepts one complete `FusionWindow` serialized as JSON
  or NumPy arrays saved by the capture process.
- `GET /health`: reports whether the model checkpoint, serial port, and camera
  are available.
- Response:

```json
{
  "activity": "fall",
  "class_id": 4,
  "confidence": 0.91,
  "trust": {"imu": 0.23, "radar": 0.62, "vision": 0.15},
  "valid": {"imu": true, "radar": true, "vision": true},
  "t_start": 1785900000.25
}
```

For the first implementation, keep serial capture and API inference in separate
modules. That makes it easy to test the model with recorded windows before live
hardware is stable.

## 4. Model training architecture

FusionSense should use the existing two-stage training plan, because no public
dataset perfectly matches camera + mmWave radar + wearable IMU together.

### Stage 1 — pretrain modality encoders

Train each encoder on the best available real single-modality dataset:

1. **IMU encoder** on SisFall or UCI-HAR.
2. **Radar encoder** on RadHAR.
3. **Vision encoder** on pose features extracted from labeled videos.

This stage teaches each sensor-specific encoder useful motion representations.
Unpaired data is acceptable here because each encoder is learning within one
modality only.

### Stage 2 — train fusion and attention on paired data

Train the cross-modal attention on paired windows, starting with UP-Fall for
camera + IMU. Because UP-Fall has no mmWave radar, mark `radar_valid=False` for
those windows and rely on:

- the RadHAR-pretrained radar encoder,
- modality-dropout augmentation,
- sensor-validity masks,
- and later a small self-collected tri-modal calibration dataset.

Do **not** claim final tri-modal accuracy from simulator data. Use simulator
runs only as plumbing checks for tensor shapes, masks, and robustness logic.

## 5. Recommended dataset approach

The dataset feels weak because the exact tri-modal combination is missing. Treat
this as an engineering risk, not a dead end:

1. **Use public datasets for representation learning.** Pretrain each encoder on
   the closest real data available.
2. **Use paired camera + IMU data for the first fusion result.** This validates
   the attention and masking behavior with real synchronized sensors.
3. **Make vision storage-light.** Do not keep a huge raw-video corpus on the
   laptop. Record or download short clips, extract MediaPipe pose/keypoint arrays,
   and train on the cached numeric features. This keeps the vision branch close
   to the actual deployment path while avoiding hundreds of GB of video storage.
4. **Collect a small local tri-modal dataset.** Record 5–10 subjects or repeated
   trials for safe activities, plus controlled fall-like events using a cushion
   or dummy object. Even a small dataset can be used for calibration, qualitative
   demos, and robustness tests.
5. **Run ablations honestly.** Report simulator results as pipeline validation,
   public-dataset results as encoder/fusion learning, and local tri-modal data as
   the hardware demonstration.

## 6. RTX 4060 training guidance

An RTX 4060 is enough for this architecture because the model is intentionally
small:

- modality tokens are only `d_model=128`,
- attention runs over three tokens, not full video frames,
- vision uses pose/keypoint embeddings instead of raw pixels,
- batch sizes can be reduced if VRAM is limited.

Start with the simulator smoke tests, then train one encoder at a time. After
that, freeze pretrained encoders and train only the fusion attention/head. If the
real datasets are noisy or label mappings are inconsistent, prefer smaller clean
subsets over a large but poorly normalized dataset.

## 7. Immediate build order

1. Keep the current ESP32/Arduino firmware as the sensor gateway.
2. Add a laptop serial reader that writes timestamped IMU/radar rows to disk.
3. Add a window builder that converts recorded rows into `FusionWindow`s.
4. Run `scripts/train_fusion.py --sim` to verify the training loop.
5. Download/preprocess public datasets and run Stage 1 pretraining.
6. Train Stage 2 fusion on UP-Fall.
7. Expose laptop inference through a small local API.
8. Collect a small local tri-modal recording set and use it for validation.
