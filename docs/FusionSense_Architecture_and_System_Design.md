# FusionSense — Architecture & System Design

**Cross-Modal Attention-Based Fusion for Real-Time Human Activity Understanding**

Authors: Sanjay E (24CS0836), Shirdithan Pon (24CS0899)
Document type: System architecture + design decisions (software-first build)
Status: Draft v1 · Date: 2026-06-26

> This document is the engineering blueprint for FusionSense. It is written **software-first** because hardware is not yet in hand — every component is designed so you can build and validate it in simulation now, then drop in real sensors later with minimal change. It doubles as the backbone for your research paper (the novelty argument lives in Sections 4 and 9).

---

## 0. How to read this document

1. **Sections 1–2** define *what* the system must do and *how the pieces fit*. Read these first.
2. **Sections 3–4** are the core engineering: the data contract between modalities and the fusion model. This is where your paper's contribution lives.
3. **Sections 5–6** are the decisions (ADRs) and the data strategy — the *hidden hard part* of this project.
4. **Sections 7–9** are the build plan: repo layout, milestones, and the paper framing.

A glossary is at the end (Section 10) — HAR, modality, token, etc.

---

## 1. Requirements

### 1.1 Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | Acquire synchronized streams from three modalities: vision (camera), spatial (LD2410 mmWave radar), kinematics (MPU-6050 IMU). |
| F2 | Segment each stream into fixed-length, time-aligned windows (e.g., 2 s windows, 50% overlap). |
| F3 | Classify each window into one of N human-activity classes (e.g., walking, sitting, standing, lying, **fall**). |
| F4 | Dynamically weight each modality's contribution per window via a learned cross-modal attention mechanism (the core innovation). |
| F5 | Run all inference locally on the edge (Raspberry Pi 5) — no cloud, no raw video leaves the device. |
| F6 | Publish classified activity + per-modality attention weights over MQTT to a local dashboard. |
| F7 | Raise a distinct, low-latency alert on the **fall** class. |
| F8 | Degrade gracefully: if one modality drops out (camera blinded, radar unplugged), the system still classifies using the remaining modalities. |

### 1.2 Non-functional requirements

| ID | Requirement | Target |
|----|-------------|--------|
| N1 | End-to-end latency (sensor → classification → MQTT) | < 300 ms |
| N2 | Inference time on Pi 5 (CPU, single window) | < 100 ms |
| N3 | Model size (quantized) | < 10 MB |
| N4 | Sustained throughput | ≥ 5 windows/s (with overlap) |
| N5 | Privacy | Raw camera frames never persisted or transmitted; only derived features leave the vision node |
| N6 | Robustness | No single modality failure causes a system-wide failure (see F8) |
| N7 | Power (target deployment) | Runs on Pi 5 + USB peripherals, < 15 W |

### 1.3 Constraints & assumptions

- **Constraint:** Two-person student team, ~4-month timeline (per your project plan), commodity hardware (ESP32, Pi 5).
- **Constraint:** No existing public dataset combines *camera + mmWave + IMU* for the same subjects/activities. **This is the single biggest risk** — addressed in Section 6.
- **Assumption:** Activities are single-person, indoor, short-duration (seconds-scale). Multi-person tracking is explicitly out of scope for v1.
- **Assumption:** "Real-time" means soft real-time (sub-second), not hard deadlines.

---

## 2. High-level architecture

FusionSense is a **three-tier edge pipeline**: a bare-metal sensor gateway, an edge compute node, and a local IoT/visualization layer.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TIER 1 — SENSOR GATEWAY (ESP32)                     │
│                                                                            │
│   MPU-6050 ──I²C──┐                                                         │
│                   ├──► [Acquisition + timestamp + framing] ──UART/WiFi──┐  │
│   LD2410   ──UART─┘                                                      │  │
│                                                                         │  │
└─────────────────────────────────────────────────────────────────────────┼──┘
                                                                          │
                  ESP32 sends timestamped IMU + radar frames             │
                                                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    TIER 2 — EDGE COMPUTE NODE (Raspberry Pi 5)              │
│                                                                            │
│   USB/CSI Camera ──► [Vision feature extractor]  ──┐                        │
│   ESP32 stream  ──► [IMU windower]   ─────────────►├─► [Time-sync &         │
│                 ──► [Radar windower] ─────────────►│    windowing buffer]   │
│                                                    │          │            │
│                                                    │          ▼            │
│                                            ┌───────────────────────────┐    │
│                                            │  FUSION MODEL (PyTorch)   │    │
│                                            │  per-modality encoders →  │    │
│                                            │  cross-modal attention →  │    │
│                                            │  classifier head          │    │
│                                            └───────────────────────────┘    │
│                                                    │                        │
│                                          activity + attention weights       │
│                                                    │                        │
│                                            [MQTT publisher]                  │
└────────────────────────────────────────────────────┼──────────────────────┘
                                                     │
                          MQTT (local broker, e.g. Mosquitto on Pi)
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              TIER 3 — IoT / DASHBOARD (Node-RED, local)                     │
│   Live activity tile · attention-weight bars · fall alert · history log    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Why this split

- **ESP32 is a sensor gateway, not a compute node.** It does deterministic, low-jitter acquisition of I²C (IMU) and UART (radar) — the one thing it's genuinely good at — and timestamps each sample. It does *not* run the model.
- **The Pi 5 is the brain.** It owns the camera (which needs more bandwidth/compute than the ESP32 can give), runs the fusion model, and hosts the MQTT broker.
- **The dashboard is just a subscriber.** It holds no logic and can be swapped (Node-RED, a small web app, Grafana) without touching the pipeline.

### 2.2 The two data planes

There are two distinct flows, and keeping them separate is what makes the system tractable:

1. **Training plane (offline, now):** recorded/simulated multimodal windows → PyTorch training loop → trained checkpoint. Runs on your laptop/Colab. **This is what you build first.**
2. **Inference plane (online, later):** live sensors → windowing → quantized model → MQTT. Runs on the Pi.

Because hardware isn't here, **you build the entire training plane and validate the model on a simulated/recorded dataset, behind the same data contract the live system will use** (Section 3). When sensors arrive, you only swap the data *source*, not the model.

---

## 3. The data contract (the most important interface)

Everything downstream depends on one decision: **what does a single "window" of fused data look like?** Define this once and both the simulator and the real sensors must produce it identically.

### 3.1 Per-modality raw rates and window definition

| Modality | Sensor | Native rate | Sampled rate | Per-window shape (2 s window) |
|----------|--------|-------------|--------------|-------------------------------|
| Kinematics | MPU-6050 (6-DoF) | up to 1 kHz | 50 Hz | `(100, 6)` — 100 samples × [ax, ay, az, gx, gy, gz] |
| Spatial | LD2410 mmWave | ~10–20 Hz frames | 20 Hz | `(40, K)` — 40 frames × K gate/energy features |
| Vision | Camera | 30 fps | 10 fps | `(20, D_v)` — 20 frames × D_v embedding dims |

> **Vision privacy note:** the vision modality is reduced to a per-frame **embedding vector** (e.g., a small MobileNet/pose-keypoint vector of dim `D_v`) on the Pi. Raw pixels are processed in-memory and discarded — they never enter the window buffer that the model or MQTT sees. This is how you satisfy N5 (privacy) *by construction*, and it's a defensible design claim for the paper.

### 3.2 The canonical Window object

```python
# This is the contract. Simulator and live capture both emit this.
@dataclass
class FusionWindow:
    t_start: float            # epoch seconds, window start
    imu:    np.ndarray        # shape (100, 6),  float32
    radar:  np.ndarray        # shape (40, K),   float32
    vision: np.ndarray        # shape (20, D_v), float32
    imu_valid:    bool        # False if modality dropped out this window
    radar_valid:  bool
    vision_valid: bool
    label: Optional[int]      # set only for training data
```

The `*_valid` flags are how F8 (graceful degradation) is implemented end-to-end: a dropped modality is zeroed and **masked out of the attention** rather than fed as garbage.

### 3.3 Synchronization strategy

Sensors run at different rates and on different clocks (ESP32 vs. Pi). Strategy:

- **Single authority clock:** the Pi's monotonic clock is the truth. ESP32 samples carry an ESP32-local timestamp; on first contact the Pi computes a clock offset and corrects incoming timestamps.
- **Windowing by wall-clock buckets:** every 1 s the Pi closes a 2 s window (50% overlap). For each modality it takes all samples whose corrected timestamp falls in the window and **resamples to the fixed length** (linear interpolation for IMU/radar; nearest-frame for vision).
- **Late/missing data:** if a modality contributes < X% of expected samples in a window, mark `*_valid = False`.

This bucket-and-resample approach is simple, robust to jitter, and — crucially — identical in simulation and on hardware.

---

## 4. The fusion model (core contribution)

This is the part your paper is *about*, so it gets the most detail. You're PyTorch-comfortable, so this is spec-level, not tutorial-level.

### 4.1 Shape of the idea

Three modalities → three **per-modality encoders** that each produce a single fixed-dim **token** (embedding) → a **shared self-attention block over the 3 modality tokens** (this is our cross-modal mechanism) → an **attention-weighted pool** that produces the dynamic per-modality trust weights → a **classifier head**.

> **Terminology (be precise in the paper):** we use *merged-token self-attention* — each modality is compressed to one token and the three tokens attend to one another in a single shared Transformer encoder layer. This is a legitimate form of cross-modal attention (modalities exchange information and re-weight each other), and it is chosen deliberately over *pairwise cross-attention over full sequences*. The latter builds an O(sequence²) attention matrix for every modality pair every window, which is incompatible with real-time edge inference on a Pi-class CPU. Pairwise cross-attention *after* token compression is also rejected — it adds parameters and code with no measurable benefit over shared self-attention at 3-token granularity. The interpretable "which sensor did we trust" weights come from the **masked attention-weighted pooling** step, which is the output you export to the dashboard and the paper.

The novelty vs. prior work (Section 9): prior edge HAR fuses **two** modalities (camera+IMU) or applies attention within a **single** modality. FusionSense does **tri-modal** fusion with an attention mechanism whose weights are (a) *interpretable* (you can read which sensor was trusted) and (b) *robust to modality dropout* via masking — all in a model small enough to run on a Pi CPU.

### 4.2 Architecture, layer by layer

```
        IMU (100,6)         Radar (40,K)        Vision (20,D_v)
            │                    │                    │
     ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
     │ 1D-CNN +    │      │ 1D-CNN +    │      │ temporal    │   per-modality
     │ GRU encoder │      │ GRU encoder │      │ pool + MLP  │   encoders
     └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
            │ e_imu (d)          │ e_rad (d)          │ e_vis (d)
            └──────────┬─────────┴──────────┬─────────┘
                       ▼                     ▼
              stack → tokens X = [e_imu, e_rad, e_vis]   shape (3, d)
                       │
              add modality-type embedding + validity mask
                       ▼
        ┌───────────────────────────────────────────┐
        │  Cross-Modal Attention (1–2 Transformer    │
        │  encoder layers, multi-head self-attention │
        │  over the 3 tokens, masked)                │
        └───────────────────────────────────────────┘
                       │  attended tokens (3, d)
                       │  + attention weights A (interpretable output)
                       ▼
              attention-pooled fused vector z (d)
                       ▼
                 MLP classifier → logits (N classes)
```

### 4.3 Component notes

- **IMU / Radar encoders:** a small 1D-CNN (2–3 conv layers over the time axis) to extract local motion features, followed by a GRU (or temporal mean-pool) to summarize into one `d`-dim token. `d ≈ 64–128`.
- **Vision encoder:** frames already arrive as embeddings (Section 3.1), so this is a temporal pool + small MLP — cheap by design.
- **Modality-type embedding:** a learned vector added to each token so the attention block knows *which* sensor a token came from (analogous to positional embeddings).
- **Cross-modal attention:** 1–2 standard Transformer encoder layers operating over a sequence of **just 3 tokens**. This is what keeps it lightweight (N3) — attention over 3 tokens is nearly free; the cost is in the encoders, which are small CNNs.
- **Validity masking:** invalid modalities get their attention scores set to −∞ before softmax, so a dropped sensor contributes exactly zero and the remaining sensors renormalize. This is F8 implemented in one line.
- **Interpretability output:** export the attention weights `A` alongside the prediction. Publishing these to the dashboard ("radar 0.7, camera 0.1, imu 0.2 in the dark") is both a great demo and a paper figure.

### 4.4 Loss & training

- Cross-entropy on activity class, with **class weighting** for the rare/critical `fall` class.
- **Modality-dropout augmentation during training:** randomly zero a modality (and set its valid flag false) for some fraction of batches. This is what *teaches* the attention to reweight — without it, the model leans on whichever modality is easiest and never learns robustness. **This single technique is arguably your strongest novelty/ablation result.**
- Optional auxiliary loss: encourage attention to match a "ground-truth reliability" signal when you know a modality was degraded (e.g., you simulated darkness).

### 4.5 Reference module sketch

```python
class ModalityEncoder(nn.Module):
    def __init__(self, in_ch, d=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, d, 5, padding=2),    nn.ReLU())
        self.gru = nn.GRU(d, d, batch_first=True)
    def forward(self, x):            # x: (B, T, C)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d)
        _, hn = self.gru(h)
        return hn[-1]                # (B, d) — one token

class FusionSense(nn.Module):
    def __init__(self, d=128, n_classes=5, n_heads=4, n_layers=2):
        super().__init__()
        self.enc_imu   = ModalityEncoder(6,   d)
        self.enc_radar = ModalityEncoder(K,   d)
        self.enc_vis   = ModalityEncoder(D_v, d)
        self.mod_emb   = nn.Parameter(torch.randn(3, d))   # modality-type
        layer = nn.TransformerEncoderLayer(d, n_heads, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, imu, radar, vis, valid):   # valid: (B,3) bool
        toks = torch.stack([self.enc_imu(imu),
                            self.enc_radar(radar),
                            self.enc_vis(vis)], dim=1)      # (B,3,d)
        toks = toks + self.mod_emb
        pad_mask = ~valid                                   # True = ignore
        z = self.attn(toks, src_key_padding_mask=pad_mask)  # (B,3,d)
        # masked mean-pool over valid tokens
        w = valid.unsqueeze(-1).float()
        z = (z * w).sum(1) / w.sum(1).clamp(min=1)
        return self.head(z)
```

> This is a faithful skeleton, not the final code. When we move to the implementation phase I'll flesh out the encoders, the windowing/dataset code, the training loop, metrics, and the TFLite/ONNX export for the Pi.

---

## 5. Architecture Decision Records

Short ADRs for the decisions that actually shape the build. Format follows your `engineering:architecture` skill.

### ADR-001: Cross-modal attention vs. naive fusion

**Status:** Accepted
**Context:** Need to combine three mismatched sensors; naive averaging/concatenation produces conflicting signals and high false-positive falls (your problem statement).

| Option | Complexity | Robust to dropout | Interpretable | Edge cost |
|--------|-----------|-------------------|---------------|-----------|
| A. Feature concatenation + MLP | Low | Poor (fixed weights) | No | Low |
| B. Hand-tuned weighted average | Low | Poor | Partly | Very low |
| C. **Cross-modal attention (chosen)** | Med | **Good (masking)** | **Yes (weights)** | Low (3 tokens) |

**Decision:** Option C. The attention is over only 3 tokens, so the "Transformer" is cheap; the win is dynamic, interpretable, dropout-robust weighting — which is exactly the paper's claim. **Consequence:** more training complexity and the need for modality-dropout augmentation; revisit head design if classes grow beyond ~10.

### ADR-002: Where to split compute (ESP32 vs. Pi)

**Status:** Accepted
**Context:** Could push some inference onto the ESP32, or treat it purely as a gateway.
**Decision:** ESP32 = acquisition + timestamping + framing only; all ML on the Pi. ESP32 lacks the RAM/throughput for the camera path and the model; splitting the model across two clocks would be a synchronization nightmare. **Consequence:** ESP32 firmware stays simple and testable; the Pi is the single point of compute (and single point of failure — acceptable for v1).

### ADR-003: Transport between tiers

**Status:** Accepted
**Context:** ESP32→Pi link and Pi→dashboard link.
**Decision:** ESP32→Pi over **UART (wired) for v1**, with Wi-Fi/ESP-NOW as a later option; Pi→dashboard over **MQTT** (local Mosquitto). UART removes wireless jitter while you're still validating sync; MQTT is the natural fit for the pub/sub dashboard and matches your project plan. **Consequence:** wired tether for the demo; revisit wireless once timing is proven.

### ADR-004: Model export / runtime on the Pi

**Status:** Proposed
**Context:** Train in PyTorch; must run fast on Pi 5 CPU.
**Decision (proposed):** train in PyTorch → export to **ONNX or TFLite**, apply **int8 post-training quantization**, run with onnxruntime/tflite-runtime on the Pi. Keeps training ergonomic and inference small (N2/N3). **Consequence:** need a quantization-validation step (accuracy drop check) before deployment; revisit if accuracy loss > ~2%.

### ADR-005: Window size & overlap

**Status:** Proposed
**Context:** Trade latency (N1) against having enough signal to classify, especially falls.
**Decision (proposed):** 2 s window, 1 s stride (50% overlap). Falls are sub-second events but need pre/post context; 2 s is the common HAR sweet spot. **Consequence:** ~1 s detection granularity; revisit with a shorter fast-path window if fall latency is too high.

---

## 6. Data strategy (the hidden hard part)

You have no hardware and there is **no off-the-shelf tri-modal (camera+mmWave+IMU) dataset**. Do not let this block the model work. Three-stage plan, in order:

**Stage A — Build the model against a simulator (start now).**
Write a synthetic `FusionWindow` generator: parametric signals per activity (e.g., walking = periodic IMU + steady radar range + bobbing vision embedding; fall = IMU spike + sudden radar range change + vision vertical drop). Add noise and *simulate modality degradation* (zero the vision channel to mimic darkness). This lets you build and unit-test the **entire training plane and the attention/masking logic** before any sensor exists. It also produces your first ablation: does attention shift to radar when vision is zeroed? If yes, the core thesis works.

**Stage B — Pretrain encoders on real single-modality public datasets.**
Even without a combined dataset, each modality has public data you can use to pretrain its encoder:
- IMU/HAR: UCI-HAR, PAMAP2, WISDM, MotionSense.
- Vision HAR / pose: public action-recognition or pose-keypoint datasets.
- Radar: published mmWave HAR datasets exist (smaller; survey them during the literature phase).
Pretrain each `ModalityEncoder` separately, then fine-tune the fusion stack. This is a legitimate, paper-worthy transfer-learning angle.

**Stage C — Collect a small real tri-modal set (once hardware arrives).**
You don't need thousands of samples — a few subjects × a handful of activities × many repetitions, recorded through the *same data contract*, is enough to fine-tune and, importantly, to **report real-world numbers** in the paper. Record raw → derive windows offline.

> Search the literature (alphaXiv / Google Scholar) for existing mmWave+IMU or mmWave+vision HAR datasets before committing to Stage C — if one exists, it saves you weeks. I can run that search when you're ready.

---

## 7. Repository scaffold (software-first)

Build this structure now; it cleanly separates the training plane (works today) from the edge plane (needs hardware).

```
fusionsense/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml            # window size, rates, model dims, classes
├── fusionsense/
│   ├── contract.py             # FusionWindow dataclass (Section 3.2)
│   ├── data/
│   │   ├── simulator.py        # Stage-A synthetic generator  ← build first
│   │   ├── public_loaders.py   # UCI-HAR / PAMAP2 / radar adapters (Stage B)
│   │   ├── windowing.py        # bucket-and-resample sync (Section 3.3)
│   │   └── dataset.py          # torch Dataset over FusionWindows
│   ├── models/
│   │   ├── encoders.py         # ModalityEncoder
│   │   ├── fusion.py           # FusionSense (Section 4.5)
│   │   └── export.py           # ONNX/TFLite + int8 quant (ADR-004)
│   ├── train/
│   │   ├── loop.py             # training, modality-dropout aug, class weights
│   │   └── metrics.py          # accuracy, F1, per-class fall recall, confusion
│   └── infer/
│       ├── pipeline.py         # live windowing → model → result
│       └── mqtt_publisher.py
├── edge/                       # ← needs hardware, stub for now
│   ├── esp32/                  # C/C++ firmware: I²C IMU + UART radar + framing
│   └── pi/                     # camera embed extractor, serial reader, broker cfg
├── dashboard/
│   └── node_red_flow.json
└── tests/
    ├── test_windowing.py       # sync correctness
    ├── test_masking.py         # dropout → renormalized attention
    └── test_shapes.py          # contract conformance
```

**The golden rule:** `simulator.py` and the real `edge/` capture both emit the *same* `FusionWindow`. Everything in `models/`, `train/`, and `infer/` is hardware-agnostic and testable on your laptop today.

---

## 8. Implementation roadmap (software-first reordering of your plan)

Your slide plan is hardware-first (ESP32 → pipeline → AI → deploy). Since hardware isn't here, **invert it** so you make real progress now and de-risk the AI — the part the paper rests on.

| Phase | Focus | Deliverable | Needs HW? |
|-------|-------|-------------|-----------|
| **0 (now)** | Contract + simulator + repo scaffold | `FusionWindow`, synthetic dataset, passing tests | No |
| **1** | Fusion model + training plane | Trained model on sim data; attention reweights under dropout (first result) | No |
| **2** | Encoder pretraining on public data | Per-modality encoders + transfer-learning ablation | No |
| **3** | Windowing/sync + inference pipeline + MQTT + dashboard (run on simulated stream) | End-to-end live pipeline, sensors mocked | No |
| **4** | ESP32 firmware + Pi capture, swap mock → real sensors | Real tri-modal capture through the same contract | **Yes** |
| **5** | Quantize, deploy to Pi, measure latency, collect Stage-C data | On-device numbers for the paper | **Yes** |

Phases 0–3 are everything except the physical sensors — and they include your headline result (attention-based robustness). You can write most of the paper from Phases 1–3.

---

## 9. Research-paper framing

**Working contribution statement:** *"A lightweight, interpretable cross-modal attention model that fuses vision, mmWave radar, and inertial data for human activity recognition, runs in real time on a Raspberry Pi-class edge device, and degrades gracefully under per-modality sensor failure."*

What makes it publishable (each maps to prior-work gaps from your Slide 7):

1. **Tri-modal, not bi-modal** — Nakabayashi & Saito (2024) do camera+IMU on edge; you add radar for dark/privacy-sensitive settings.
2. **Edge-deployed attention** — Koupai et al. (2022) use fusion Transformers but on cloud/desktop; you bring it to a Pi with quantization numbers.
3. **Robustness as a measured property** — modality-dropout training + masked attention, with an ablation showing graceful degradation. Prior surveys (Aguileta 2019) note naive fusion fails under sensor loss; you quantify that you don't.
4. **Interpretability** — published attention weights as a reliability signal ("trusted radar in the dark"). This is a clean figure.

**Suggested evaluation table for the paper:** overall accuracy & macro-F1; per-class fall recall; accuracy under each single-modality dropout (vision-off ≈ darkness); model size and Pi-5 latency; ablation with/without modality-dropout training.

---

## 10. Glossary

- **HAR** — Human Activity Recognition: classifying what a person is doing from sensor data.
- **Modality** — one sensor "type" / information channel (vision, radar, IMU).
- **IMU (MPU-6050)** — inertial sensor giving 3-axis acceleration + 3-axis rotation (6-DoF).
- **mmWave radar (LD2410)** — millimetre-wave radar; detects presence/distance/micro-motion, works in the dark, sees no identifiable image (privacy-friendly).
- **Token / embedding** — a fixed-length vector summarizing a modality's window.
- **Cross-modal attention** — a mechanism that lets each modality's token weigh the others and produces per-modality importance weights.
- **Modality dropout** — deliberately removing a modality (in training or at runtime) to build/test robustness.
- **Quantization** — shrinking a model (e.g., float32 → int8) so it runs faster/smaller on the edge.
- **MQTT** — lightweight publish/subscribe messaging used to send results to the dashboard.

---

*Next step: when you're ready to write code, we start at Phase 0 — the `FusionWindow` contract and the simulator — and build the model against it. I can scaffold the repo and the first training run on synthetic data without any hardware.*
