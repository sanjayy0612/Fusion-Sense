# FusionSense — Literature Review, Novelty Assessment & Plan

Date: 2026-06-29 · Status: Discovery-level review (≈30 papers surfaced via alphaXiv; not yet a full 60-paper deep read)

> **Read this first.** I ran a structured search across every sensor-combination and every "novelty axis" we discussed. The honest headline is below, and some of it corrects things you and I said earlier. I'd rather give you the uncomfortable-but-true version now than let you build months of code on a novelty claim a reviewer will puncture.

---

## 0. The honest headline

1. **Your tri-modal hardware combination (camera + mmWave radar + wearable IMU, edge-deployed) appears genuinely uncommon** — I did not find a published system that integrates exactly these three. That's real, but on its own it reads as *"engineering integration,"* not a research contribution.
2. **Almost every individual "novelty axis" we got excited about already exists in the literature.** Reliability-aware fusion, modality-dropout robustness, missing-modality transformers, interpretable fusion, uncertainty-aware HAR, edge HAR — each is an active or mature sub-field with recent papers. **This is the part that corrects our earlier optimism.** "Reliability-aware cross-modal attention" is *not* new in general multimodal ML.
3. **So the defensible contribution is not a new fusion equation.** It's the **systems + empirical** contribution: a deployed tri-modal edge HAR system studied under *real, physically-induced* sensor degradation, using *hardware-reported sensor-health signals* as explicit reliability priors. That last idea (Section 5) is the one angle I found genuinely under-explored.
4. **Venue calibration:** as-is, V1 is a solid **undergraduate IEEE conference / workshop / Sensors-type** paper. A top-tier ML venue would reject "we combined three sensors with attention." Aim realistically.

---

## 1. The five generations (where the field is)

| Gen | Approach | Maturity | Representative work (arXiv) |
|-----|----------|----------|------------------------------|
| 1 | Single sensor (camera OR IMU OR radar) | Mature | UCI-HAR-era; radar: RadMamba (2504.12039), Neural-HAR (2510.22772) |
| 2 | Two-sensor fusion | Very mature | Radar+IMU intake detection (2507.07261); WiFi+vision MaskFi (2402.19258) |
| 3 | Attention/Transformer multimodal fusion | Active | MMTSA (2210.09222); your cited Koupai (2209.03765) |
| 4 | Edge / lightweight HAR | Growing fast | TinierHAR (2507.07949), nanoML (2502.12173), SPECTRA (2603.26482), MicroBi-ConvLSTM (2602.06523) |
| 5 | Reliability-aware / adaptive / explainable / uncertainty | **Emerging — and more crowded than we thought** | Unbiased Dynamic Multimodal Fusion (2603.19681); Contextual Calibration (2606.02679); CARING uncertainty HAR (2101.00468) |

FusionSense sits at the **Gen 4 ↔ Gen 5 boundary**. Good place to be — but Gen 5 is filling up.

---

## 2. Literature by sensor combination (the overlap map)

**Camera + IMU** — well established. *COMODO* (2503.07259) distills video→IMU; *CMD-HAR* (2503.21843) does cross-modal disentanglement for wearable HAR. Cross-modal attention here is *done*.

**Radar + IMU** — growing quickly. *Robust Multimodal Intake Gesture Detection using contactless radar + wearable IMU* (2507.07261, KU Leuven) is very close in spirit to your motion-sensing pair, **and it's explicitly about robustness**. Read this one closely — it overlaps your robustness pitch.

**Camera + Radar** — common in autonomous driving (GaussianCaR 2602.08784, hybrid-attention 2604.04797) and increasingly in fall detection. *Edge-Efficient Two-Stream Multimodal bathroom fall detection* (2603.17069) fuses mmWave + another modality on the edge — a direct neighbor of your application.

**Tri-modal (camera + radar + IMU)** — **the gap.** Closest hits are *Triple Spectral Fusion* (2605.02743, but that's three spectral views of IMU-type data, not three physical sensors) and *Real-Time HAR on Edge Microcontrollers with multi-spectral fusion* (2602.00152). **No integrated camera + mmWave + wearable-IMU edge system surfaced.** This is your strongest structural claim — keep it, but frame it carefully.

---

## 3. The "novelty axes" — reality check (this corrects our earlier chat)

| Axis we called novel | Reality from the literature | Verdict |
|----------------------|------------------------------|---------|
| Cross-modal attention | Everywhere (MMTSA, Koupai, CMD-HAR) | **Not novel** |
| Modality dropout / missing-modality robustness | Whole sub-field: *Are MM Transformers Robust to Missing Modality?* (2204.05454), *Masked Modality Projection* (2410.03010), *Cross-Modal Proxy Tokens* (2501.17823), *survey* (2409.07825) | **Not novel by itself** — this was our biggest overclaim |
| Reliability/confidence-aware fusion | Active: *Unbiased Dynamic Multimodal Fusion* (2603.19681), *Contextual Calibration* (2606.02679) | **Not novel by itself** |
| Uncertainty-aware HAR | Exists since *CARING* (2101.00468) | **Not novel** |
| Interpretable per-modality weights | *Layer-Wise Modality Decomposition* (2511.00859) | **Not novel** |
| Edge/lightweight HAR | Saturated (TinierHAR, nanoML, SPECTRA, RadMamba…) | **Not novel** |
| Tri-modal camera+radar+IMU *integration* | Not found as one system | **Uncommon — defensible** |

**Takeaway:** novelty cannot come from any single mechanism. It has to come from a *specific combination plus a specific empirical question nobody has answered for this sensor mix.*

---

## 4. Where the genuine gaps actually are

After filtering for what's truly under-explored *for this specific tri-modal edge setting*:

**Gap A — Real (not simulated) heterogeneous sensor degradation on deployed hardware.** Nearly all missing-modality papers drop modalities *artificially on benchmark datasets*. Very few study what happens when a *real* camera is *actually* blinded, a *real* radar is *actually* noisy, and a *real* IMU *actually* saturates — measured end-to-end on an edge device. An honest empirical study here is a legitimate contribution.

**Gap B — Hardware-reported sensor-health as a reliability prior (the best angle — see Section 5).** Existing reliability-aware fusion *infers* modality quality from learned features. Your sensors *report their own health for free* (radar gate energy, image brightness/blur, IMU clipping). Feeding those physical signals into the attention as explicit priors is concrete, cheap, and under-explored.

**Gap C — Tri-modal complementarity under the *privacy/darkness* axis specifically.** A focused study of "vision dies in the dark → radar+IMU carry the load, quantified" is application-relevant for elderly care and not well covered for this exact triple.

**Gaps we should *de-prioritize*** (crowded, hard, or not feasible in your timeline): adaptive windowing, hierarchical fusion, self-supervised tri-modal pretraining, full uncertainty calibration. These are real but each is its own paper and several need data you won't have.

---

## 5. The one angle I'd actually build the paper around

**"Sensor-health-conditioned cross-modal attention."**

The insight: every sensor in your rig emits a cheap, physical *self-report of its own reliability*, and the literature's reliability-aware methods mostly ignore these in favor of learned confidence:

- **LD2410 radar** reports per-gate **energy / signal strength** — low energy = unreliable detection.
- **Camera** gives free image-quality cues — **mean brightness** (darkness) and **Laplacian variance** (blur/occlusion).
- **MPU-6050 IMU** reports **range saturation / clipping** and can flag disconnection.

Feed these three scalar *health signals* as an explicit prior into the attention/pooling, alongside the learned features. Now your model's trust weights are grounded in **physically measurable sensor state**, not just learned correlations — which is (a) more interpretable, (b) more robust, (c) cheaper, and (d) **a framing I did not find for tri-modal edge HAR.** It directly turns your earlier "reliability-aware" idea into something concrete and defensible.

This is the difference between "we added confidence scores" (crowded) and "we condition fusion on free hardware reliability telemetry, validated under real degradation on a Pi" (much narrower, much more your own).

---

## 6. Application framing (answering your direct question)

**Is FusionSense an elderly-monitoring system?** Technically **no**; application-wise **yes** — and you should write it exactly that way:

> *FusionSense is a lightweight, sensor-health-aware multimodal HAR framework for edge devices. We demonstrate it on elderly fall monitoring.*

The **framework** (tri-modal, health-conditioned fusion, edge) is the contribution; **elderly fall detection** is the demonstration. This makes it reusable (hospital, rehab, workplace safety) and stops a reviewer from dismissing it as one narrow app.

---

## 7. The version roadmap (revised to match the evidence)

| Version | What it is | Honest contribution level | Realistic venue |
|---------|-----------|---------------------------|-----------------|
| **V1 (build this first)** | Working tri-modal edge prototype: camera + radar + IMU → merged-token attention → HAR → MQTT dashboard. | Systems integration + working demo | Project / workshop / poster |
| **V2 (the paper)** | Add **sensor-health-conditioned attention** (Section 5) + a real **degradation study** (Gap A) + interpretable trust output. | Genuine empirical + modest methodological | IEEE conference / Sensors-type journal |
| **V3 (stretch)** | Move toward camera+radar-only (drop the wearable), add uncertainty calibration, or context-conditioned attention. | Methodological | Stronger journal |

**Do not** try to make V1 itself "novel by architecture." Its job is to *work*. The novelty is bolted on in V2 once the rig runs.

---

## 8. What to do — concrete next steps

1. **Build V1 now, software-first.** Start the repo + simulator (the FusionWindow generator). You don't need novelty resolved to start coding the working system — and a running system is your stated #1 goal.
2. **Bake the V2 hooks in from day one.** Make the `FusionWindow` carry three extra scalars — `radar_energy`, `image_quality`, `imu_clip` — even if the simulator fakes them. That one decision keeps the Section-5 idea cheap to add later instead of a rewrite.
3. **Read the 4 closest neighbors before writing any claims:** intake-gesture radar+IMU robustness (2507.07261), bathroom-fall edge two-stream (2603.17069), Unbiased Dynamic Multimodal Fusion (2603.19681), missing-modality survey (2409.07825). These define what you must differentiate against.
4. **When ready, I run the full 50–70 paper deep review** and build the real gap matrix with per-paper contribution columns, so your related-work section and novelty claims are airtight before submission.

---

## Appendix — papers surfaced (arXiv IDs)

Tri-modal / fusion: 2605.02743, 2602.00152, 2503.07259, 2507.07261, 2602.08784, 2603.17069, 2210.09222, 2209.03765 (cited in your slides).
Missing-modality / robustness: 2204.05454, 2410.03010, 2501.17823, 2409.07825, 2407.04458, 2503.21843, 2402.19258, 2604.08971, 2511.00859.
Reliability / uncertainty / dynamic fusion: 2603.19681, 2606.02679, 2606.05437, 2604.06728, 2604.04797, 2101.00468.
Edge / lightweight / radar HAR: 2512.12013, 2603.27571, 2602.06523, 2504.12039, 2507.07949, 2502.13-era nanoML (2502.12173), 2510.22772, 2510.26148, 2605.20649, 2603.26482.

*This is a discovery-level list. The full deep review (step 4) will read these, add ~30 more, and extract per-paper contribution/sensor/edge/robustness columns.*
