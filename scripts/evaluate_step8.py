"""Evaluate Step 8 fall safety, robustness, synchronization, and laptop latency."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionsense.config import CFG  # noqa: E402
from fusionsense.contract import FusionWindow, LABEL2ID  # noqa: E402
from fusionsense.data.cmhad_loader import (  # noqa: E402
    fit_normalization,
    load_cmhad_windows,
    normalize_windows,
)
from fusionsense.data.splitting import split_paired_windows  # noqa: E402
from fusionsense.evaluation import (  # noqa: E402
    binary_alert_metrics,
    binary_brier_score,
    expected_calibration_error,
    shift_time_series,
)
from fusionsense.inference import classification_metrics  # noqa: E402
from fusionsense.models.encoders import EncoderClassifier  # noqa: E402
from fusionsense.models.fusion import FusionSense  # noqa: E402


THRESHOLDS = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
SYNC_SHIFTS_MS = [-500, -200, -100, -50, 0, 50, 100, 200, 500]


def load_stats(directory: Path) -> dict[str, np.ndarray]:
    saved = np.load(directory / "normalization.npz", allow_pickle=False)
    return {name: saved[name].astype(np.float32) for name in saved.files}


def load_fusion(directory: Path, device: torch.device) -> FusionSense:
    model = FusionSense(CFG).to(device)
    state = torch.load(
        directory / "fusionsense_cmhad.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def normalized_arrays(windows, stats, shift_ms=0.0):
    imu = np.stack(
        [
            shift_time_series(window.imu, shift_ms, CFG.imu_hz)
            if shift_ms
            else window.imu
            for window in windows
        ]
    ).astype(np.float32)
    vision = np.stack([window.vision for window in windows]).astype(np.float32)
    imu = (imu - stats["imu_mean"]) / stats["imu_std"]
    vision = (vision - stats["vision_mean"]) / stats["vision_std"]
    radar = np.stack([window.radar for window in windows]).astype(np.float32)
    valid = np.stack([window.valid_vector() for window in windows]).astype(bool)
    health = np.stack([window.health_vector() for window in windows]).astype(np.float32)
    return imu, radar, vision, valid, health


def apply_mode(valid: np.ndarray, health: np.ndarray, mode: str):
    valid = valid.copy()
    health = health.copy()
    if mode == "imu_only":
        valid[:, 2] = False
        health[:, 2] = 0.0
    elif mode == "camera_only":
        valid[:, 0] = False
        health[:, 0] = 0.0
    elif mode != "both":
        raise ValueError(f"Unknown modality mode {mode}")
    if np.any(valid.sum(axis=1) == 0):
        raise ValueError(f"Mode {mode} leaves at least one window without a modality")
    return valid, health


@torch.inference_mode()
def predict_fusion(model, windows, stats, device, *, mode="both", shift_ms=0.0):
    imu, radar, vision, valid, health = normalized_arrays(windows, stats, shift_ms)
    valid, health = apply_mode(valid, health, mode)
    probabilities, trust_values = [], []
    for first in range(0, len(windows), 128):
        last = first + 128
        logits, trust = model(
            torch.from_numpy(imu[first:last]).to(device),
            torch.from_numpy(radar[first:last]).to(device),
            torch.from_numpy(vision[first:last]).to(device),
            torch.from_numpy(valid[first:last]).to(device),
            torch.from_numpy(health[first:last]).to(device),
            return_trust=True,
        )
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        trust_values.append(trust.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(trust_values)


@torch.inference_mode()
def benchmark_fusion(model, window, stats, device, repeats=250):
    arrays = normalized_arrays([window], stats)
    tensors = [torch.from_numpy(array).to(device) for array in arrays]
    for _ in range(25):
        model(*tensors)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        model(*tensors)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return distribution(elapsed)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(values)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def evaluate_probabilities(labels, probabilities, threshold=0.6):
    labels = np.asarray(labels, dtype=np.int64)
    truth = labels == LABEL2ID["stand_to_fall"]
    fall_probability = probabilities[:, LABEL2ID["stand_to_fall"]]
    return {
        "classification": classification_metrics(labels, probabilities.argmax(axis=1)),
        "alert_at_0_60": binary_alert_metrics(truth, fall_probability, threshold),
        "fall_brier_score": binary_brier_score(truth, fall_probability),
        "fall_expected_calibration_error_10_bins": expected_calibration_error(
            truth, fall_probability, bins=10
        ),
        "fall_probability": {
            "positive_median": float(np.median(fall_probability[truth])),
            "positive_min": float(fall_probability[truth].min()),
            "negative_p95": float(np.percentile(fall_probability[~truth], 95)),
            "negative_max": float(fall_probability[~truth].max()),
        },
        "threshold_sweep": [
            binary_alert_metrics(truth, fall_probability, value)
            for value in THRESHOLDS
        ],
    }


def evaluate_baseline(directory, modality, windows, stats, device):
    values = np.stack([getattr(window, modality) for window in windows]).astype(np.float32)
    values = (values - stats[f"{modality}_mean"]) / stats[f"{modality}_std"]
    input_channels = values.shape[-1]
    model = EncoderClassifier(input_channels, CFG.d_model, CFG.n_classes).to(device)
    payload = torch.load(
        directory / f"baseline_{modality}.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(values).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return model, probabilities


def benchmark_baseline(model, window, modality, stats, device, repeats=250):
    values = getattr(window, modality)[None].astype(np.float32)
    values = (values - stats[f"{modality}_mean"]) / stats[f"{modality}_std"]
    tensor = torch.from_numpy(values).to(device)
    with torch.inference_mode():
        for _ in range(25):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return distribution(elapsed)


def benchmark_pose(frames_directory: Path, samples=100):
    from fusionsense.data.vision_extractor import PoseExtractor, _lazy_imports

    cv2, _ = _lazy_imports()
    all_paths = sorted(frames_directory.glob("*.jpg"))
    if not all_paths:
        raise FileNotFoundError(f"No JPEG frames under {frames_directory}")
    if len(all_paths) > samples:
        indices = np.linspace(0, len(all_paths) - 1, samples, dtype=np.int64)
        paths = [all_paths[index] for index in indices]
    else:
        paths = all_paths
    elapsed, valid = [], 0
    with PoseExtractor() as extractor:
        for index, path in enumerate(paths):
            started = time.perf_counter_ns()
            frame = cv2.imread(str(path))
            result = extractor.extract(frame, captured_at=index / CFG.vision_fps)
            elapsed.append((time.perf_counter_ns() - started) / 1e6)
            valid += int(result.valid)
    return {
        **distribution(elapsed),
        "source_frames": len(all_paths),
        "valid_frames": valid,
        "valid_ratio": valid / len(paths),
    }


def count_alert_episodes(background, alert):
    episodes = 0
    previous_recording = None
    previous_start = None
    previous_alert = False
    for window, active in sorted(
        zip(background, alert), key=lambda item: (item[0].recording_id, item[0].t_start)
    ):
        contiguous = (
            window.recording_id == previous_recording
            and previous_start is not None
            and window.t_start - previous_start <= CFG.window_seconds + 1e-6
        )
        if active and not (previous_alert and contiguous):
            episodes += 1
        previous_recording = window.recording_id
        previous_start = window.t_start
        previous_alert = bool(active)
    return episodes


def background_probability_report(background, fall_probability, hours, threshold=0.6):
    truth = np.zeros(len(background), dtype=bool)
    alert = fall_probability >= threshold
    episodes = count_alert_episodes(background, alert)
    threshold_sweep = []
    for value in THRESHOLDS:
        threshold_alert = fall_probability >= value
        threshold_episodes = count_alert_episodes(background, threshold_alert)
        threshold_sweep.append(
            {
                **binary_alert_metrics(
                    truth, fall_probability, value, negative_hours=hours
                ),
                "false_alert_episodes": threshold_episodes,
                "false_alert_episodes_per_hour": threshold_episodes / hours,
            }
        )
    ranked = np.argsort(fall_probability)[::-1][:20]
    return {
        "alert": binary_alert_metrics(
            truth, fall_probability, threshold, negative_hours=hours
        ),
        "false_alert_episodes": episodes,
        "false_alert_episodes_per_hour": episodes / hours,
        "threshold_sweep": threshold_sweep,
        "fall_probability": {
            "median": float(np.median(fall_probability)),
            "p95": float(np.percentile(fall_probability, 95)),
            "p99": float(np.percentile(fall_probability, 99)),
            "max": float(fall_probability.max()),
        },
        "highest_probability_windows": [
            {
                "recording_id": background[index].recording_id,
                "t_start_s": float(background[index].t_start),
                "fall_probability": float(fall_probability[index]),
            }
            for index in ranked
        ],
        "vision_valid_windows": int(sum(window.vision_valid for window in background)),
        "windows": len(background),
    }


def background_report(model, background, stats, device, hours, threshold=0.6):
    probabilities, _ = predict_fusion(model, background, stats, device, mode="both")
    fall_probability = probabilities[:, LABEL2ID["stand_to_fall"]]
    return background_probability_report(
        background, fall_probability, hours, threshold=threshold
    )


def load_recorded_hardware_windows(path: Path):
    data = np.load(path, allow_pickle=False)
    windows = []
    for index in range(len(data["t_start"])):
        window = FusionWindow(
            t_start=float(data["t_start"][index]),
            imu=data["imu"][index].astype(np.float32),
            radar=data["radar"][index].astype(np.float32),
            vision=data["vision"][index].astype(np.float32),
            imu_valid=bool(data["imu_valid"][index]),
            radar_valid=bool(data["radar_valid"][index]),
            vision_valid=bool(data["vision_valid"][index]),
            imu_health=float(data["imu_health"][index]),
            radar_energy=0.0,
            image_quality=float(data["image_quality"][index]),
            label=0,
            subject_id="live_hardware_user",
            recording_id="fusion_usb_step4_20260819",
        )
        if window.imu_valid and window.vision_valid and not window.radar_valid:
            windows.append(window)
    return windows


def unlabeled_nonfall_replay_report(model, windows, stats, device, threshold=0.6):
    probabilities, trust = predict_fusion(model, windows, stats, device, mode="both")
    fall_probability = probabilities[:, LABEL2ID["stand_to_fall"]]
    return {
        "assumed_ground_truth": "non-fall synchronization movements; not formally labelled",
        "eligible_bimodal_windows": len(windows),
        "alert_windows": int(np.sum(fall_probability >= threshold)),
        "fall_probability": {
            "median": float(np.median(fall_probability)),
            "p95": float(np.percentile(fall_probability, 95)),
            "max": float(fall_probability.max()),
        },
        "mean_sensor_trust": {
            "imu": float(trust[:, 0].mean()),
            "radar": float(trust[:, 1].mean()),
            "camera": float(trust[:, 2].mean()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each corrected fusion checkpoint directory",
    )
    parser.add_argument("--background-cache", type=Path, required=True)
    parser.add_argument("--background-metadata", type=Path, required=True)
    parser.add_argument(
        "--hardware-session",
        type=Path,
        default=Path("data/recordings/fusion_usb_step4_20260819"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    all_windows = load_cmhad_windows()
    train_windows, validation_windows, split = split_paired_windows(all_windows, seed=42)
    validation_windows = [
        window
        for window in validation_windows
        if window.imu_valid and window.vision_valid and not window.radar_valid
    ]
    labels = np.asarray([window.label for window in validation_windows], dtype=np.int64)
    background = load_cmhad_windows(args.background_cache)
    background_metadata = json.loads(args.background_metadata.read_text(encoding="utf-8"))
    negative_hours = float(background_metadata["negative_hours"])
    hardware_windows = load_recorded_hardware_windows(
        args.hardware_session / "step5_windows.npz"
    )
    evaluation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hardware = json.loads((args.hardware_session / "session.json").read_text(encoding="utf-8"))
    validation_hardware = hardware["validation"]
    step5 = json.loads(
        (args.hardware_session / "step5_validation.json").read_text(encoding="utf-8")
    )["validation"]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "four-subject pilot; Subject3 validation has been observed during tuning",
        "split": split,
        "validation_windows": len(validation_windows),
        "validation_falls": int(np.sum(labels == LABEL2ID["stand_to_fall"])),
        "alert_threshold": 0.6,
        "background": background_metadata,
        "models": {},
        "hardware_synchronization": {
            "post_calibration_alignment_lag_ms": validation_hardware["alignment"][
                "absolute_lag_ms"
            ],
            "shared_motion_correlation": validation_hardware["alignment"][
                "best_correlation"
            ],
            "clock_fit_residual_p95_ms": {
                "imu": validation_hardware["clock_sync"]["imu"]["mapping"][
                    "fit_residual_ms"
                ]["p95"],
                "camera": validation_hardware["clock_sync"]["camera"]["mapping"][
                    "fit_residual_ms"
                ]["p95"],
            },
            "transport_latency_ms": validation_hardware["transport_latency"],
            "window_sampling_skew_ms": {
                "imu": step5["diagnostics"]["imu_max_sampling_skew_ms"],
                "camera": step5["diagnostics"]["camera_max_sampling_skew_ms"],
            },
            "recorded_window_validity": step5["validity"],
        },
        "latency": {
            "window_seconds": CFG.window_seconds,
            "live_stride_seconds": 1.0,
            "pose_decode_and_extract_ms": benchmark_pose(args.hardware_session / "frames"),
            "note": "Model timings are batch-1; acquisition/window wait is reported separately.",
        },
    }

    baseline_directory = args.checkpoint[-1]
    baseline_stats = load_stats(baseline_directory)
    normalized_validation = normalize_windows(validation_windows, baseline_stats)
    for modality in ("imu", "vision"):
        model, probabilities = evaluate_baseline(
            baseline_directory,
            modality,
            normalized_validation,
            {"imu_mean": np.zeros(CFG.imu_ch, np.float32),
             "imu_std": np.ones(CFG.imu_ch, np.float32),
             "vision_mean": np.zeros(CFG.vision_dv, np.float32),
             "vision_std": np.ones(CFG.vision_dv, np.float32)},
            evaluation_device,
        )
        name = f"{modality}_only_baseline"
        report["models"][name] = evaluate_probabilities(labels, probabilities)
        _, background_probabilities = evaluate_baseline(
            baseline_directory,
            modality,
            background,
            baseline_stats,
            evaluation_device,
        )
        report["models"][name]["continuous_background"] = (
            background_probability_report(
                background,
                background_probabilities[:, LABEL2ID["stand_to_fall"]],
                negative_hours,
            )
        )
        latency_stats = {str(evaluation_device): benchmark_baseline(
                model,
                normalized_validation[0],
                modality,
                {"imu_mean": np.zeros(CFG.imu_ch, np.float32),
                 "imu_std": np.ones(CFG.imu_ch, np.float32),
                 "vision_mean": np.zeros(CFG.vision_dv, np.float32),
                 "vision_std": np.ones(CFG.vision_dv, np.float32)},
                evaluation_device,
            )}
        if evaluation_device.type == "cuda":
            cpu = torch.device("cpu")
            cpu_model, _ = evaluate_baseline(
                baseline_directory,
                modality,
                normalized_validation,
                {"imu_mean": np.zeros(CFG.imu_ch, np.float32),
                 "imu_std": np.ones(CFG.imu_ch, np.float32),
                 "vision_mean": np.zeros(CFG.vision_dv, np.float32),
                 "vision_std": np.ones(CFG.vision_dv, np.float32)},
                cpu,
            )
            latency_stats["cpu"] = benchmark_baseline(
                cpu_model,
                normalized_validation[0],
                modality,
                {"imu_mean": np.zeros(CFG.imu_ch, np.float32),
                 "imu_std": np.ones(CFG.imu_ch, np.float32),
                 "vision_mean": np.zeros(CFG.vision_dv, np.float32),
                 "vision_std": np.ones(CFG.vision_dv, np.float32)},
                cpu,
            )
        report["models"][name]["model_latency_ms"] = latency_stats

    for directory in args.checkpoint:
        name = directory.name
        stats = load_stats(directory)
        model = load_fusion(directory, evaluation_device)
        model_report = {"checkpoint": str(directory.resolve()), "modalities": {}}
        for mode in ("both", "imu_only", "camera_only"):
            probabilities, trust = predict_fusion(
                model, validation_windows, stats, evaluation_device, mode=mode
            )
            mode_report = evaluate_probabilities(labels, probabilities)
            mode_report["mean_sensor_trust"] = {
                "imu": float(trust[:, 0].mean()),
                "radar": float(trust[:, 1].mean()),
                "camera": float(trust[:, 2].mean()),
            }
            model_report["modalities"][mode] = mode_report
        model_report["synchronization_shift"] = {}
        for shift_ms in SYNC_SHIFTS_MS:
            probabilities, _ = predict_fusion(
                model,
                validation_windows,
                stats,
                evaluation_device,
                mode="both",
                shift_ms=shift_ms,
            )
            summary = evaluate_probabilities(labels, probabilities)
            model_report["synchronization_shift"][str(shift_ms)] = {
                "classification": summary["classification"],
                "alert_at_0_60": summary["alert_at_0_60"],
            }
        model_report["continuous_background"] = background_report(
            model,
            background,
            stats,
            evaluation_device,
            negative_hours,
        )
        model_report["recorded_hardware_nonfall_replay"] = unlabeled_nonfall_replay_report(
            model, hardware_windows, stats, evaluation_device
        )
        latency_stats = {
            str(evaluation_device): benchmark_fusion(
                model, validation_windows[0], stats, evaluation_device
            )
        }
        if evaluation_device.type == "cuda":
            cpu = torch.device("cpu")
            cpu_model = load_fusion(directory, cpu)
            latency_stats["cpu"] = benchmark_fusion(
                cpu_model, validation_windows[0], stats, cpu
            )
        model_report["model_latency_ms"] = latency_stats
        report["models"][name] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved Step 8 report -> {args.output}")
    for name, model_report in report["models"].items():
        labelled = model_report.get("alert_at_0_60")
        if labelled is None:
            labelled = model_report["modalities"]["both"]["alert_at_0_60"]
        background_alert = model_report["continuous_background"]["alert"]
        print(
            f"{name}: fall_tp={labelled['true_positive']}, "
            f"fall_fp={labelled['false_positive']}, "
            f"fall_fn={labelled['false_negative']}, "
            f"background_false_alerts={background_alert['false_positive']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
