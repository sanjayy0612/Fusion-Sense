"""Run FusionSense inference on a mentor replay or an ESP32 recording."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fusionsense.contract import ACTIVITIES, LABEL2ID
from fusionsense.data.cmhad_loader import DEFAULT_CACHE, load_cmhad_windows
from fusionsense.data.imu_units import (
    CANONICAL_IMU_UNITS,
    live_conversion_metadata,
    validate_canonical_imu,
)
from fusionsense.data.recorded_session import build_recorded_session_windows
from fusionsense.data.splitting import split_paired_windows
from fusionsense.inference import FallInferenceEngine, classification_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "cmhad_step7_bimodal_dropout_fixed"
)
DEFAULT_MENTOR_OUTPUT = REPO_ROOT / "dashboard" / "mentor_demo.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stamp_event(event: dict, index: int, mode: str, truth: str | None = None) -> dict:
    result = deepcopy(event)
    result["event_id"] = f"{mode}-{index + 1:03d}"
    result["sequence"] = index + 1
    result["source_mode"] = mode
    result["ground_truth"] = truth
    result["event_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def build_mentor_demo(engine: FallInferenceEngine, cache_path: Path) -> dict:
    windows = load_cmhad_windows(cache_path)
    _, validation, split_description = split_paired_windows(windows, seed=42)
    predictions = [engine.predict(window) for window in validation]
    truth = [int(window.label) for window in validation]
    predicted = [int(result["predicted_class_id"]) for result in predictions]
    metrics = classification_metrics(truth, predicted)

    candidates: dict[int, list[int]] = {}
    for class_id in range(len(ACTIVITIES)):
        matching = [
            index
            for index, (actual, guess) in enumerate(zip(truth, predicted))
            if actual == class_id and guess == class_id
        ]
        if not matching:
            matching = [index for index, actual in enumerate(truth) if actual == class_id]
        candidates[class_id] = sorted(
            matching,
            key=lambda index: predictions[index]["class_probabilities"][ACTIVITIES[class_id]],
            reverse=True,
        )

    # Independent, held-out labeled examples. Non-falls lead into the clearest
    # stand-to-fall example so the dashboard ends in a visible alert state.
    playlist_classes = [0, 1, 2, 5, 6]
    events = []
    for class_id in playlist_classes:
        chosen = candidates[class_id][0]
        event = _stamp_event(
            predictions[chosen],
            len(events),
            "held_out_cmhad_replay",
            truth=ACTIVITIES[truth[chosen]],
        )
        event["dataset_subject"] = validation[chosen].subject_id
        event["dataset_recording"] = validation[chosen].recording_id
        event["demo_disclosure"] = (
            "Labeled held-out C-MHAD replay; this event is not a live ESP32 detection."
        )
        events.append(event)

    all_imu = np.concatenate([window.imu for window in validation], axis=0)
    dataset_unit_validation = validate_canonical_imu(all_imu).as_dict()
    fall_event = events[-1]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if metrics["fall_recall"] > 0 else "FAIL",
        "mode": "mentor_demo",
        "title": "FusionSense Fall Detection",
        "disclosure": (
            "Dashboard events are checkpoint predictions on labeled held-out C-MHAD "
            "windows. They demonstrate the completed laptop pipeline while ESP32 wiring "
            "is being finalized."
        ),
        "model": {
            "name": engine.model_name,
            "classes": ACTIVITIES,
            "checkpoint": str(engine.checkpoint_dir.resolve()),
            "training_subjects": engine.labels_metadata.get("subjects", []),
            "evaluation_split": split_description,
            "scope_note": "Four-subject research pilot; not a clinical or safety-certified device.",
        },
        "input_contract": {
            "window_seconds": 2.0,
            "imu": {"shape": [100, 6], "units": list(CANONICAL_IMU_UNITS)},
            "camera_pose": {"shape": [20, 99], "rate_fps": 10},
            "radar": {"active": False, "note": "Not present in the C-MHAD pilot"},
            "live_esp32_conversion": live_conversion_metadata(),
            "dataset_unit_validation": dataset_unit_validation,
        },
        "evaluation": metrics,
        "alert_policy": {
            "class": "stand_to_fall",
            "threshold": engine.fall_threshold,
            "condition": "stand_to_fall probability >= threshold",
            "required_modalities": ["imu", "camera_pose"],
        },
        "events": events,
        "current_event": fall_event,
    }


def build_recorded_output(
    engine: FallInferenceEngine,
    session_dir: Path,
    *,
    extract_poses: bool,
) -> dict:
    session_windows, metadata = build_recorded_session_windows(
        session_dir, extract_poses=extract_poses
    )
    eligible_windows, withheld_windows = select_bimodal_windows(session_windows)
    events = []
    for index, item in enumerate(eligible_windows):
        event = _stamp_event(engine.predict(item.window), index, "recorded_esp32")
        event["window_diagnostics"] = item.diagnostics
        event["demo_disclosure"] = (
            "Fused inference using both real ESP32 IMU and ESP32-CAM pose data."
        )
        events.append(event)
    alerts = [event for event in events if event["alert"]]
    capture_result = metadata.get("capture_validation", {}).get("result", "UNKNOWN")
    unavailable = sum(event["status"] == "degraded" for event in events)
    bimodal_checks = {
        "capture_validation_passed": capture_result == "PASS",
        "at_least_one_bimodal_window": bool(events),
        "both_sensors_valid_for_every_inference": all(
            event["input"]["valid_modalities"]["imu"]
            and event["input"]["valid_modalities"]["camera_pose"]
            for event in events
        ),
        "radar_disabled_for_every_inference": all(
            not event["input"]["valid_modalities"]["radar"] for event in events
        ),
        "positive_imu_trust_for_every_inference": all(
            event["sensor_trust"]["imu"] > 0.0 for event in events
        ),
        "positive_camera_trust_for_every_inference": all(
            event["sensor_trust"]["camera"] > 0.0 for event in events
        ),
        "zero_radar_trust_for_every_inference": all(
            event["sensor_trust"]["radar"] == 0.0 for event in events
        ),
        "no_sensor_unavailable_predictions": unavailable == 0,
    }
    pipeline_result = "PASS" if all(bimodal_checks.values()) else "DEGRADED"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": pipeline_result if events else "FAIL",
        "mode": "recorded_esp32",
        "title": "FusionSense Recorded ESP32 Replay",
        "disclosure": (
            "Every prediction fuses real synchronized ESP32 IMU and ESP32-CAM "
            "pose data. Windows missing either input are withheld, not predicted."
        ),
        "model": {
            "name": engine.model_name,
            "checkpoint": str(engine.checkpoint_dir.resolve()),
            "classes": ACTIVITIES,
            "scope_note": "Four-subject research pilot; not a clinical or safety-certified device.",
        },
        "input_contract": {
            "window_seconds": 2.0,
            "imu": {"shape": [100, 6], "units": list(CANONICAL_IMU_UNITS)},
            "camera_pose": {"shape": [20, 99], "rate_fps": 10},
            "fusion_policy": "valid IMU and valid camera pose required",
            "radar": {"active": False, "note": "Not an input to this project"},
            "live_esp32_conversion": live_conversion_metadata(),
        },
        "session": metadata,
        "summary": {
            "assembled_windows": len(session_windows),
            "bimodal_inference_windows": len(events),
            "withheld_non_bimodal_windows": len(withheld_windows),
            "alerts": len(alerts),
            "sensor_unavailable_windows": unavailable,
            "capture_validation_result": capture_result,
        },
        "bimodal_checks": bimodal_checks,
        "withheld_windows": withheld_windows,
        "alert_policy": {
            "class": ACTIVITIES[LABEL2ID["stand_to_fall"]],
            "threshold": engine.fall_threshold,
            "required_modalities": ["imu", "camera_pose"],
        },
        "events": events,
        "current_event": events[-1] if events else None,
    }


def select_bimodal_windows(session_windows: list) -> tuple[list, list[dict]]:
    """Keep only IMU+camera windows and document every withheld window."""
    eligible = []
    withheld = []
    for item in session_windows:
        reasons = []
        if not item.window.imu_valid:
            reasons.append("imu_invalid")
        if not item.window.vision_valid:
            reasons.append("camera_pose_invalid")
        if item.window.radar_valid:
            reasons.append("radar_must_be_disabled")
        if reasons:
            withheld.append(
                {
                    "window_start_s": item.diagnostics.get("window_start_s"),
                    "reasons": reasons,
                }
            )
        else:
            eligible.append(item)
    return eligible, withheld


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mentor", "recorded"), default="mentor")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--fall-threshold", type=float, default=0.60)
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    engine = FallInferenceEngine(
        args.checkpoint_dir,
        fall_threshold=args.fall_threshold,
        device="cpu",
        require_bimodal=True,
    )
    if args.mode == "mentor":
        output = args.output or DEFAULT_MENTOR_OUTPUT
        payload = build_mentor_demo(engine, args.cache)
    else:
        if args.session_dir is None:
            parser.error("--session-dir is required for --mode recorded")
        output = args.output or (REPO_ROOT / "dashboard" / "session_output.json")
        payload = build_recorded_output(
            engine, args.session_dir, extract_poses=not args.skip_pose
        )
    _write_json(output, payload)
    print(json.dumps({
        "result": payload["result"],
        "mode": payload["mode"],
        "output": str(output.resolve()),
        "events": len(payload.get("events", [])),
        "alerts": sum(event.get("alert", False) for event in payload.get("events", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
