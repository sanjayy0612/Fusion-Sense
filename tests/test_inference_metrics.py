import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.contract import FusionWindow
from fusionsense.inference import FallInferenceEngine, classification_metrics
from scripts.run_fall_pipeline import select_bimodal_windows


def test_classification_metrics_reports_fall_recall():
    truth = [0, 1, 2, 3, 4, 5, 6, 6]
    predicted = [0, 1, 2, 3, 4, 5, 6, 0]
    result = classification_metrics(truth, predicted)
    assert result["samples"] == 8
    assert result["accuracy"] == 0.875
    assert result["fall_recall"] == 0.5


def test_recorded_inference_selects_only_imu_plus_camera_windows():
    def item(imu, camera, radar, start):
        window = SimpleNamespace(
            imu_valid=imu, vision_valid=camera, radar_valid=radar
        )
        return SimpleNamespace(
            window=window, diagnostics={"window_start_s": start}
        )

    windows = [
        item(True, True, False, 0.0),
        item(True, False, False, 1.0),
        item(False, True, False, 2.0),
        item(True, True, True, 3.0),
    ]
    eligible, withheld = select_bimodal_windows(windows)

    assert len(eligible) == 1
    assert len(withheld) == 3
    assert withheld[0]["reasons"] == ["camera_pose_invalid"]
    assert withheld[1]["reasons"] == ["imu_invalid"]
    assert withheld[2]["reasons"] == ["radar_must_be_disabled"]


def test_bimodal_engine_refuses_an_imu_only_window_before_model_execution():
    engine = object.__new__(FallInferenceEngine)
    engine.cfg = CFG
    engine.checkpoint_dir = Path("checkpoint")
    engine.fall_threshold = 0.6
    engine.require_bimodal = True
    engine.model_name = "test fusion"
    window = FusionWindow(
        t_start=0.0,
        imu=np.zeros((CFG.t_imu, CFG.imu_ch), dtype=np.float32),
        radar=np.zeros((CFG.t_radar, CFG.radar_k), dtype=np.float32),
        vision=np.zeros((CFG.t_vis, CFG.vision_dv), dtype=np.float32),
        imu_valid=True,
        radar_valid=False,
        vision_valid=False,
    )

    result = engine.predict(window)

    assert result["status"] == "degraded"
    assert result["predicted_label"] is None
    assert not result["alert"]


if __name__ == "__main__":
    test_classification_metrics_reports_fall_recall()
    test_recorded_inference_selects_only_imu_plus_camera_windows()
    test_bimodal_engine_refuses_an_imu_only_window_before_model_execution()
    print("ALL INFERENCE METRIC TESTS PASSED")
