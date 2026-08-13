"""Numpy-only tests for the data pipeline (no torch needed).

Run:  python tests/test_pipeline.py
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.contract import ACTIVITIES, FusionWindow
from fusionsense.data.simulator import (
    sample_window,
    make_dataset,
    record_simulation_output,
    load_simulation_output,
)
from fusionsense.data.paired_loader import _imu_windows
from fusionsense.data.splitting import split_paired_windows


def test_shapes():
    rng = np.random.default_rng(0)
    w = sample_window("walking", rng, degrade=False)
    assert w.imu.shape == (CFG.t_imu, CFG.imu_ch), w.imu.shape
    assert w.radar.shape == (CFG.t_radar, CFG.radar_k), w.radar.shape
    assert w.vision.shape == (CFG.t_vis, CFG.vision_dv), w.vision.shape
    assert w.valid_vector().shape == (3,)
    assert w.health_vector().shape == (3,)
    print("PASS test_shapes")


def test_no_nans():
    rng = np.random.default_rng(2)
    for act in ACTIVITIES:
        w = sample_window(act, rng, degrade=True)
        for arr in (w.imu, w.radar, w.vision):
            assert np.isfinite(arr).all(), act
    print("PASS test_no_nans")


def test_never_all_dropped():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        w = sample_window(rng.choice(ACTIVITIES), rng, degrade=True)
        assert w.valid_vector().any(), "all modalities dropped — contract violated"
    print("PASS test_never_all_dropped")


def test_health_in_range():
    rng = np.random.default_rng(4)
    for _ in range(1000):
        w = sample_window(rng.choice(ACTIVITIES), rng, degrade=True)
        h = w.health_vector()
        assert (h >= 0).all() and (h <= 1).all(), h
    print("PASS test_health_in_range")


def test_balanced_dataset():
    ds = make_dataset(n_per_class=50, seed=5)
    assert len(ds) == 50 * len(ACTIVITIES)
    counts = np.bincount([w.label for w in ds], minlength=len(ACTIVITIES))
    assert (counts == 50).all(), counts
    print("PASS test_balanced_dataset")


def test_degraded_dataset_never_all_dropped():
    ds = make_dataset(n_per_class=400, seed=0, degrade=True)
    assert all(w.valid_vector().any() for w in ds), "degraded dataset has all-invalid window"
    print("PASS test_degraded_dataset_never_all_dropped")


def test_record_and_replay_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sim_output.npz")
        windows = record_simulation_output(out_path, n_per_class=2, seed=7, degrade=False)
        replayed = load_simulation_output(out_path)
        assert len(replayed) == len(windows)
        assert replayed[0].label == windows[0].label
        assert replayed[0].imu.shape == windows[0].imu.shape
        assert np.allclose(replayed[0].imu, windows[0].imu)
        print("PASS test_record_and_replay_roundtrip")


def test_collector_imu_csv_is_read_without_timestamp_channel():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "imu.csv")
        with open(path, "w") as handle:
            handle.write("t_ms,ax,ay,az,gx,gy,gz\n")
            for i in range(150):
                handle.write(f"{i * 20},1,2,3,4,5,6\n")
        windows = _imu_windows(path, CFG)
        assert windows and windows[0].shape == (CFG.t_imu, CFG.imu_ch)
        assert np.allclose(windows[0][0], [1, 2, 3, 4, 5, 6])
        print("PASS test_collector_imu_csv_is_read_without_timestamp_channel")


def test_paired_split_keeps_recordings_together():
    windows = []
    for label in range(len(ACTIVITIES)):
        for trial in range(3):
            for offset in range(2):
                windows.append(FusionWindow(
                    t_start=float(offset),
                    imu=np.zeros((CFG.t_imu, CFG.imu_ch), np.float32),
                    radar=np.zeros((CFG.t_radar, CFG.radar_k), np.float32),
                    vision=np.zeros((CFG.t_vis, CFG.vision_dv), np.float32),
                    radar_valid=False,
                    label=label,
                    subject_id="s01",
                    recording_id=f"s01/{label}/{trial}",
                ))
    train, val, description = split_paired_windows(windows, seed=7)
    train_ids = {window.recording_id for window in train}
    val_ids = {window.recording_id for window in val}
    assert train_ids.isdisjoint(val_ids)
    assert {window.label for window in train} == set(range(len(ACTIVITIES)))
    assert {window.label for window in val} == set(range(len(ACTIVITIES)))
    assert description == "stratified by recording"
    print("PASS test_paired_split_keeps_recordings_together")


if __name__ == "__main__":
    test_shapes()
    test_no_nans()
    test_never_all_dropped()
    test_health_in_range()
    test_balanced_dataset()
    test_degraded_dataset_never_all_dropped()
    test_record_and_replay_roundtrip()
    test_collector_imu_csv_is_read_without_timestamp_channel()
    test_paired_split_keeps_recordings_together()
    print("\nALL PIPELINE TESTS PASSED")
