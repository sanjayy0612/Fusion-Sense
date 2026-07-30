"""Numpy-only tests for the data pipeline (no torch needed).

Run:  python tests/test_pipeline.py
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionsense.config import CFG
from fusionsense.contract import ACTIVITIES
from fusionsense.data.simulator import (
    sample_window,
    make_dataset,
    record_simulation_output,
    load_simulation_output,
)


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


if __name__ == "__main__":
    test_shapes()
    test_no_nans()
    test_never_all_dropped()
    test_health_in_range()
    test_balanced_dataset()
    test_record_and_replay_roundtrip()
    print("\nALL PIPELINE TESTS PASSED")
