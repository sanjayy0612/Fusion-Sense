"""Metrics: accuracy, macro-F1, per-class fall recall, and the dropout
robustness study (the headline experiment)."""
from __future__ import annotations

import numpy as np
import torch

from ..contract import LABEL2ID, ACTIVITIES


def summarize(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    f1s = []
    for c in range(len(ACTIVITIES)):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = float(np.mean(f1s))
    fall = LABEL2ID["falling"]
    fall_recall = float(((y_pred == fall) & (y_true == fall)).sum()
                        / ((y_true == fall).sum() + 1e-9))
    return float(acc), macro_f1, fall_recall


def confusion(y_true, y_pred):
    n = len(ACTIVITIES)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


@torch.no_grad()
def robustness_report(model, windows, device):
    """Evaluate accuracy when each modality is force-dropped at inference.
    This table is the core result of the paper's robustness claim."""
    from ..data.dataset import FusionDataset
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(FusionDataset(windows, train=False, dropout_p=0.0),
                        batch_size=128)

    configs = {
        "all sensors": [1, 1, 1],
        "no vision (dark)": [1, 1, 0],
        "no radar": [1, 0, 1],
        "no imu": [0, 1, 1],
        "imu only": [1, 0, 0],
        "radar only": [0, 1, 0],
        "vision only": [0, 0, 1],
    }
    rows = {}
    for name, mask in configs.items():
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        ys, ps = [], []
        for imu, radar, vision, valid, health, y in loader:
            imu, radar, vision = imu.to(device), radar.to(device), vision.to(device)
            health = health.to(device)
            v = valid.to(device) & mask_t.unsqueeze(0)
            h = health * mask_t.float().unsqueeze(0)
            keep = v.any(1)                      # skip samples with nothing left
            if keep.sum() == 0:
                continue
            pred = model(imu, radar, vision, v, h).argmax(1)
            ys.append(y[keep.cpu()].numpy()); ps.append(pred[keep].cpu().numpy())
        acc, f1, fall = summarize(np.concatenate(ys), np.concatenate(ps))
        rows[name] = dict(acc=acc, macro_f1=f1, fall_recall=fall)
    return rows


def print_robustness(rows):
    print(f"\n{'configuration':<20} {'acc':>6} {'macroF1':>8} {'fall-rec':>9}")
    print("-" * 46)
    for name, r in rows.items():
        print(f"{name:<20} {r['acc']:>6.3f} {r['macro_f1']:>8.3f} {r['fall_recall']:>9.3f}")
