"""Training + evaluation loop for FusionSense."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import CFG
from ..contract import ACTIVITIES, LABEL2ID
from ..models.fusion import FusionSense
from ..data.dataset import make_loaders
from .metrics import summarize, robustness_report


def _class_weights(windows, device):
    counts = np.zeros(CFG.n_classes)
    for w in windows:
        counts[w.label] += 1
    inv = counts.sum() / (counts + 1e-6)
    inv = inv / inv.mean()
    # extra emphasis on the critical 'falling' class
    inv[LABEL2ID["falling"]] *= 1.5
    return torch.tensor(inv, dtype=torch.float32, device=device)


def train(train_windows, val_windows, epochs=15, batch_size=64, lr=1e-3,
          device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionSense(CFG).to(device)
    tr_loader, va_loader = make_loaders(train_windows, val_windows, batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=_class_weights(train_windows, device))

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for imu, radar, vision, valid, health, y in tr_loader:
            imu, radar, vision = imu.to(device), radar.to(device), vision.to(device)
            valid, health, y = valid.to(device), health.to(device), y.to(device)
            logits = model(imu, radar, vision, valid, health)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.size(0)
        acc, f1, fall_recall = evaluate(model, va_loader, device)
        print(f"epoch {ep:2d} | loss {tot/len(tr_loader.dataset):.3f} "
              f"| val acc {acc:.3f} | macroF1 {f1:.3f} | fall-recall {fall_recall:.3f}")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imu, radar, vision, valid, health, y in loader:
        imu, radar, vision = imu.to(device), radar.to(device), vision.to(device)
        valid, health = valid.to(device), health.to(device)
        pred = model(imu, radar, vision, valid, health).argmax(1)
        ys.append(y.numpy()); ps.append(pred.cpu().numpy())
    return summarize(np.concatenate(ys), np.concatenate(ps))
