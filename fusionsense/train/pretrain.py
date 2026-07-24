"""Pretrain ONE modality encoder on real single-modality data, then save just
the encoder weights (the head is discarded). Produces checkpoints/enc_<mod>.pt
for the fusion stage to load.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..config import CFG
from ..models.encoders import EncoderClassifier


def pretrain_encoder(X, y, n_classes, in_ch, out_path,
                     d=None, epochs=20, batch_size=128, lr=1e-3, device=None):
    """X: (N,T,in_ch) float32,  y: (N,) int.  Saves encoder to out_path."""
    from ..device import get_device
    device = device or get_device()
    d = d or CFG.d_model

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.long)
    n_val = max(1, int(0.2 * len(X)))
    perm = torch.randperm(len(X))
    tr, va = perm[n_val:], perm[:n_val]
    tl = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(X[va], y[va]), batch_size=batch_size)

    model = EncoderClassifier(in_ch, d, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for xb, yb in vl:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); tot += len(yb)
        print(f"  epoch {ep:2d} | val acc {correct/tot:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_encoder(out_path)
    print(f"saved encoder -> {out_path}")
    return out_path
