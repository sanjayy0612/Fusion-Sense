"""Pretrain ONE modality encoder on real single-modality data, then save just
the encoder weights (the head is discarded). Produces checkpoints/enc_<mod>.pt
for the fusion stage to load.
"""
from __future__ import annotations

import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..config import CFG
from ..models.encoders import EncoderClassifier


def _accuracy_macro_f1(y_true, y_pred, n_classes):
    accuracy = float((y_true == y_pred).mean())
    f1s = []
    for class_id in range(n_classes):
        tp = ((y_pred == class_id) & (y_true == class_id)).sum()
        fp = ((y_pred == class_id) & (y_true != class_id)).sum()
        fn = ((y_pred != class_id) & (y_true == class_id)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1s.append(2 * precision * recall / (precision + recall + 1e-9))
    return accuracy, float(np.mean(f1s))


def pretrain_encoder(X, y, n_classes, in_ch, out_path,
                     d=None, epochs=20, batch_size=128, lr=1e-3, device=None,
                     X_val=None, y_val=None, seed=42):
    """X: (N,T,in_ch) float32,  y: (N,) int.  Saves encoder to out_path."""
    from ..device import get_device
    device = device or get_device()
    d = d or CFG.d_model

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.long)
    if X_val is None or y_val is None:
        n_val = max(1, int(0.2 * len(X)))
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator=generator)
        tr, va = perm[n_val:], perm[:n_val]
        train_ds = TensorDataset(X[tr], y[tr])
        val_ds = TensorDataset(X[va], y[va])
    else:
        train_ds = TensorDataset(X, y)
        val_ds = TensorDataset(
            torch.tensor(np.asarray(X_val), dtype=torch.float32),
            torch.tensor(np.asarray(y_val), dtype=torch.long),
        )
    loader_generator = torch.Generator().manual_seed(seed)
    tl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator
    )
    vl = DataLoader(val_ds, batch_size=batch_size)

    model = EncoderClassifier(in_ch, d, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = torch.bincount(y, minlength=n_classes).float()
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Encoder training split is missing classes {missing}")
    class_weights = counts.sum() / (n_classes * counts)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_score = -1.0
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for xb, yb in vl:
                pred = model(xb.to(device)).argmax(1).cpu()
                ys.append(yb.numpy())
                ps.append(pred.numpy())
        if not ys:
            raise RuntimeError("Encoder validation split is empty")
        val_acc, val_f1 = _accuracy_macro_f1(
            np.concatenate(ys), np.concatenate(ps), n_classes
        )
        if val_f1 > best_score:
            best_score = val_f1
            best_acc = val_acc
            best_epoch = ep
            best_state = deepcopy(model.state_dict())
        print(f"  epoch {ep:2d} | val acc {val_acc:.3f} | macroF1 {val_f1:.3f}")

    if best_state is None:
        raise RuntimeError("Encoder training produced no validation result")
    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_encoder(out_path)
    print(
        f"saved best encoder -> {out_path} "
        f"(epoch {best_epoch}, val acc {best_acc:.3f}, macroF1 {best_score:.3f})"
    )
    return out_path
