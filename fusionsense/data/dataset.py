"""Torch Dataset over FusionWindows + modality-dropout augmentation.

Modality dropout is what *teaches* the attention to reweight. Without it the
model leans on whichever modality is easiest and never learns robustness.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CFG


class FusionDataset(Dataset):
    def __init__(self, windows, train: bool = True, dropout_p: float = None):
        self.windows = windows
        self.train = train
        self.dropout_p = CFG.modality_dropout_p if dropout_p is None else dropout_p

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        imu = torch.from_numpy(w.imu).float()
        radar = torch.from_numpy(w.radar).float()
        vision = torch.from_numpy(w.vision).float()
        valid = torch.from_numpy(w.valid_vector())
        health = torch.from_numpy(w.health_vector())

        if self.train and self.dropout_p > 0:
            # randomly kill modalities, but never all three
            for m, (tensor,) in enumerate([(imu,), (radar,), (vision,)]):
                if valid.sum() > 1 and np.random.rand() < self.dropout_p:
                    tensor.zero_()
                    valid[m] = False
                    health[m] = 0.0

        label = torch.tensor(w.label, dtype=torch.long)
        return imu, radar, vision, valid, health, label


def make_loaders(train_windows, val_windows, batch_size=64):
    from torch.utils.data import DataLoader
    tr = FusionDataset(train_windows, train=True)
    va = FusionDataset(val_windows, train=False, dropout_p=0.0)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(va, batch_size=batch_size, shuffle=False),
    )
