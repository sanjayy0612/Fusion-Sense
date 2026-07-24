"""Per-modality encoders — now standalone so each can be PRETRAINED separately
on its own real dataset, then loaded into the fusion model.

Two classes:
  ModalityEncoder    : (B, T, C) -> (B, d) single token. The reusable part.
  EncoderClassifier  : ModalityEncoder + a linear head, used ONLY during
                       single-modality pretraining. The head is thrown away;
                       the encoder weights are what we keep.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """1D-CNN + GRU -> one d-dim token summarizing a modality window."""

    def __init__(self, in_ch: int, d: int = 128):
        super().__init__()
        self.in_ch, self.d = in_ch, d
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, d, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, C)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)    # (B, T, d)
        _, hn = self.gru(h)
        return hn[-1]                                       # (B, d)


class EncoderClassifier(nn.Module):
    """Encoder + linear head for single-modality pretraining."""

    def __init__(self, in_ch: int, d: int, n_classes: int):
        super().__init__()
        self.encoder = ModalityEncoder(in_ch, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        return self.head(self.encoder(x))

    def save_encoder(self, path: str):
        torch.save(self.encoder.state_dict(), path)


def load_encoder_weights(encoder: ModalityEncoder, path: str, strict: bool = True):
    """Load pretrained weights into a ModalityEncoder (dims must match)."""
    sd = torch.load(path, map_location="cpu")
    encoder.load_state_dict(sd, strict=strict)
    return encoder
