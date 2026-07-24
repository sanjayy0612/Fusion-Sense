"""Pick the best available training device across machines:
  - Nvidia GPU (your RTX 4060)      -> 'cuda'
  - Apple Silicon (M2 MacBook)      -> 'mps'
  - otherwise                       -> 'cpu'
"""
from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
