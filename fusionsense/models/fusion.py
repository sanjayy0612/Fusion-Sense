"""FusionSense model: per-modality encoders -> merged-token self-attention
(cross-modal) -> sensor-health-conditioned masked pooling -> classifier.

Design notes
------------
- Each modality is compressed to ONE token. Self-attention over the 3 tokens is
  our (lightweight) cross-modal mechanism. Attention over 3 tokens is nearly
  free; the cost is in the small CNN encoders.
- Graceful degradation: invalid modalities are masked out of attention AND out
  of the final pooling, so a dropped sensor contributes exactly zero.
- Sensor-health conditioning (the unique angle): the [imu, radar, vision] health
  scalars bias the pooling weights, so the model trusts physically-healthy
  sensors more. Toggle with CFG.use_health_conditioning. Exposed as an
  interpretable `trust` output for the dashboard.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG


class ModalityEncoder(nn.Module):
    """(B, T, C) -> (B, d) single token via 1D-CNN + GRU."""

    def __init__(self, in_ch: int, d: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, d, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d)
        _, hn = self.gru(h)
        return hn[-1]                                      # (B, d)


class FusionSense(nn.Module):
    def __init__(self, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.enc_imu = ModalityEncoder(cfg.imu_ch, d)
        self.enc_radar = ModalityEncoder(cfg.radar_k, d)
        self.enc_vis = ModalityEncoder(cfg.vision_dv, d)

        self.mod_emb = nn.Parameter(torch.randn(3, d) * 0.02)   # modality-type emb
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
            batch_first=True, dropout=0.1,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        # pooling: learn a score per token; combine with health prior + validity.
        self.pool_score = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.n_classes))

    def forward(self, imu, radar, vision, valid, health=None, return_trust=False):
        """
        imu    (B, T_imu, 6)   radar (B, T_radar, K)   vision (B, T_vis, D_v)
        valid  (B, 3) bool     health (B, 3) float in [0,1] or None
        """
        toks = torch.stack(
            [self.enc_imu(imu), self.enc_radar(radar), self.enc_vis(vision)], dim=1
        )                                                  # (B, 3, d)
        toks = toks + self.mod_emb

        pad_mask = ~valid                                  # True => ignore in attn
        z = self.attn(toks, src_key_padding_mask=pad_mask)  # (B, 3, d)

        # --- sensor-health-conditioned masked pooling ---
        score = self.pool_score(z).squeeze(-1)             # (B, 3)
        if self.cfg.use_health_conditioning and health is not None:
            # push trust toward healthy sensors (log so it's an additive prior)
            score = score + torch.log(health.clamp(min=1e-3))
        score = score.masked_fill(~valid, float("-inf"))   # dead sensors -> 0 weight
        trust = F.softmax(score, dim=1)                    # (B, 3) interpretable
        pooled = torch.einsum("bt,btd->bd", trust, z)      # (B, d)

        logits = self.head(pooled)
        if return_trust:
            return logits, trust
        return logits


# keep type hint import lazy to avoid circulars
from ..config import Config  # noqa: E402
