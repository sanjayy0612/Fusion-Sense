"""FusionSense model: per-modality encoders -> merged-token self-attention
(cross-modal) -> sensor-health-conditioned masked pooling -> classifier.

Now supports PRETRAINED encoders: build the model, then call
load_pretrained_encoders(...) to plug in encoders trained separately on real
single-modality data, and optionally freeze them while the cross-modal
attention learns on paired data.

Design notes
------------
- Each modality is compressed to ONE token. Self-attention over the 3 tokens is
  our lightweight cross-modal mechanism.
- Graceful degradation: invalid modalities are masked out of attention AND of
  the pooling, so a dropped sensor contributes exactly zero.
- Sensor-health conditioning: [imu, radar, vision] health scalars bias the
  pooling weights (exported as an interpretable `trust` output).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, Config
from .encoders import ModalityEncoder, load_encoder_weights


class FusionSense(nn.Module):
    def __init__(self, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.enc_imu = ModalityEncoder(cfg.imu_ch, d)
        self.enc_radar = ModalityEncoder(cfg.radar_k, d)
        self.enc_vis = ModalityEncoder(cfg.vision_dv, d)

        self.mod_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
            batch_first=True, dropout=0.1,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.pool_score = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.n_classes))

    # ---- pretraining support ----
    def load_pretrained_encoders(self, imu=None, radar=None, vision=None, strict=True):
        """Load pretrained encoder weights (paths to .pt files). Any left None
        stay randomly initialized."""
        if imu:
            load_encoder_weights(self.enc_imu, imu, strict)
        if radar:
            load_encoder_weights(self.enc_radar, radar, strict)
        if vision:
            load_encoder_weights(self.enc_vis, vision, strict)
        return self

    def freeze_encoders(self, imu=True, radar=True, vision=True):
        """Freeze chosen encoders so only the cross-modal attention + head train."""
        for flag, enc in [(imu, self.enc_imu), (radar, self.enc_radar), (vision, self.enc_vis)]:
            for p in enc.parameters():
                p.requires_grad = not flag
        return self

    def forward(self, imu, radar, vision, valid, health=None, return_trust=False):
        toks = torch.stack(
            [self.enc_imu(imu), self.enc_radar(radar), self.enc_vis(vision)], dim=1
        )                                                  # (B, 3, d)
        toks = toks + self.mod_emb
        pad_mask = ~valid                                  # True => ignore in attn
        z = self.attn(toks, src_key_padding_mask=pad_mask)  # (B, 3, d)

        score = self.pool_score(z).squeeze(-1)             # (B, 3)
        if self.cfg.use_health_conditioning and health is not None:
            score = score + torch.log(health.clamp(min=1e-3))
        score = score.masked_fill(~valid, float("-inf"))
        trust = F.softmax(score, dim=1)                    # (B, 3) interpretable
        pooled = torch.einsum("bt,btd->bd", trust, z)      # (B, d)

        logits = self.head(pooled)
        if return_trust:
            return logits, trust
        return logits
