"""Stage 1 — pretrain the radar encoder on real mmWave data (RadHAR).

  python scripts/pretrain_radar.py         # real data (data/raw/radhar/)
  python scripts/pretrain_radar.py --sim   # simulator fallback (smoke test)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.config import CFG
from fusionsense.data.registry import get_pretrain_data
from fusionsense.train.pretrain import pretrain_encoder

sim = "--sim" in sys.argv
X, y, n_classes = get_pretrain_data("radar", allow_sim_fallback=sim)
print(f"radar pretrain: X={X.shape}, classes={n_classes}")
pretrain_encoder(X, y, n_classes, in_ch=CFG.radar_k, out_path="checkpoints/enc_radar.pt")
