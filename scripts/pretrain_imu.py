"""Stage 1 — pretrain the IMU encoder on real data (SisFall / UCI-HAR).

  python scripts/pretrain_imu.py           # real data (data/raw/sisfall/)
  python scripts/pretrain_imu.py --sim     # simulator fallback (smoke test)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.config import CFG
from fusionsense.data.registry import get_pretrain_data
from fusionsense.train.pretrain import pretrain_encoder

sim = "--sim" in sys.argv
X, y, n_classes = get_pretrain_data("imu", allow_sim_fallback=sim)
print(f"IMU pretrain: X={X.shape}, classes={n_classes}")
pretrain_encoder(X, y, n_classes, in_ch=CFG.imu_ch, out_path="checkpoints/enc_imu.pt")
