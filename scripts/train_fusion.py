"""Stage 2 — train the CROSS-MODAL ATTENTION on paired data.

V1 expects pretrained radar + IMU encoders when available and uses open-source
camera pose features rather than a separately trained raw camera model. A vision
pose-head checkpoint is optional; if absent, the pose branch trains with fusion.

  python scripts/train_fusion.py              # real paired data + available encoders
  python scripts/train_fusion.py --sim        # simulator fallback (smoke test)
  python scripts/train_fusion.py --no-freeze  # fine-tune encoders too

Encoders are loaded from checkpoints/enc_{imu,radar,vis}.pt if present.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fusionsense.config import CFG
from fusionsense.models.fusion import FusionSense
from fusionsense.data.registry import get_paired_windows
from fusionsense.train.loop import train
from fusionsense.train.metrics import robustness_report, print_robustness
from fusionsense.device import get_device

sim = "--sim" in sys.argv
freeze = "--no-freeze" not in sys.argv
device = get_device()
print("device:", device)

windows = get_paired_windows(allow_sim_fallback=sim)
n = len(windows); k = int(0.8 * n)
train_w, val_w = windows[:k], windows[k:]
print(f"paired windows: {n} (train {len(train_w)}, val {len(val_w)})")

# build model, plug in pretrained encoders that exist
model = FusionSense(CFG)
ck = "checkpoints"
paths = {m: os.path.join(ck, f) for m, f in
         [("imu", "enc_imu.pt"), ("radar", "enc_radar.pt"), ("vision", "enc_vis.pt")]}
present = {m: p for m, p in paths.items() if os.path.exists(p)}
if present:
    print("loading pretrained encoders:", list(present))
    model.load_pretrained_encoders(**present)
    if freeze:
        model.freeze_encoders(**{m: True for m in present})
        print("froze:", list(present))
else:
    print("no pretrained encoders found — training end-to-end from scratch")

model = train(train_w, val_w, epochs=15, device=device, model=model)

print("\n=== ROBUSTNESS UNDER MODALITY DROPOUT ===")
print_robustness(robustness_report(model, val_w, device))

os.makedirs(ck, exist_ok=True)
torch.save(model.state_dict(), os.path.join(ck, "fusionsense.pt"))
print("\nsaved checkpoints/fusionsense.pt")
