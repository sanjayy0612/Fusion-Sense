"""Stage 2 — train the CROSS-MODAL ATTENTION on paired data.

V1 expects pretrained radar + IMU encoders when available and uses open-source
camera pose features rather than a separately trained raw camera model. A vision
pose-head checkpoint is optional; if absent, the pose branch trains with fusion.

  python scripts/train_fusion.py              # real paired data + available encoders
  python scripts/train_fusion.py --sim        # simulator fallback (smoke test)
  python scripts/train_fusion.py --no-freeze  # fine-tune pretrained encoders too

Encoders are loaded from checkpoints/enc_{imu,radar,vis}.pt if present.
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fusionsense.config import CFG
from fusionsense.models.fusion import FusionSense
from fusionsense.data.registry import get_paired_windows
from fusionsense.data.splitting import split_paired_windows
from fusionsense.train.loop import train
from fusionsense.train.metrics import robustness_report, print_robustness
from fusionsense.device import get_device

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sim", action="store_true", help="simulator smoke test only")
parser.add_argument(
    "--no-freeze",
    action="store_true",
    help="fine-tune pretrained encoders instead of freezing them",
)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = get_device()
print("device:", device)

windows = get_paired_windows(allow_sim_fallback=args.sim)
train_w, val_w, split_description = split_paired_windows(
    windows, val_fraction=args.val_fraction, seed=args.seed
)
print(
    f"paired windows: {len(windows)} (train {len(train_w)}, val {len(val_w)}; "
    f"{split_description})"
)

# build model, plug in pretrained encoders that exist
model = FusionSense(CFG)
ck = "checkpoints"
paths = {m: os.path.join(ck, f) for m, f in
         [("imu", "enc_imu.pt"), ("radar", "enc_radar.pt"), ("vision", "enc_vis.pt")]}
present = {m: p for m, p in paths.items() if os.path.exists(p)}
if present:
    print("loading pretrained encoders:", list(present))
    model.load_pretrained_encoders(**present)
    if not args.no_freeze:
        model.freeze_encoders(
            imu="imu" in present,
            radar="radar" in present,
            vision="vision" in present,
        )
        print("froze:", list(present))
else:
    print("no pretrained encoders found — training end-to-end from scratch")

trainable_groups = []
for name, module in [
    ("IMU temporal encoder", model.enc_imu),
    ("pose temporal encoder", model.enc_vis),
    ("cross-modal Transformer", model.attn),
    ("health-conditioned pooling", model.pool_score),
    ("activity classifier", model.head),
]:
    if any(parameter.requires_grad for parameter in module.parameters()):
        trainable_groups.append(name)
print("training:", ", ".join(trainable_groups))

model = train(
    train_w,
    val_w,
    epochs=args.epochs,
    batch_size=args.batch_size,
    device=device,
    model=model,
)

print("\n=== ROBUSTNESS UNDER MODALITY DROPOUT ===")
print_robustness(robustness_report(model, val_w, device))

os.makedirs(ck, exist_ok=True)
torch.save(model.state_dict(), os.path.join(ck, "fusionsense.pt"))
print("\nsaved checkpoints/fusionsense.pt")
