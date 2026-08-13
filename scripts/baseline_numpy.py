"""NumPy-only baseline classifier for FusionSense simulation results.

Why this exists: it produces REAL, reproducible numbers (accuracy, confusion
matrix, robustness-under-dropout) with zero heavy dependencies, so the project
has empirical simulation results even before the full PyTorch attention model is
trained on a GPU. It is a *feature + softmax-regression* baseline, NOT the
attention model — but it validates the two core claims:
  (1) the simulated multimodal data is separable, and
  (2) fusing modalities degrades gracefully when a sensor is dropped.

Run:  python scripts/baseline_numpy.py   ->  prints tables, writes figures/*.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusionsense.contract import ACTIVITIES, LABEL2ID
from fusionsense.data.simulator import make_dataset

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)


# ---------- feature extraction (per modality, so we can drop them) ----------
def feats_imu(w):
    x = w.imu
    return np.concatenate([x.mean(0), x.std(0), np.abs(x).max(0)])      # 18

def feats_radar(w):
    x = w.radar
    return np.concatenate([x.mean(0), x.std(0), x.max(0)])              # 24

def feats_vision(w):
    x = w.vision
    return np.concatenate([x.mean(0), x.std(0)])                        # 64

IMU_N, RAD_N, VIS_N = 18, 24, 64


def features(w, drop=(1, 1, 1)):
    fi = feats_imu(w) * drop[0]
    fr = feats_radar(w) * drop[1]
    fv = feats_vision(w) * drop[2]
    return np.concatenate([fi, fr, fv]).astype(np.float64)


def build_matrix(windows, drop=(1, 1, 1)):
    X = np.stack([features(w, drop) for w in windows])
    y = np.array([w.label for w in windows])
    return X, y


# ---------- softmax regression (pure numpy) ----------
class SoftmaxReg:
    def __init__(self, n_feat, n_cls, lr=0.1, l2=1e-3):
        self.W = np.zeros((n_feat, n_cls)); self.b = np.zeros(n_cls)
        self.lr, self.l2 = lr, l2

    def _soft(self, Z):
        Z = Z - Z.max(1, keepdims=True)
        e = np.exp(Z); return e / e.sum(1, keepdims=True)

    def fit(self, X, y, epochs=300):
        n, k = X.shape[0], self.W.shape[1]
        Y = np.eye(k)[y]
        for _ in range(epochs):
            P = self._soft(X @ self.W + self.b)
            gW = X.T @ (P - Y) / n + self.l2 * self.W
            gb = (P - Y).mean(0)
            self.W -= self.lr * gW; self.b -= self.lr * gb

    def predict(self, X):
        return (X @ self.W + self.b).argmax(1)


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def scores(y, p):
    acc = (y == p).mean()
    f1s = []
    for c in range(len(ACTIVITIES)):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f1s.append(2 * pr * rc / (pr + rc + 1e-9))
    fall = LABEL2ID["stand_to_fall"]
    fr = ((p == fall) & (y == fall)).sum() / ((y == fall).sum() + 1e-9)
    return acc, float(np.mean(f1s)), float(fr)


def main():
    train = make_dataset(n_per_class=800, seed=0, degrade=True)
    test = make_dataset(n_per_class=200, seed=1, degrade=True)

    Xtr, ytr = build_matrix(train)
    Xte, yte = build_matrix(test)
    Xtr_s, Xte_s, mu, sd = standardize(Xtr, Xte)

    clf = SoftmaxReg(Xtr.shape[1], len(ACTIVITIES), lr=0.5, l2=1e-3)
    clf.fit(Xtr_s, ytr, epochs=400)

    p = clf.predict(Xte_s)
    acc, f1, fr = scores(yte, p)
    print("=== BASELINE (all sensors) ===")
    print(f"accuracy {acc:.3f} | macro-F1 {f1:.3f} | fall-recall {fr:.3f}")

    # confusion matrix
    n_classes = len(ACTIVITIES)
    C = np.zeros((n_classes, n_classes), int)
    for t, pp in zip(yte, p):
        C[t, pp] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(ACTIVITIES, rotation=40, ha="right"); ax.set_yticklabels(ACTIVITIES)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, C[i, j], ha="center", va="center",
                    color="white" if C[i, j] > C.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix (acc {acc:.2f})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{FIG}/confusion_matrix.png", dpi=120); plt.close(fig)

    # robustness: drop each modality at test time (retrain-free: zero features)
    print("\n=== ROBUSTNESS UNDER MODALITY DROPOUT ===")
    configs = {
        "all sensors": (1, 1, 1), "no vision (dark)": (1, 1, 0),
        "no radar": (1, 0, 1), "no imu": (0, 1, 1),
        "imu only": (1, 0, 0), "radar only": (0, 1, 0), "vision only": (0, 0, 1),
    }
    print(f"{'configuration':<20}{'acc':>7}{'macroF1':>9}{'fall-rec':>10}")
    rob = {}
    for name, drop in configs.items():
        Xd, yd = build_matrix(test, drop=drop)
        Xd_s = (Xd - mu) / sd
        pd_ = clf.predict(Xd_s)
        a, f, r = scores(yd, pd_)
        rob[name] = a
        print(f"{name:<20}{a:>7.3f}{f:>9.3f}{r:>10.3f}")

    # robustness bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(rob.keys()); vals = [rob[n] for n in names]
    colors = ["#2e8b57" if v >= 0.8 else "#e08a2b" if v >= 0.6 else "#c0392b" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(1 / 5, ls="--", color="grey", label="random (0.20)")
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title("Graceful degradation under sensor dropout")
    ax.legend(); plt.xticks(rotation=35, ha="right"); fig.tight_layout()
    fig.savefig(f"{FIG}/robustness.png", dpi=120); plt.close(fig)
    print("\nwrote confusion_matrix.png and robustness.png to figures/")


if __name__ == "__main__":
    main()
