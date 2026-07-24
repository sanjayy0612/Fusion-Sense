"""Draw block / physical / circuit diagrams for the review deck (matplotlib).
Run:  python scripts/make_diagrams.py  ->  figures/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, GREEN, ORANGE, GREY = "#3b7dd8", "#2e8b57", "#e08a2b", "#5b6570"


def box(ax, x, y, w, h, text, color, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color="black"))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.2, 2.6, 1.9, 0.7, "MPU-6050\nIMU (I2C)", GREEN, 9)
    box(ax, 0.2, 1.6, 1.9, 0.7, "LD2410\nRadar (UART)", GREEN, 9)
    box(ax, 0.2, 0.6, 1.9, 0.7, "Camera\n(USB/CSI)", GREEN, 9)
    box(ax, 2.9, 1.6, 1.9, 0.7, "ESP32\nGateway", ORANGE, 9)
    box(ax, 5.5, 0.6, 2.4, 2.7, "Raspberry Pi 5\n(edge brain)\n\nsync + windowing\nFusionWindow", BLUE, 9)
    box(ax, 8.5, 1.9, 3.2, 1.2, "FusionSense model\nencoders -> cross-modal\nattention -> trust", BLUE, 9)
    box(ax, 8.5, 0.5, 3.2, 0.9, "MQTT -> Dashboard\nactivity + trust + fall alert", GREY, 9)
    arrow(ax, 2.1, 2.95, 2.9, 2.1); arrow(ax, 2.1, 1.95, 2.9, 1.95)
    arrow(ax, 2.1, 0.95, 5.5, 1.3)               # camera -> pi
    arrow(ax, 4.8, 1.95, 5.5, 1.95)              # esp -> pi
    arrow(ax, 7.9, 2.3, 8.5, 2.5)                # pi -> model
    arrow(ax, 10.1, 1.9, 10.1, 1.4)             # model -> mqtt
    ax.set_title("FusionSense — Logical / System Architecture", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/architecture_block.png", dpi=120); plt.close(fig)


def physical():
    fig, ax = plt.subplots(figsize=(9, 6)); ax.set_xlim(0, 9); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 8.4, 5.4, boxstyle="round,pad=0.02",
                                fc="#f4f4f4", ec=GREY, lw=1.5))
    ax.text(4.5, 5.4, "Room (top view)", ha="center", fontsize=11, color=GREY)
    box(ax, 0.7, 4.3, 2.4, 0.7, "Camera + LD2410\n(on wall/tripod)", GREEN, 9)
    box(ax, 0.7, 3.4, 2.4, 0.7, "Raspberry Pi 5\n(beside sensors)", BLUE, 9)
    # person
    ax.plot(6.2, 2.6, "o", ms=18, color=ORANGE)
    ax.plot([6.2, 6.2], [2.4, 1.4], color=ORANGE, lw=3)
    ax.plot([5.8, 6.6], [2.0, 2.0], color=ORANGE, lw=3)
    box(ax, 6.7, 1.9, 1.9, 0.6, "ESP32 + IMU\n(waist)", ORANGE, 8)
    arrow(ax, 3.1, 4.2, 6.0, 2.9); ax.text(4.3, 3.9, "camera + radar\nobserve person", fontsize=8, color=GREY)
    ax.annotate("", xy=(6.7, 2.2), xytext=(3.1, 3.7),
                arrowprops=dict(arrowstyle="-|>", ls="--", color=BLUE))
    ax.text(4.2, 2.7, "ESP32 -> Pi\n(UART/Wi-Fi)", fontsize=8, color=BLUE)
    ax.set_title("FusionSense — Physical Design (deployment layout)", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/physical_layout.png", dpi=120); plt.close(fig)


def circuit():
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")
    box(ax, 4.2, 2.0, 2.6, 1.6, "ESP32\nDevKit", ORANGE, 11)
    box(ax, 0.4, 3.6, 2.3, 1.0, "MPU-6050\nIMU", GREEN, 10)
    box(ax, 0.4, 0.7, 2.3, 1.0, "LD2410\nRadar", GREEN, 10)

    def wire(x1, y1, x2, y2, color, label):
        ax.plot([x1, x2], [y1, y2], color=color, lw=2)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=8, color=color, ha="center")

    # IMU I2C
    wire(2.7, 4.35, 4.2, 3.35, "red",   "VCC->3V3")
    wire(2.7, 4.15, 4.2, 3.15, "black", "GND")
    wire(2.7, 3.95, 4.2, 2.95, "green", "SDA->D21")
    wire(2.7, 3.75, 4.2, 2.75, "blue",  "SCL->D22")
    # Radar UART
    wire(2.7, 1.5, 4.2, 2.55, "red",    "VCC->5V")
    wire(2.7, 1.3, 4.2, 2.35, "black",  "GND")
    wire(2.7, 1.1, 4.2, 2.15, "purple", "TX->D16(RX2)")
    wire(2.7, 0.9, 4.2, 2.0,  "orange", "RX->D17(TX2)")
    # ESP32 -> host
    box(ax, 8.2, 2.3, 2.4, 1.0, "Edge node\n(Pi / laptop)", BLUE, 10)
    wire(6.8, 2.8, 8.2, 2.8, GREY, "USB serial 115200")
    ax.set_title("FusionSense — Circuit / Wiring Diagram", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/circuit_diagram.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    architecture(); physical(); circuit()
    print("wrote diagrams to", os.path.abspath(FIG))
