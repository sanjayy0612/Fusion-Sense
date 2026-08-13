"""Central configuration. Change window/rate/dim + dataset paths here."""
from dataclasses import dataclass
from pathlib import Path


# repo-root/data/raw/<dataset>/  — where you unzip downloaded datasets
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class Config:
    # --- window definition ---
    window_seconds: float = 2.0

    imu_hz: int = 50
    radar_hz: int = 20
    vision_fps: int = 10

    # --- per-modality channel dims ---
    imu_ch: int = 6          # ax ay az gx gy gz
    radar_k: int = 8         # radar per-frame features (voxel/energy summary)
    vision_dv: int = 99      # open-source camera pose dim: 33 MediaPipe landmarks * xyz.
    #   V1 uses MediaPipe/MoveNet-style pose features instead of training a raw
    #   camera model. The simulator scales to whatever vision_dv is.

    # --- model dims ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    n_classes: int = 7

    # --- training ---
    modality_dropout_p: float = 0.3   # prob. of zeroing a modality per sample
    use_health_conditioning: bool = True

    # --- dataset directory names under DATA_ROOT ---
    imu_dir: str = "sisfall"          # single-modality IMU (or "uci_har")
    radar_dir: str = "radhar"         # single-modality mmWave
    paired_dir: str = "cmhad"          # prepared C-MHAD camera + IMU cache

    @property
    def t_imu(self) -> int:
        return int(self.window_seconds * self.imu_hz)     # 100

    @property
    def t_radar(self) -> int:
        return int(self.window_seconds * self.radar_hz)   # 40

    @property
    def t_vis(self) -> int:
        return int(self.window_seconds * self.vision_fps)  # 20


CFG = Config()
