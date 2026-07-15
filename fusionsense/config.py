"""Central configuration. Change window/rate/dim here; everything reads from this."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- window definition ---
    window_seconds: float = 2.0

    imu_hz: int = 50
    radar_hz: int = 20
    vision_fps: int = 10

    # --- per-modality channel dims ---
    imu_ch: int = 6          # ax ay az gx gy gz
    radar_k: int = 8         # LD2410-style per-frame features (gates/energy/range...)
    vision_dv: int = 32      # vision embedding dim (NOT raw pixels)

    # --- model dims ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    n_classes: int = 5

    # --- training ---
    modality_dropout_p: float = 0.3   # prob. of zeroing a modality per sample
    use_health_conditioning: bool = True

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
