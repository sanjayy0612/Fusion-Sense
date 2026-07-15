"""FusionSense — lightweight sensor-health-aware multimodal HAR."""
from .config import CFG, Config
from .contract import FusionWindow, ACTIVITIES, LABEL2ID, ID2LABEL

__all__ = ["CFG", "Config", "FusionWindow", "ACTIVITIES", "LABEL2ID", "ID2LABEL"]
