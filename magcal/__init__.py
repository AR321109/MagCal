"""Public API for the compact MagCal-MSNet implementation."""

from .pipeline import MagCalConfig, MagCalDetector
from .magnitude import patchwise_same_kernel_rarity

__all__ = ["MagCalConfig", "MagCalDetector", "patchwise_same_kernel_rarity"]
