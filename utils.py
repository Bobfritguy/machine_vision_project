"""
utils.py – Shared utilities for the ASL event-camera pipeline.

Provides:
  - PreprocessConfig : dataclass that fully describes the preprocessing pipeline
  - seed_everything  : deterministic seeding helper
  - get_logger       : module-level logger factory
  - CLASSES / CLASS_TO_IDX / NUM_CLASSES : dataset label maps
"""

import logging
import os
import random
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Label map  (24 ASL letters; J and Z require motion and are excluded)
# ---------------------------------------------------------------------------
CLASSES: list = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
    "v", "w", "x", "y",
]
CLASS_TO_IDX: dict = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """
    Single source-of-truth for how raw events are turned into model inputs.

    Attributes
    ----------
    source_resolution : (W, H) of the sensor that produced the events.
        LR dataset  -> (120,  90)   [detected by inspect_dataset.py]
        HR dataset  -> (240, 180)
        GENX320     -> (320, 320)
    target_resolution : (W, H) of the final model input (default 256x256).
    representation : one of {"two_channel", "signed", "voxel_grid"}.
    num_bins : number of temporal bins for voxel_grid representation.
    normalization : one of {"log1p", "minmax", "none"}.
    polarity_convention : one of {"01", "-11"}.
        "01"  -> polarity in {0, 1}   (as stored in this dataset)
        "-11" -> polarity in {-1, +1}
    remap_strategy : how to handle source_resolution != target_resolution.

        "remap_then_accumulate"
            Scale (x, y) coordinates into [0, target_W) x [0, target_H)
            first, then accumulate pixel counts.
            Pro:  no blur from image interpolation; memory-efficient.
            Con:  integer rounding during coord scaling can merge adjacent
                  pixels and discard fine spatial differences.

        "accumulate_then_resize"  [DEFAULT]
            Accumulate in native source_resolution, then bicubic-resize
            the resulting image to target_resolution.
            Pro:  preserves all spatial detail during accumulation; the
                  resize is a smooth linear operation.
            Con:  slight blur from bicubic interpolation; transiently
                  holds a native-resolution image in memory.

        For the GENX320 (320x320) -> training LR (120x90) mismatch the
        default strategy is the better choice because the aspect ratios
        differ; accumulate_then_resize lets you centre-crop or letterbox
        as needed before the final resize.
    """

    source_resolution: Tuple[int, int] = (120, 90)    # (W, H) – LR default
    target_resolution: Tuple[int, int] = (256, 256)   # (W, H) – model input
    representation: str = "two_channel"               # two_channel | signed | voxel_grid
    num_bins: int = 5                                 # voxel_grid only
    normalization: str = "log1p"                      # log1p | minmax | none
    polarity_convention: str = "01"                   # 01 | -11
    remap_strategy: str = "accumulate_then_resize"    # see docstring above

    def __post_init__(self) -> None:
        assert self.representation in {"two_channel", "signed", "voxel_grid"}, \
            f"Unknown representation: {self.representation!r}"
        assert self.normalization in {"log1p", "minmax", "none"}, \
            f"Unknown normalization: {self.normalization!r}"
        assert self.polarity_convention in {"01", "-11"}, \
            f"Unknown polarity_convention: {self.polarity_convention!r}"
        assert self.remap_strategy in {"remap_then_accumulate", "accumulate_then_resize"}, \
            f"Unknown remap_strategy: {self.remap_strategy!r}"

    @property
    def num_channels(self) -> int:
        """Number of image channels produced by this config."""
        if self.representation == "two_channel":
            return 2
        if self.representation == "signed":
            return 1
        if self.representation == "voxel_grid":
            return self.num_bins
        raise ValueError(self.representation)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(**{
            k: tuple(v) if k.endswith("_resolution") else v
            for k, v in d.items()
        })


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set all relevant RNG seeds for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s %(levelname)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
