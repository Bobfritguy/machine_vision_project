"""
dataset.py – PyTorch Dataset for the ASL event-camera dataset.

Each .npy file contains one event sample as a (N, 4) array with columns
[t, x, y, p].  The class label is the name of the parent directory.

Usage
-----
    from utils import PreprocessConfig
    from dataset import ASLEventDataset

    cfg = PreprocessConfig(source_resolution=(120, 90))
    train_ds = ASLEventDataset(root="/big_boy_hdd/ASL_Dataset/ASL",
                               split="train", resolution="LR", cfg=cfg)
    img, label = train_ds[0]   # img: Tensor[C,H,W]  label: int
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import events_to_frame, load_events
from utils import CLASSES, CLASS_TO_IDX, PreprocessConfig, get_logger

logger = get_logger(__name__)

# Canonical sensor resolutions discovered by inspect_dataset.py
_SENSOR_RES = {
    "LR": (120, 90),    # (W, H)
    "HR": (240, 180),
}


class ASLEventDataset(Dataset):
    """
    Parameters
    ----------
    root        : path to the ASL/ directory (contains SR_Train/, SR_Test/).
    split       : "train" or "test".
    resolution  : "LR" (default) or "HR".
    cfg         : PreprocessConfig; source_resolution is set automatically
                  from `resolution` unless you override it manually.
    transform   : optional callable applied to the output Tensor [C,H,W].
    skip_errors : if True, silently skip malformed files (they are logged).
                  if False, raise on the first bad file (good for debugging).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: str = "LR",
        cfg: Optional[PreprocessConfig] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = True,
    ) -> None:
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        assert resolution in ("LR", "HR"), f"resolution must be 'LR' or 'HR', got {resolution!r}"

        split_dir = "SR_Train" if split == "train" else "SR_Test"
        self.data_root = Path(root) / split_dir / resolution

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        # Auto-set source_resolution if not explicitly overridden
        auto_res = _SENSOR_RES[resolution]
        if cfg is None:
            cfg = PreprocessConfig(source_resolution=auto_res)
        elif cfg.source_resolution != auto_res:
            logger.warning(
                "cfg.source_resolution %s differs from the canonical %s "
                "resolution %s. Using your value.",
                cfg.source_resolution, resolution, auto_res,
            )

        self.cfg = cfg
        self.transform = transform
        self.skip_errors = skip_errors
        self.resolution = resolution
        self.split = split

        logger.info(
            "Loading %s %s dataset from %s  [source_res=%s target_res=%s repr=%s]",
            resolution, split, self.data_root,
            cfg.source_resolution, cfg.target_resolution, cfg.representation,
        )

        self.samples: List[Tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        missing_classes = []
        for cls in CLASSES:
            cls_dir = self.data_root / cls
            if not cls_dir.exists():
                missing_classes.append(cls)
                continue
            for fname in sorted(cls_dir.iterdir()):
                if fname.suffix == ".npy":
                    self.samples.append((str(fname), CLASS_TO_IDX[cls]))

        if missing_classes:
            logger.warning("Classes not found in %s: %s", self.data_root, missing_classes)

        logger.info("Index built: %d samples, %d classes", len(self.samples),
                    len(CLASSES) - len(missing_classes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            events = load_events(path)
            tensor = events_to_frame(events, self.cfg)  # [C, H, W]
        except Exception as exc:
            if self.skip_errors:
                logger.warning("Skipping bad file %s: %s", path, exc)
                # Return a zero tensor so the DataLoader doesn't crash.
                # The caller should use a collate_fn that filters these out,
                # or simply accept the occasional zero-frame.
                C = self.cfg.num_channels
                W, H = self.cfg.target_resolution
                tensor = torch.zeros(C, H, W, dtype=torch.float32)
            else:
                raise

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    def class_counts(self) -> dict:
        """Return {class_name: count} for all samples in this split."""
        counts: dict = {c: 0 for c in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts
