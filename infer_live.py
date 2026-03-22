#!/usr/bin/env python3
"""
infer_live.py – Live inference for event streams from a Prophesee GENX320.

The GENX320 is a 320×320 pixel event camera. Because our model was trained on
the ASL-DVS dataset captured with a DAVIS240C camera (downsampled to 120×90
for the LR split), we must map live events into the same representation before
running the classifier.

See the README section "Adapting from dataset camera to GENX320" for a full
explanation of the coordinate and resolution mismatch and how it is handled.

Usage as a module
-----------------
    from infer_live import LiveInferencer

    inf = LiveInferencer("runs/exp/checkpoint_best.pt")
    # events: np.ndarray shape (N, 4), columns [t, x, y, p]
    result = inf.predict(events)
    print(result["predicted_class"], result["confidence"])

Command-line demo (simulate with a saved .npy file)
----------------------------------------------------
    python infer_live.py --checkpoint runs/exp/checkpoint_best.pt \\
        --demo-npy /big_boy_hdd/ASL_Dataset/ASL/SR_Test/LR/a/a_0002.npy \\
        --simulate-genx320

    --simulate-genx320  rescales the demo file's coordinates to 320×320
    to mimic what a GENX320 would produce before feeding into the inferencer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model
from preprocess import events_to_frame, load_events, remap_coords
from utils import CLASSES, NUM_CLASSES, PreprocessConfig, get_logger

logger = get_logger(__name__)

# Prophesee GENX320 sensor resolution (W × H)
GENX320_RESOLUTION = (320, 320)


class LiveInferencer:
    """
    Wraps a trained checkpoint for real-time ASL classification from live
    event streams captured by the Prophesee GENX320.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt checkpoint produced by train.py.
    device : str, optional
        "cuda", "cpu", or None (auto-detect).
    live_sensor_resolution : tuple (W, H)
        Resolution of the live camera. Default: GENX320_RESOLUTION (320, 320).

    Notes on the training ↔ live resolution mismatch
    -------------------------------------------------
    Training data (LR split) was captured by a DAVIS240C at 240×180 px, then
    2×2 downsampled to give 120×90 (the LR sensor resolution).

    The GENX320 is a square 320×320 sensor. Two differences must be handled:

    1. Different spatial scale:  320 ≠ 120 or 240.
    2. Different aspect ratio:   320×320 (1:1)  vs  120×90 (4:3).

    Strategy used here (accumulate_then_resize, the default):
      a) Accumulate events in the native 320×320 coordinate space.
      b) Apply the same log1p normalisation as training.
      c) Bicubic-resize from 320×320 to the model's target_resolution
         (256×256 by default).  This is a pure scaling step; the model
         never sees raw pixel counts.

    The aspect ratio change (1:1 → anything needed for the model) is handled
    gracefully by the resize. Alternatively you can centre-crop the 320×320
    accumulation to 240×180 (matching the original DAVIS240C FOV crop) before
    resizing; enable this with crop_to_training_aspect=True.

    Alternative strategy (remap_then_accumulate):
      Scale x by 120/320, y by 90/320, round, then accumulate at 120×90.
      This squashes the square FOV into a 4:3 rectangle before accumulation.
      It is lossier but produces the smallest intermediate buffer.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        live_sensor_resolution: tuple = GENX320_RESOLUTION,
        crop_to_training_aspect: bool = False,
        flip_y: bool = True,
    ):
        self.flip_y = flip_y
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.live_sensor_resolution = live_sensor_resolution
        self.crop_to_training_aspect = crop_to_training_aspect

        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=True)

        # Reconstruct PreprocessConfig from checkpoint, then override
        # source_resolution with the live sensor.
        if "preprocess_cfg" in ckpt:
            train_cfg = PreprocessConfig.from_dict(ckpt["preprocess_cfg"])
        else:
            cfg_json = Path(checkpoint_path).parent / "preprocess_config.json"
            if cfg_json.exists():
                with open(cfg_json) as f:
                    train_cfg = PreprocessConfig.from_dict(json.load(f))
            else:
                logger.warning("No PreprocessConfig found; using LR defaults.")
                train_cfg = PreprocessConfig()

        # Live config: same as training except source_resolution = live sensor
        self.live_cfg = PreprocessConfig(
            source_resolution=live_sensor_resolution,
            target_resolution=train_cfg.target_resolution,
            representation=train_cfg.representation,
            num_bins=train_cfg.num_bins,
            normalization=train_cfg.normalization,
            polarity_convention=train_cfg.polarity_convention,
            remap_strategy="accumulate_then_resize",  # best for cross-sensor
        )
        self.train_cfg = train_cfg

        logger.info("Training config:  source_res=%s  target_res=%s",
                    train_cfg.source_resolution, train_cfg.target_resolution)
        logger.info("Live config:      source_res=%s  target_res=%s",
                    self.live_cfg.source_resolution, self.live_cfg.target_resolution)

        self.model = build_model(
            in_channels=train_cfg.num_channels,
            num_classes=NUM_CLASSES,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)
        logger.info("Model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def predict(self, events: np.ndarray) -> dict:
        """
        Classify a chunk of live events.

        Parameters
        ----------
        events : np.ndarray, shape (N, 4), columns [t, x, y, p]
            Raw events from the GENX320 (or any sensor; coordinates must
            lie within live_sensor_resolution).

            Polarity convention: {0,1} or {-1,+1} – both are accepted.

        Returns
        -------
        dict:
            predicted_class  : str  – top-1 ASL letter
            confidence       : float  – softmax probability of top class
            scores           : dict str→float  – all 24 class probabilities
        """
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) event array, got {events.shape}")
        if len(events) == 0:
            raise ValueError("Event array is empty.")

        events = events.astype(np.float32)

        # Flip y-axis to match training data convention.
        # The DAVIS240C (training sensor) and GenX320 use opposite y-axis
        # orientations: the same gesture appears vertically flipped between
        # the two. We flip y here so the model sees the same orientation
        # it was trained on.
        if self.flip_y:
            _, H = self.live_sensor_resolution
            events[:, 2] = (H - 1) - events[:, 2]

        # Optional: centre-crop from square to 4:3 before accumulation.
        # This mimics the training camera's FOV if desired.
        if self.crop_to_training_aspect:
            events = self._centre_crop_events(events)

        # Convert to frame using live_cfg (source=320×320, resize to 256×256)
        tensor = events_to_frame(events, self.live_cfg)   # [C, 256, 256]
        tensor = tensor.unsqueeze(0).to(self.device)      # [1, C, 256, 256]

        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()
        predicted  = CLASSES[pred_idx]

        return {
            "predicted_class": predicted,
            "confidence": confidence,
            "scores": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        }

    def _centre_crop_events(self, events: np.ndarray) -> np.ndarray:
        """
        Crop events from a square sensor to the training camera's 4:3
        aspect ratio, centred.

        GENX320: 320×320.  Training sensor: 120×90 → ratio = 4:3.
        A 4:3 crop of a 320-wide square has height = 320 * (3/4) = 240.
        We discard the top and bottom (320-240)/2 = 40 rows each.
        """
        W, H = self.live_sensor_resolution
        new_H = int(round(W * 3 / 4))    # 240 for 320-wide sensor
        margin = (H - new_H) // 2        # 40 px top & bottom
        mask = (events[:, 2] >= margin) & (events[:, 2] < margin + new_H)
        cropped = events[mask].copy()
        cropped[:, 2] -= margin           # shift y to start at 0
        return cropped


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Live inference demo for Prophesee GENX320 events"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--demo-npy", default=None,
                   help="A .npy event file to use as demo input")
    p.add_argument("--simulate-genx320", action="store_true",
                   help="Rescale demo file coords to 320×320 before inference, "
                        "simulating GENX320 input")
    p.add_argument("--crop-to-training-aspect", action="store_true", default=True,
                   help="Centre-crop the 320×320 frame to 4:3 before accumulation (default: on)")
    p.add_argument("--no-crop-to-training-aspect", dest="crop_to_training_aspect",
                   action="store_false",
                   help="Disable centre-crop; use full square sensor FOV")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    inferencer = LiveInferencer(
        args.checkpoint,
        device=device_str,
        live_sensor_resolution=GENX320_RESOLUTION,
        crop_to_training_aspect=args.crop_to_training_aspect,
    )

    if args.demo_npy:
        events = load_events(args.demo_npy)
        logger.info("Loaded %d events from %s", len(events), args.demo_npy)
        logger.info("  native coords: x=[%d,%d] y=[%d,%d]",
                    int(events[:,1].min()), int(events[:,1].max()),
                    int(events[:,2].min()), int(events[:,2].max()))

        if args.simulate_genx320:
            # Remap the dataset (LR 120×90) coordinates to 320×320
            src_res = inferencer.train_cfg.source_resolution
            events = remap_coords(events, src_res, GENX320_RESOLUTION)
            logger.info("Remapped to GENX320 320×320: x=[%d,%d] y=[%d,%d]",
                        int(events[:,1].min()), int(events[:,1].max()),
                        int(events[:,2].min()), int(events[:,2].max()))

        result = inferencer.predict(events)

        sorted_scores = sorted(result["scores"].items(),
                               key=lambda x: x[1], reverse=True)
        print(f"\nPredicted class : {result['predicted_class']}")
        print(f"Confidence      : {result['confidence']:.2%}")
        print(f"Top-{args.top_k} scores:")
        for cls, score in sorted_scores[:args.top_k]:
            bar = "█" * int(score * 40)
            print(f"  {cls:>2}  {score:.4f}  {bar}")
    else:
        print("No --demo-npy provided.  Import LiveInferencer from this module.")
        print("Example:")
        print("  from infer_live import LiveInferencer")
        print("  inf = LiveInferencer('runs/exp/checkpoint_best.pt')")
        print("  result = inf.predict(events_array)")


if __name__ == "__main__":
    main()
