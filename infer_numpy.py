#!/usr/bin/env python3
"""
infer_numpy.py – Run offline inference on one or more .npy event files.

Usage
-----
    # Single file
    python infer_numpy.py --checkpoint runs/exp/checkpoint_best.pt \\
        --input /big_boy_hdd/ASL_Dataset/ASL/SR_Test/LR/a/a_0002.npy

    # Directory of .npy files (all files in dir)
    python infer_numpy.py --checkpoint runs/exp/checkpoint_best.pt \\
        --input /big_boy_hdd/ASL_Dataset/ASL/SR_Test/LR/b/

    # Override preprocessing strategy
    python infer_numpy.py --checkpoint runs/exp/checkpoint_best.pt \\
        --input sample.npy --remap-strategy remap_then_accumulate
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import build_model
from preprocess import events_to_frame, load_events
from utils import CLASSES, NUM_CLASSES, PreprocessConfig, get_logger

logger = get_logger(__name__)


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Load preprocessing config
    if "preprocess_cfg" in ckpt:
        cfg = PreprocessConfig.from_dict(ckpt["preprocess_cfg"])
    else:
        cfg_json = Path(ckpt_path).parent / "preprocess_config.json"
        if cfg_json.exists():
            with open(cfg_json) as f:
                cfg = PreprocessConfig.from_dict(json.load(f))
        else:
            logger.warning("No PreprocessConfig found; using LR defaults.")
            cfg = PreprocessConfig()

    model = build_model(in_channels=cfg.num_channels, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def infer_file(path: str, model: torch.nn.Module,
               cfg: PreprocessConfig, device: torch.device) -> dict:
    """
    Run inference on a single .npy event file.

    Returns
    -------
    dict with keys: predicted_class, confidence, scores (dict class->prob)
    """
    events = load_events(path)
    tensor = events_to_frame(events, cfg)             # [C, H, W]
    tensor = tensor.unsqueeze(0).to(device)           # [1, C, H, W]

    logits = model(tensor)                            # [1, num_classes]
    probs  = F.softmax(logits, dim=1).squeeze(0)      # [num_classes]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    predicted  = CLASSES[pred_idx]

    return {
        "file": path,
        "predicted_class": predicted,
        "confidence": confidence,
        "scores": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
    }


def parse_args():
    p = argparse.ArgumentParser(description="Offline inference on .npy event files")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True,
                   help="Path to a .npy file or a directory of .npy files")
    p.add_argument("--top-k", type=int, default=3,
                   help="Print top-k class predictions (default 3)")
    p.add_argument("--remap-strategy", default=None,
                   choices=["remap_then_accumulate", "accumulate_then_resize"],
                   help="Override remap strategy from checkpoint")
    p.add_argument("--out", default=None,
                   help="Optional path to save results as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, cfg = load_checkpoint(args.checkpoint, device)

    if args.remap_strategy:
        cfg.remap_strategy = args.remap_strategy
        logger.info("Overriding remap_strategy → %s", args.remap_strategy)

    logger.info("PreprocessConfig: %s", cfg.to_dict())

    # Collect files
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npy"))
    elif input_path.suffix == ".npy":
        files = [input_path]
    else:
        raise ValueError(f"--input must be a .npy file or directory, got: {input_path}")

    if not files:
        logger.error("No .npy files found at %s", input_path)
        return

    results = []
    for fpath in files:
        try:
            result = infer_file(str(fpath), model, cfg, device)
            results.append(result)

            # Print top-k predictions
            sorted_scores = sorted(result["scores"].items(),
                                   key=lambda x: x[1], reverse=True)
            top_str = "  ".join(
                f"{cls}:{score:.2%}" for cls, score in sorted_scores[:args.top_k]
            )
            print(f"{fpath.name:<30}  pred={result['predicted_class']}  "
                  f"conf={result['confidence']:.2%}  top{args.top_k}=[{top_str}]")
        except Exception as exc:
            logger.warning("Failed on %s: %s", fpath, exc)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved → %s", args.out)


if __name__ == "__main__":
    main()
