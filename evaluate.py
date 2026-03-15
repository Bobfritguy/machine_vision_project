#!/usr/bin/env python3
"""
evaluate.py – Evaluate a trained checkpoint on the SR_Test split.

Reports
-------
  - Top-1 accuracy (overall)
  - Per-class accuracy
  - Confusion matrix saved as PNG

Usage
-----
    python evaluate.py --checkpoint runs/exp/checkpoint_best.pt \\
        --root /big_boy_hdd/ASL_Dataset/ASL

    # Use HR split
    python evaluate.py --checkpoint runs/exp/checkpoint_best.pt \\
        --root /big_boy_hdd/ASL_Dataset/ASL --resolution HR
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ASLEventDataset
from model import build_model
from utils import CLASSES, NUM_CLASSES, PreprocessConfig, get_logger

logger = get_logger(__name__)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_preds, all_labels


def compute_metrics(preds: np.ndarray, labels: np.ndarray):
    overall_acc = (preds == labels).mean()

    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        if mask.sum() == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = float((preds[mask] == labels[mask]).mean())

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    return overall_acc, per_class_acc, cm


def save_confusion_matrix(cm: np.ndarray, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", out_path)
    except ImportError:
        logger.warning("matplotlib not installed; skipping confusion matrix plot.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ASL event-camera classifier")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--root", default="/big_boy_hdd/ASL_Dataset/ASL")
    p.add_argument("--resolution", choices=["LR", "HR"], default=None,
                   help="Override resolution (defaults to value stored in checkpoint)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default=None,
                   help="Directory for outputs. Defaults to checkpoint directory.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Reconstruct PreprocessConfig from checkpoint
    if "preprocess_cfg" in ckpt:
        cfg = PreprocessConfig.from_dict(ckpt["preprocess_cfg"])
        logger.info("Loaded PreprocessConfig from checkpoint: %s", cfg.to_dict())
    else:
        # Fall back: look for sidecar JSON
        cfg_path = ckpt_path.parent / "preprocess_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = PreprocessConfig.from_dict(json.load(f))
            logger.info("Loaded PreprocessConfig from %s", cfg_path)
        else:
            logger.warning("No preprocess config found; using defaults.")
            cfg = PreprocessConfig()

    # Allow CLI override of resolution
    if args.resolution is not None:
        from dataset import _SENSOR_RES
        cfg = PreprocessConfig(
            source_resolution=_SENSOR_RES[args.resolution],
            target_resolution=cfg.target_resolution,
            representation=cfg.representation,
            num_bins=cfg.num_bins,
            normalization=cfg.normalization,
            polarity_convention=cfg.polarity_convention,
            remap_strategy=cfg.remap_strategy,
        )

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset and loader
    # ------------------------------------------------------------------
    # Determine resolution string from source_resolution
    from dataset import _SENSOR_RES
    res_str = next(
        (k for k, v in _SENSOR_RES.items() if v == cfg.source_resolution), "LR"
    )
    test_ds = ASLEventDataset(args.root, split="test",
                              resolution=res_str, cfg=cfg)
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    logger.info("Test set: %d samples", len(test_ds))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(in_channels=cfg.num_channels, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    logger.info("Model loaded (epoch %d)", ckpt.get("epoch", -1) + 1)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    preds, labels = run_evaluation(model, loader, device)
    overall_acc, per_class_acc, cm = compute_metrics(preds, labels)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Overall Top-1 Accuracy: %.2f%%", overall_acc * 100)
    logger.info("Per-class accuracy:")
    for cls, acc in per_class_acc.items():
        if acc is None:
            logger.info("  %-4s  N/A (no samples)", cls)
        else:
            logger.info("  %-4s  %.2f%%", cls, acc * 100)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "overall_top1_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "checkpoint": str(ckpt_path),
        "preprocess_cfg": cfg.to_dict(),
    }
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)

    save_confusion_matrix(cm, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
