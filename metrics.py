#!/usr/bin/env python3
"""
metrics.py – Dataset and model evaluation metrics for the ASL event-camera project.

Produces three figures:
  1. class_counts.png        – sample counts per class across all dataset splits
  2. per_class_f1.png        – per-class F1 score for both models on the ASL test set
  3. confusion_matrices.png  – confusion matrices for both models, labelled by model name

Usage
-----
    python metrics.py \
        --asl-root datasets/ASL \
        --genx-root datasets/genx320_recorded \
        --model-aug models/model_with_augmentations.pt \
        --model-rec models/with_recorded.pt \
        --out metrics_output
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ASLEventDataset, FlatEventDataset
from model import build_model
from utils import CLASSES, NUM_CLASSES, PreprocessConfig, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset class counting (no model, no preprocessing needed)
# ---------------------------------------------------------------------------

def count_classes_from_dir(root: Path, classes: list) -> dict:
    """Count .npy files per class in a flat class-subdirectory layout."""
    counts = {c: 0 for c in classes}
    for cls in classes:
        cls_dir = root / cls
        if cls_dir.exists():
            counts[cls] = sum(1 for f in cls_dir.iterdir() if f.suffix == ".npy")
    return counts


def count_asl_split(asl_root: Path, split: str, resolution: str = "LR") -> dict:
    split_dir = "SR_Train" if split == "train" else "SR_Test"
    return count_classes_from_dir(asl_root / split_dir / resolution, CLASSES)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def load_model_and_cfg(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = PreprocessConfig.from_dict(ckpt["preprocess_cfg"])
    model = build_model(in_channels=cfg.num_channels, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    return model, cfg, ckpt.get("best_val_acc", None)


# ---------------------------------------------------------------------------
# Per-class F1 score
# ---------------------------------------------------------------------------

def per_class_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    f1 = np.zeros(num_classes)
    for i in range(num_classes):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1[i] = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_cm(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_class_counts(splits: dict, out_path: Path):
    """
    splits: {label: {class: count}}  e.g. {"ASL Train": {...}, "ASL Test": {...}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_splits = len(splits)
    x = np.arange(NUM_CLASSES)
    width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

    fig, ax = plt.subplots(figsize=(16, 5))
    for (label, counts), offset in zip(splits.items(), offsets):
        values = [counts.get(c, 0) for c in CLASSES]
        ax.bar(x + offset, values, width=width * 0.9, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("Class (ASL letter)")
    ax.set_ylabel("Sample count")
    ax.set_title("Class distribution across datasets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Class counts saved → %s", out_path)


def plot_per_class_f1(model_results: dict, out_path: Path):
    """
    model_results: {model_label: f1_array}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_results)
    x = np.arange(NUM_CLASSES)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(16, 5))
    for (label, f1), offset in zip(model_results.items(), offsets):
        ax.bar(x + offset, f1 * 100, width=width * 0.9, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("Class (ASL letter)")
    ax.set_ylabel("F1 score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class F1 score on ASL test set")
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Per-class F1 saved → %s", out_path)


def plot_confusion_matrices(model_results: dict, out_path: Path):
    """
    model_results: {model_label: (cm, overall_acc)}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(13 * n_models, 12))
    if n_models == 1:
        axes = [axes]

    for ax, (label, (cm, overall_acc)) in zip(axes, model_results.items()):
        # Normalise rows → recall per class (makes colour scale consistent
        # regardless of class imbalance, and highlights off-diagonal errors).
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{label}\n(overall acc: {overall_acc * 100:.2f}%)", fontsize=11)

        # Annotate with raw counts
        thresh = 0.5
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=6,
                )

    fig.suptitle("Confusion matrices (row-normalised) – ASL test set", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset and model metrics")
    p.add_argument("--asl-root",   default="datasets/ASL",
                   help="Path to ASL/ directory (contains SR_Train/, SR_Test/)")
    p.add_argument("--genx-root",  default="datasets/genx320_recorded",
                   help="Path to genx320_recorded/ flat dataset")
    p.add_argument("--model-aug",  default="models/model_with_augmentations.pt",
                   help="Checkpoint: model trained with augmentations only")
    p.add_argument("--model-rec",  default="models/with_recorded.pt",
                   help="Checkpoint: model trained with augmentations + recorded data")
    p.add_argument("--resolution", choices=["LR", "HR"], default="LR",
                   help="ASL resolution split to use for evaluation (default: LR)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--out",        default="metrics_output",
                   help="Output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Class counts (no model needed)
    # ------------------------------------------------------------------
    asl_root  = Path(args.asl_root)
    genx_root = Path(args.genx_root)

    split_counts = {}

    if asl_root.exists():
        split_counts["ASL Train"] = count_asl_split(asl_root, "train", args.resolution)
        split_counts["ASL Test"]  = count_asl_split(asl_root, "test",  args.resolution)
        logger.info("ASL train total: %d", sum(split_counts["ASL Train"].values()))
        logger.info("ASL test  total: %d", sum(split_counts["ASL Test"].values()))
    else:
        logger.warning("ASL root not found: %s", asl_root)

    if genx_root.exists():
        split_counts["GenX320"] = count_classes_from_dir(genx_root, CLASSES)
        logger.info("GenX320 total: %d", sum(split_counts["GenX320"].values()))
    else:
        logger.warning("GenX320 root not found: %s", genx_root)

    if split_counts:
        plot_class_counts(split_counts, out_dir / "class_counts.png")
    else:
        logger.error("No dataset directories found – skipping class count plot.")

    # ------------------------------------------------------------------
    # 2 & 3. Model evaluation – per-class F1 + confusion matrices
    # ------------------------------------------------------------------
    model_specs = {
        "Augmentations only": args.model_aug,
        "Aug + Recorded":     args.model_rec,
    }

    f1_results = {}
    cm_results = {}

    # We evaluate both models on the same ASL test set.
    # The first model we load determines the dataset (they share the same cfg).
    test_loader = None
    test_labels_cached = None

    for model_label, ckpt_path_str in model_specs.items():
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found, skipping: %s", ckpt_path)
            continue

        model, cfg, best_val_acc = load_model_and_cfg(ckpt_path, device)
        logger.info("Loaded '%s'  (best_val_acc=%.4f)", model_label,
                    best_val_acc if best_val_acc is not None else float("nan"))

        # Build the test loader once (all models share the same ASL test set)
        if test_loader is None:
            if not asl_root.exists():
                logger.error("ASL root not found; cannot run model evaluation.")
                break
            test_ds = ASLEventDataset(
                str(asl_root), split="test",
                resolution=args.resolution, cfg=cfg,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            logger.info("Test set: %d samples", len(test_ds))

        preds, labels = run_inference(model, test_loader, device)
        overall_acc = float((preds == labels).mean())
        logger.info("'%s'  overall acc: %.2f%%", model_label, overall_acc * 100)

        f1_results[model_label] = per_class_f1(preds, labels, NUM_CLASSES)
        cm_results[model_label] = (build_cm(preds, labels, NUM_CLASSES), overall_acc)

    if f1_results:
        plot_per_class_f1(f1_results, out_dir / "per_class_f1.png")

    if cm_results:
        plot_confusion_matrices(cm_results, out_dir / "confusion_matrices.png")

    logger.info("Done. Outputs in %s/", out_dir.resolve())


if __name__ == "__main__":
    main()
