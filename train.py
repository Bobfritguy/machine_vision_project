#!/usr/bin/env python3
"""
train.py – Train a MobileNetV3-Small classifier on the ASL event-camera dataset.

Usage
-----
    # Quickstart (LR, two-channel, 256x256, 30 epochs)
    python train.py --root /big_boy_hdd/ASL_Dataset/ASL

    # Use HR split, voxel-grid representation, custom output dir
    python train.py --root /big_boy_hdd/ASL_Dataset/ASL \\
        --resolution HR --repr voxel_grid --bins 10 \\
        --target-size 256 --epochs 50 --out runs/hr_voxel

    # Resume from checkpoint
    python train.py --root /big_boy_hdd/ASL_Dataset/ASL \\
        --resume runs/lr_twochannel/checkpoint_best.pt

Checkpoints and training curves are saved under --out (default: runs/exp).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

from dataset import ASLEventDataset
from model import build_model, count_parameters
from utils import CLASSES, NUM_CLASSES, PreprocessConfig, get_logger, seed_everything

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data augmentation for event frames
# ---------------------------------------------------------------------------

class EventFrameAugment:
    """
    On-the-fly augmentation for event-frame tensors [C, H, W].

    Designed to bridge the DAVIS240C → GenX320 domain gap by simulating
    variations in scale, aspect ratio, and event density that differ
    between the two sensors.
    """

    def __init__(
        self,
        scale_range: tuple = (0.7, 1.3),
        aspect_jitter: float = 0.15,
        event_dropout: float = 0.2,
        noise_std: float = 0.05,
        flip_lr: bool = False,
    ):
        self.scale_range = scale_range
        self.aspect_jitter = aspect_jitter
        self.event_dropout = event_dropout
        self.noise_std = noise_std
        self.flip_lr = flip_lr

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        C, H, W = tensor.shape

        # Random horizontal flip (disabled by default – some ASL letters
        # are mirror-ambiguous, e.g. d/g).
        if self.flip_lr and torch.rand(1).item() < 0.5:
            tensor = tensor.flip(-1)

        # Random scale + aspect ratio jitter via affine crop-and-resize.
        if torch.rand(1).item() < 0.5:
            scale = self.scale_range[0] + torch.rand(1).item() * (
                self.scale_range[1] - self.scale_range[0]
            )
            aspect = 1.0 + (torch.rand(1).item() * 2 - 1) * self.aspect_jitter

            crop_h = min(int(H / scale * aspect), H)
            crop_w = min(int(W / scale / aspect), W)
            top = torch.randint(0, max(H - crop_h, 1) + 1, (1,)).item()
            left = torch.randint(0, max(W - crop_w, 1) + 1, (1,)).item()

            tensor = tensor[:, top:top + crop_h, left:left + crop_w]
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(H, W), mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Random event dropout – zero out a fraction of pixels to simulate
        # sparser event patterns (GenX320 spreads events over more pixels).
        if self.event_dropout > 0 and torch.rand(1).item() < 0.5:
            mask = torch.rand(1, H, W) > self.event_dropout
            tensor = tensor * mask.float()

        # Additive Gaussian noise – simulates the different noise floor of
        # the GenX320 (smaller pixel pitch → different noise characteristics).
        if self.noise_std > 0 and torch.rand(1).item() < 0.5:
            tensor = tensor + torch.randn_like(tensor) * self.noise_std

        return tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0, num_epochs=0):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", acc=f"{total_acc/n_batches*100:.1f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, num_epochs=0):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]  ", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", acc=f"{total_acc/n_batches*100:.1f}%")
    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ASL event-camera classifier")
    p.add_argument("--root", default="/big_boy_hdd/ASL_Dataset/ASL")
    p.add_argument("--resolution", choices=["LR", "HR"], default="LR")
    p.add_argument("--repr", choices=["two_channel", "signed", "voxel_grid"],
                   default="two_channel")
    p.add_argument("--bins", type=int, default=5, help="Temporal bins for voxel_grid")
    p.add_argument("--target-size", type=int, default=256,
                   help="Model input size (square). Default 256.")
    p.add_argument("--remap-strategy",
                   choices=["remap_then_accumulate", "accumulate_then_resize"],
                   default="accumulate_then_resize")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable data augmentation during training")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Build preprocessing config
    # ------------------------------------------------------------------
    from dataset import _SENSOR_RES
    src_res = _SENSOR_RES[args.resolution]
    cfg = PreprocessConfig(
        source_resolution=src_res,
        target_resolution=(args.target_size, args.target_size),
        representation=args.repr,
        num_bins=args.bins,
        remap_strategy=args.remap_strategy,
    )
    logger.info("PreprocessConfig: %s", cfg.to_dict())

    # Save config for later inference / evaluation
    with open(out_dir / "preprocess_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    train_transform = None if args.no_augment else EventFrameAugment()
    train_ds = ASLEventDataset(args.root, split="train",
                               resolution=args.resolution, cfg=cfg,
                               transform=train_transform)
    val_ds   = ASLEventDataset(args.root, split="test",
                               resolution=args.resolution, cfg=cfg)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = build_model(in_channels=cfg.num_channels, num_classes=NUM_CLASSES)
    model = model.to(device)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", history)
        logger.info("Resumed from epoch %d (best val acc %.2f%%)",
                    start_epoch, best_val_acc * 100)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    logger.info("Starting training for %d epochs", args.epochs)
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, num_epochs=args.epochs)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            epoch=epoch, num_epochs=args.epochs)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc*100:.1f}%",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc*100:.1f}%",
        )
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.2f%%  "
            "val_loss=%.4f  val_acc=%.2f%%  lr=%.2e  time=%.1fs",
            epoch + 1, args.epochs,
            tr_loss, tr_acc * 100,
            val_loss, val_acc * 100,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_val_acc": best_val_acc,
                 "history": history,
                 "preprocess_cfg": cfg.to_dict()},
                out_dir / "checkpoint_best.pt",
            )

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_val_acc": best_val_acc,
                 "history": history,
                 "preprocess_cfg": cfg.to_dict()},
                out_dir / f"checkpoint_epoch{epoch+1:03d}.pt",
            )

    # Save training curves
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best val acc: %.2f%%", best_val_acc * 100)
    logger.info("Artifacts saved to %s", out_dir.resolve())

    # ------------------------------------------------------------------
    # Save training curve plot (if matplotlib available)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs_range, history["train_loss"], label="train")
        axes[0].plot(epochs_range, history["val_loss"],   label="val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss"); axes[0].legend()

        axes[1].plot(epochs_range,
                     [a * 100 for a in history["train_acc"]], label="train")
        axes[1].plot(epochs_range,
                     [a * 100 for a in history["val_acc"]],   label="val")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy"); axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=120)
        plt.close(fig)
        logger.info("Training curves saved.")
    except ImportError:
        logger.warning("matplotlib not installed; skipping curve plot.")


if __name__ == "__main__":
    main()
