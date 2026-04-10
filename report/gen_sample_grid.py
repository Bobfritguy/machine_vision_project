"""Generate a 2x3 grid of ASL sample event frames for the report."""
import sys
sys.path.insert(0, "/home/seamus/Programming/Machine_Vision_Project")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocess import events_to_frame, load_events
from utils import PreprocessConfig

cfg = PreprocessConfig(
    representation="two_channel",
    normalization="log1p",
    source_resolution=(120, 90),
    target_resolution=(256, 256),
    remap_strategy="accumulate_then_resize",
)

CLASSES = ["a", "b", "c", "d", "e", "f"]
DATASET_ROOT = "/big_boy_hdd/ASL_Dataset/ASL/SR_Train/LR"

fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5))
fig.patch.set_facecolor("black")

import os, glob

for ax, cls in zip(axes.flat, CLASSES):
    files = sorted(glob.glob(f"{DATASET_ROOT}/{cls}/{cls}_*.npy"))
    path = files[0]
    events = load_events(path)
    tensor = events_to_frame(events, cfg).numpy()  # [2, H, W]

    # Red = ON (ch0), Blue = OFF (ch1), black background
    on  = tensor[0]   # [H, W], normalised 0–1
    off = tensor[1]

    rgb = np.zeros((*on.shape, 3), dtype=np.float32)
    rgb[..., 0] = on    # red channel
    rgb[..., 2] = off   # blue channel
    rgb = rgb.clip(0, 1)

    ax.imshow(rgb, interpolation="nearest", aspect="equal")
    ax.set_title(cls.upper(), color="white", fontsize=13, fontweight="bold", pad=4)
    ax.axis("off")

fig.suptitle("ASL-DVS Samples (red = ON events, blue = OFF events)",
             color="white", fontsize=10, y=0.02)
fig.tight_layout(rect=[0, 0.04, 1, 1], pad=0.4)
fig.savefig("/home/seamus/Programming/Machine_Vision_Project/report/sample_frames.png",
            dpi=150, bbox_inches="tight", facecolor="black")
print("Saved sample_frames.png")
