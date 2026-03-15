"""
preprocess.py – Event-to-image conversion for the ASL event-camera pipeline.

All public API functions accept a raw event array with shape (N, 4) and
columns [t, x, y, p], plus a PreprocessConfig, and return a float32 torch
Tensor of shape [C, H, W].

Supported representations
--------------------------
  two_channel   : [pos_counts, neg_counts]           C=2
  signed        : [pos_counts - neg_counts]           C=1
  voxel_grid    : temporal bins of signed counts      C=num_bins

The module also provides:
  events_to_frame()   : main entry-point for a single event array
  validate_events()   : sanity-checks an event array and normalises polarity
  remap_coords()      : rescale (x,y) from one sensor size to another
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from utils import PreprocessConfig


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_events(events: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Validate and normalise a raw event array.

    Parameters
    ----------
    events : np.ndarray, shape (N, 4), columns [t, x, y, p]
    cfg    : PreprocessConfig

    Returns
    -------
    events : np.ndarray, float32, same shape – polarity normalised to {0,1}

    Raises
    ------
    ValueError  on obviously malformed input
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(
            f"Expected events shape (N, 4), got {events.shape}"
        )
    if len(events) == 0:
        raise ValueError("Event array is empty.")

    events = events.astype(np.float32)
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    W, H = cfg.source_resolution
    if x.max() >= W or y.max() >= H:
        raise ValueError(
            f"Event coordinates ({x.max():.0f}, {y.max():.0f}) exceed "
            f"declared source_resolution {cfg.source_resolution}."
        )

    # Normalise polarity to {0, 1}
    unique_p = set(np.unique(p).tolist())
    if unique_p <= {-1.0, 1.0}:
        # Convert {-1, +1} -> {0, 1}
        events[:, 3] = (p + 1.0) / 2.0
    elif unique_p <= {0.0, 1.0}:
        pass  # already in {0, 1}
    else:
        raise ValueError(f"Unexpected polarity values: {unique_p}")

    return events


# ---------------------------------------------------------------------------
# Coordinate remapping
# ---------------------------------------------------------------------------

def remap_coords(
    events: np.ndarray,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    """
    Scale (x, y) coordinates from src_wh to dst_wh.

    Uses floor mapping so that coordinates stay integer after rounding.
    A copy is returned; the input is not modified.
    """
    events = events.copy()
    sw, sh = src_wh
    dw, dh = dst_wh
    events[:, 1] = np.floor(events[:, 1] * dw / sw).clip(0, dw - 1)
    events[:, 2] = np.floor(events[:, 2] * dh / sh).clip(0, dh - 1)
    return events


# ---------------------------------------------------------------------------
# Core accumulation functions
# ---------------------------------------------------------------------------

def _accumulate_two_channel(
    events: np.ndarray, W: int, H: int
) -> np.ndarray:
    """Return float32 array [2, H, W]: [pos_counts, neg_counts]."""
    frame = np.zeros((2, H, W), dtype=np.float32)
    x = events[:, 1].astype(np.int32).clip(0, W - 1)
    y = events[:, 2].astype(np.int32).clip(0, H - 1)
    p = events[:, 3].astype(np.int32)  # 0 or 1 after validate_events

    pos_mask = p == 1
    neg_mask = p == 0
    np.add.at(frame[0], (y[pos_mask], x[pos_mask]), 1)  # channel 0: positive
    np.add.at(frame[1], (y[neg_mask], x[neg_mask]), 1)  # channel 1: negative
    return frame


def _accumulate_signed(
    events: np.ndarray, W: int, H: int
) -> np.ndarray:
    """Return float32 array [1, H, W]: positive - negative counts."""
    frame = np.zeros((1, H, W), dtype=np.float32)
    x = events[:, 1].astype(np.int32).clip(0, W - 1)
    y = events[:, 2].astype(np.int32).clip(0, H - 1)
    p = events[:, 3]

    signed = np.where(p == 1, 1.0, -1.0)
    np.add.at(frame[0], (y, x), signed)
    return frame


def _accumulate_voxel_grid(
    events: np.ndarray, W: int, H: int, num_bins: int
) -> np.ndarray:
    """
    Return float32 array [num_bins, H, W].

    Each event is distributed into at most two adjacent temporal bins using
    bilinear interpolation of the bin weight (standard voxel-grid method).
    Timestamps are normalised to [0, num_bins-1] per sample.
    """
    frame = np.zeros((num_bins, H, W), dtype=np.float32)
    x = events[:, 1].astype(np.int32).clip(0, W - 1)
    y = events[:, 2].astype(np.int32).clip(0, H - 1)
    p = events[:, 3]
    t = events[:, 0].astype(np.float32)

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        t_norm = np.zeros_like(t)

    signed = np.where(p == 1, 1.0, -1.0)

    t0 = np.floor(t_norm).astype(np.int32).clip(0, num_bins - 1)
    t1 = (t0 + 1).clip(0, num_bins - 1)
    w1 = (t_norm - t0).clip(0, 1)  # weight toward t1
    w0 = 1.0 - w1

    np.add.at(frame, (t0, y, x), signed * w0)
    np.add.at(frame, (t1, y, x), signed * w1)
    return frame


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(tensor: torch.Tensor, method: str) -> torch.Tensor:
    """Apply per-frame normalisation."""
    if method == "log1p":
        # log1p(|x|) * sign(x) – handles both positive-only and signed maps
        return torch.sign(tensor) * torch.log1p(torch.abs(tensor))
    if method == "minmax":
        lo = tensor.min()
        hi = tensor.max()
        if hi > lo:
            return (tensor - lo) / (hi - lo)
        return torch.zeros_like(tensor)
    if method == "none":
        return tensor
    raise ValueError(f"Unknown normalization: {method!r}")


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def _resize(tensor: torch.Tensor, target_wh: tuple[int, int]) -> torch.Tensor:
    """Bicubic resize [C, H, W] -> [C, target_H, target_W]."""
    tw, th = target_wh
    if tensor.shape[-2] == th and tensor.shape[-1] == tw:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),       # [1, C, H, W]
        size=(th, tw),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)                   # [C, H, W]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def events_to_frame(
    events: np.ndarray,
    cfg: PreprocessConfig,
) -> torch.Tensor:
    """
    Convert a raw event array to a normalised image tensor.

    Parameters
    ----------
    events : np.ndarray, shape (N, 4), columns [t, x, y, p]
    cfg    : PreprocessConfig

    Returns
    -------
    torch.Tensor, float32, shape [C, target_H, target_W]

    Strategy
    --------
    remap_then_accumulate:
        1. Scale x,y to target_resolution
        2. Accumulate counts at target resolution
        3. Normalise
        → No resize step needed; avoids interpolation.

    accumulate_then_resize (default):
        1. Accumulate counts at source_resolution
        2. Normalise
        3. Bicubic resize to target_resolution
        → Better spatial fidelity; gentle interpolation blur.
    """
    # Validate and normalise polarity
    events = validate_events(events, cfg)

    sw, sh = cfg.source_resolution
    tw, th = cfg.target_resolution

    if cfg.remap_strategy == "remap_then_accumulate":
        events_work = remap_coords(events, (sw, sh), (tw, th))
        acc_W, acc_H = tw, th
    else:  # accumulate_then_resize
        events_work = events
        acc_W, acc_H = sw, sh

    # Accumulate
    if cfg.representation == "two_channel":
        frame = _accumulate_two_channel(events_work, acc_W, acc_H)
    elif cfg.representation == "signed":
        frame = _accumulate_signed(events_work, acc_W, acc_H)
    elif cfg.representation == "voxel_grid":
        frame = _accumulate_voxel_grid(events_work, acc_W, acc_H, cfg.num_bins)
    else:
        raise ValueError(f"Unknown representation: {cfg.representation!r}")

    tensor = torch.from_numpy(frame)      # [C, H, W]
    tensor = _normalize(tensor, cfg.normalization)

    if cfg.remap_strategy == "accumulate_then_resize":
        tensor = _resize(tensor, (tw, th))

    return tensor.float()


def load_events(path: str) -> np.ndarray:
    """
    Load a .npy event file.

    Returns
    -------
    np.ndarray, shape (N, 4), columns [t, x, y, p]
    Raises ValueError for empty or malformed files.
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Cannot load {path!r}: {exc}") from exc

    if arr.ndim == 1 and arr.dtype.names:
        # Structured array – convert to plain (N, 4) assuming t,x,y,p fields
        try:
            arr = np.stack(
                [arr[n] for n in ("t", "x", "y", "p")], axis=1
            ).astype(np.float32)
        except KeyError as exc:
            raise ValueError(
                f"Structured array in {path!r} missing expected field: {exc}"
            ) from exc

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"Expected shape (N, 4) in {path!r}, got {arr.shape}"
        )
    if len(arr) == 0:
        raise ValueError(f"Empty event array in {path!r}")

    return arr.astype(np.float32)
