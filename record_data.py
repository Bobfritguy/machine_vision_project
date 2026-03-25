#!/usr/bin/env python3
"""
record_data.py – Record labelled ASL gesture samples from the Prophesee GenX320.

Records event data in the same format as the ASL-DVS dataset: individual .npy
files per sample with shape (N, 4) and columns [t, x, y, p], saved at the
GenX320's native 320×320 resolution.

Recording workflow
------------------
The script cycles through ASL letters. For each letter:
  1. The current target letter is displayed on screen.
  2. You hold the gesture and press SPACE to start recording.
  3. The script records a burst of samples back-to-back, each ~200ms long
     (matching inference window), with a short gap between them.
  4. Press SPACE again to stop early, or let it finish the burst.
  5. Press ENTER/RIGHT to advance to the next letter, or type a letter to jump.
  6. Press Q to quit and save.

Output structure (mirrors ASL-DVS):
    <out_dir>/
      a/
        a_0001.npy
        a_0002.npy
        ...
      b/
        b_0001.npy
        ...

Usage
-----
    python record_data.py --out datasets/genx320_recorded \\
        --input-camera-config bias.json \\
        --samples-per-burst 20 --sample-duration-ms 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from utils import CLASSES, get_logger

logger = get_logger(__name__)

GENX320_RESOLUTION = (320, 320)


def resolve_events_iterator():
    try:
        from metavision_core.event_io import EventsIterator
        return EventsIterator
    except ImportError:
        pass
    try:
        from metavision_core.event_io.events_iterator import EventsIterator
        return EventsIterator
    except ImportError as exc:
        return None


def structured_events_to_nx4(events_struct: np.ndarray) -> np.ndarray:
    if events_struct.dtype.names:
        return np.stack(
            [events_struct["t"], events_struct["x"],
             events_struct["y"], events_struct["p"]],
            axis=1,
        ).astype(np.float32, copy=False)
    if events_struct.ndim == 2 and events_struct.shape[1] == 4:
        return events_struct.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported events: {events_struct.shape}, {events_struct.dtype}")


def get_existing_count(class_dir: Path) -> int:
    """Count existing .npy files in a class directory."""
    if not class_dir.exists():
        return 0
    return sum(1 for f in class_dir.iterdir() if f.suffix == ".npy")


def save_sample(events: np.ndarray, class_dir: Path, cls: str, idx: int) -> Path:
    """Save a single sample as .npy with timestamps zeroed to start at 0."""
    class_dir.mkdir(parents=True, exist_ok=True)
    # Zero timestamps to start at 0 (matches ASL-DVS convention).
    events = events.copy()
    events[:, 0] -= events[:, 0].min()
    # Save as int32 to match dataset dtype.
    out = events.astype(np.int32)
    fname = class_dir / f"{cls}_{idx:04d}.npy"
    np.save(fname, out)
    return fname


def print_status(cls: str, cls_idx: int, total_classes: int,
                 existing: int, recorded_this_session: int):
    """Print the current recording status."""
    print(f"\n{'='*60}")
    print(f"  Letter: {cls.upper()}   ({cls_idx+1}/{total_classes})")
    print(f"  Existing samples: {existing}")
    print(f"  Recorded this session: {recorded_this_session}")
    print(f"{'='*60}")
    print(f"  SPACE  = start/stop recording burst")
    print(f"  ENTER  = next letter")
    print(f"  a-y    = jump to letter")
    print(f"  q      = quit")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="Record labelled ASL samples from GenX320")
    p.add_argument("--out", default="datasets/genx320_recorded",
                   help="Output directory for recorded samples")
    p.add_argument("--input-path", default="",
                   help="Metavision input path (empty = live camera)")
    p.add_argument("--input-camera-config", default="",
                   help="Path to bias config JSON")
    p.add_argument("--sample-duration-ms", type=int, default=200,
                   help="Duration of each sample in milliseconds (default: 200)")
    p.add_argument("--samples-per-burst", type=int, default=20,
                   help="Number of samples to record per burst (default: 20)")
    p.add_argument("--gap-ms", type=int, default=50,
                   help="Gap between samples within a burst in ms (default: 50)")
    p.add_argument("--delta-t-us", type=int, default=10000,
                   help="Camera read chunk size in us (default: 10000 = 10ms)")
    p.add_argument("--start-letter", default="a",
                   help="Letter to start recording from (default: a)")
    # HW filters
    p.add_argument("--stc-threshold-us", type=int, default=None)
    p.add_argument("--erc-rate", type=int, default=None)
    p.add_argument("--afk-frequency", type=int, default=None, choices=[50, 60])
    return p.parse_args()


def setup_camera(args):
    """Initialize camera with optional bias config and HW filters."""
    EventsIterator = resolve_events_iterator()
    if EventsIterator is None:
        print("ERROR: Cannot import Metavision SDK.", file=sys.stderr)
        sys.exit(1)

    hw_filters = any([args.stc_threshold_us, args.erc_rate, args.afk_frequency])
    need_hal = args.input_camera_config or hw_filters

    if need_hal:
        from metavision_core.event_io.raw_reader import initiate_device
        device = initiate_device(path=args.input_path)

        if args.input_camera_config:
            cfg_path = str(Path(args.input_camera_config).expanduser().resolve())
            i_ll_biases = device.get_i_ll_biases()
            with open(cfg_path) as f:
                cfg = json.load(f)
            biases = cfg.get("ll_biases_state", {}).get("bias", [])
            for item in biases:
                name, value = item.get("name"), item.get("value")
                if name is None or value is None:
                    continue
                for meth_name in ("set", "set_bias", "set_bias_value"):
                    if hasattr(i_ll_biases, meth_name):
                        try:
                            getattr(i_ll_biases, meth_name)(str(name), int(value))
                            break
                        except TypeError:
                            continue
            logger.info("Applied biases from %s", cfg_path)

        # Apply HW filters using the same logic as infer_web_live.py
        if args.stc_threshold_us is not None:
            try:
                nf = device.get_i_noise_filter_module()
                if nf:
                    try:
                        nf.enable_trail(args.stc_threshold_us)
                    except AttributeError:
                        nf.set_stc_threshold(args.stc_threshold_us)
                        nf.enable_stc(True)
                    logger.info("STC enabled: %d us", args.stc_threshold_us)
            except (AttributeError, RuntimeError) as exc:
                logger.warning("STC failed: %s", exc)

        if args.erc_rate is not None:
            try:
                erc = device.get_i_erc_module()
                if erc:
                    try:
                        erc.set_cd_event_rate(args.erc_rate)
                    except AttributeError:
                        erc.set_event_rate(args.erc_rate)
                    erc.enable(True)
                    logger.info("ERC enabled: %d ev/s", args.erc_rate)
            except (AttributeError, RuntimeError) as exc:
                logger.warning("ERC failed: %s", exc)

        if args.afk_frequency is not None:
            try:
                afk = device.get_i_antiflicker_module()
                if afk:
                    try:
                        afk.set_frequency_band(args.afk_frequency, args.afk_frequency)
                    except (AttributeError, TypeError):
                        try:
                            afk.set_frequency(args.afk_frequency)
                        except AttributeError:
                            afk.set_filtering_mode(args.afk_frequency)
                    afk.enable(True)
                    logger.info("AFK enabled: %d Hz", args.afk_frequency)
            except (AttributeError, RuntimeError) as exc:
                logger.warning("AFK failed: %s", exc)

        iterator = EventsIterator.from_device(device=device, delta_t=args.delta_t_us)
    else:
        iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t_us)

    return iterator


def record_burst(iterator, n_samples: int, sample_duration_ms: int,
                 gap_ms: int, delta_t_us: int) -> list[np.ndarray]:
    """
    Record a burst of n_samples from the camera.

    Returns a list of (N, 4) float32 arrays, one per sample.
    Timestamps within each sample are in microseconds from the camera.
    """
    sample_duration_us = sample_duration_ms * 1000
    gap_us = gap_ms * 1000
    samples = []

    accumulator = []
    accum_start_t = None
    in_gap = False
    gap_start_t = None

    for events_struct in iterator:
        if events_struct is None or len(events_struct) == 0:
            continue

        events = structured_events_to_nx4(events_struct)
        chunk_t_min = float(events[0, 0])
        chunk_t_max = float(events[-1, 0])

        if in_gap:
            # Wait for gap to elapse before starting next sample.
            if chunk_t_max - gap_start_t >= gap_us:
                in_gap = False
                accumulator = []
                accum_start_t = None
            continue

        if accum_start_t is None:
            accum_start_t = chunk_t_min

        accumulator.append(events)
        elapsed = chunk_t_max - accum_start_t

        if elapsed >= sample_duration_us:
            # Concatenate and trim to exact duration.
            all_events = np.concatenate(accumulator, axis=0)
            mask = (all_events[:, 0] - accum_start_t) <= sample_duration_us
            sample = all_events[mask]

            if len(sample) > 0:
                samples.append(sample)
                print(f"    Sample {len(samples)}/{n_samples}: "
                      f"{len(sample)} events, "
                      f"{(sample[-1,0]-sample[0,0])/1000:.0f}ms", flush=True)

            if len(samples) >= n_samples:
                break

            # Enter gap before next sample.
            in_gap = True
            gap_start_t = chunk_t_max
            accumulator = []
            accum_start_t = None

    return samples


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing camera...")
    iterator = setup_camera(args)
    print("Camera ready.\n")

    # Figure out starting letter.
    if args.start_letter.lower() in CLASSES:
        cls_idx = CLASSES.index(args.start_letter.lower())
    else:
        cls_idx = 0

    import tty
    import termios
    old_settings = termios.tcgetattr(sys.stdin)

    session_counts = {c: 0 for c in CLASSES}

    try:
        tty.setcbreak(sys.stdin.fileno())

        while cls_idx < len(CLASSES):
            cls = CLASSES[cls_idx]
            class_dir = out_dir / cls
            existing = get_existing_count(class_dir)

            print_status(cls, cls_idx, len(CLASSES), existing,
                         session_counts[cls])

            # Wait for keypress.
            while True:
                ch = sys.stdin.read(1)

                if ch == 'q':
                    print("\nQuitting.")
                    _print_summary(session_counts, out_dir)
                    return

                if ch == ' ':
                    # Start recording burst.
                    next_idx = existing + session_counts[cls] + 1
                    print(f"\n  Recording {args.samples_per_burst} samples "
                          f"of '{cls.upper()}'...")
                    print(f"  Hold the gesture steady and move slightly.\n")

                    samples = record_burst(
                        iterator,
                        n_samples=args.samples_per_burst,
                        sample_duration_ms=args.sample_duration_ms,
                        gap_ms=args.gap_ms,
                        delta_t_us=args.delta_t_us,
                    )

                    for i, sample in enumerate(samples):
                        idx = next_idx + i
                        path = save_sample(sample, class_dir, cls, idx)
                    session_counts[cls] += len(samples)
                    print(f"\n  Saved {len(samples)} samples for '{cls.upper()}'.")
                    print_status(cls, cls_idx, len(CLASSES), existing,
                                 session_counts[cls])

                elif ch in ('\n', '\r'):
                    # Next letter.
                    cls_idx += 1
                    break

                elif ch.lower() in CLASSES:
                    # Jump to specific letter.
                    cls_idx = CLASSES.index(ch.lower())
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        _print_summary(session_counts, out_dir)


def _print_summary(session_counts: dict, out_dir: Path):
    total = sum(session_counts.values())
    if total == 0:
        print("No samples recorded.")
        return
    print(f"\n{'='*60}")
    print(f"  Recording session summary")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*60}")
    for cls in CLASSES:
        n = session_counts[cls]
        if n > 0:
            print(f"  {cls.upper()}: {n} samples")
    print(f"  Total: {total} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
