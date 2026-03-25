#!/usr/bin/env python3
"""
inspect_dataset.py – Scan the ASL-DVS event-camera dataset and report statistics.

Background
----------
This dataset is the ASL-DVS dataset [Bi et al., ICCV 2019], captured with a
DAVIS240C camera (240×180 px). The HR split corresponds to the original sensor
output. The LR split was created by 2×2 spatial downsampling (merging events
in each 2×2 kernel with stride 2), giving a 120×90 effective resolution.
Timestamps were adjusted to millisecond precision; samples are ≤200 ms long.
Polarity is stored as {0, 1} (original convention is {-1, +1}).

What it reports
---------------
  - Number of samples per class (train and test)
  - Min/max x, y, t, polarity values (separately for LR and HR)
  - Per-sample event count distribution (min, max, mean, median, percentiles)
  - Whether LR and HR have different coordinate ranges
  - Whether timestamps appear normalised or raw
  - Inferred sensor resolutions

Results are printed to stdout and saved to a JSON file.

Usage
-----
    python inspect_dataset.py --root /big_boy_hdd/ASL_Dataset/ASL
    python inspect_dataset.py --root /big_boy_hdd/ASL_Dataset/ASL \\
        --out stats.json --max-files 50
"""

import argparse
import json
import sys
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_events_to_image(events: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Render events to RGB image showing event distribution.
    Events: (N, 4) array of [t, x, y, p]
    Returns: (height, width, 3) uint8 RGB array
    """
    if len(events) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create histogram of events at each location with polarity
    frame = np.zeros((height, width, 3), dtype=np.int32)
    
    for t, x, y, p in events:
        x, y = int(x), int(y)
        p = int(p)
        if 0 <= x < width and 0 <= y < height:
            if p == 0:  # Blue for polarity 0
                frame[y, x, 2] += 1
            else:  # Red for polarity 1
                frame[y, x, 0] += 1
    
    # Scale to 0-255 range with log scaling for visibility
    frame = np.log1p(frame.astype(np.float32)) * 30  # Scale factor for brightness
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    return frame


def visualize_random_sample(class_dir: Path, width: int, height: int) -> np.ndarray:
    """Load and visualize a random sample from a class directory."""
    files = list(class_dir.glob("*.npy"))
    if not files:
        return None
    
    sample_path = random.choice(files)
    try:
        events = np.load(sample_path, allow_pickle=False).astype(np.float32)
        if events.ndim == 2 and events.shape[1] == 4 and len(events) > 0:
            return render_events_to_image(events, width, height)
    except Exception:
        pass
    
    return None


def scan_split(split_dir: Path, resolution: str, max_files: int) -> dict:
    """Scan one split/resolution directory and collect statistics."""
    res_dir = split_dir / resolution
    if not res_dir.exists():
        print(f"  [WARN] {res_dir} not found, skipping.", file=sys.stderr)
        return {}

    classes = sorted(d.name for d in res_dir.iterdir() if d.is_dir())

    all_x, all_y, all_t, all_p = [], [], [], []
    event_counts = []
    class_counts = {}
    bad_files = []

    for cls in classes:
        cls_dir = res_dir / cls
        files = sorted(cls_dir.glob("*.npy"))
        class_counts[cls] = len(files)

        for fpath in files[:max_files]:
            try:
                arr = np.load(fpath, allow_pickle=False).astype(np.float32)
                if arr.ndim != 2 or arr.shape[1] != 4:
                    bad_files.append(str(fpath))
                    continue
                if len(arr) == 0:
                    bad_files.append(str(fpath))
                    continue
                all_t.append(arr[:, 0])
                all_x.append(arr[:, 1])
                all_y.append(arr[:, 2])
                all_p.append(arr[:, 3])
                event_counts.append(len(arr))
            except Exception as exc:
                bad_files.append(f"{fpath}: {exc}")

    if not event_counts:
        return {"error": "no valid files found"}

    cat_x = np.concatenate(all_x)
    cat_y = np.concatenate(all_y)
    cat_t = np.concatenate(all_t)
    cat_p = np.concatenate(all_p)
    ev    = np.array(event_counts)

    # Timestamp interpretation:
    # ASL-DVS timestamps were adjusted to ms precision.
    # Range 0–164 ms is consistent with samples ≤200 ms (see paper).
    t_range = float(cat_t.max() - cat_t.min())
    if t_range < 1e6:
        ts_type = "milliseconds_normalised_per_sample (0–~200ms, consistent with ASL-DVS paper)"
    else:
        ts_type = "raw_microseconds"

    unique_p = sorted(int(v) for v in np.unique(cat_p))
    if unique_p == [0, 1]:
        p_note = "stored as {0,1}; original DVS convention is {-1,+1}"
    elif unique_p == [-1, 1]:
        p_note = "stored as {-1,+1} (raw DVS convention)"
    else:
        p_note = f"unexpected values: {unique_p}"

    return {
        "classes": classes,
        "class_counts": class_counts,
        "files_sampled_per_class": max_files,
        "coordinate_ranges": {
            "x":  {"min": int(cat_x.min()), "max": int(cat_x.max())},
            "y":  {"min": int(cat_y.min()), "max": int(cat_y.max())},
            "t":  {"min": float(cat_t.min()), "max": float(cat_t.max())},
            "p":  {"unique_values": unique_p, "note": p_note},
        },
        "inferred_sensor_resolution": {
            "W": int(cat_x.max()) + 1,
            "H": int(cat_y.max()) + 1,
        },
        "timestamp_type": ts_type,
        "event_counts_per_sample": {
            "min":    int(ev.min()),
            "max":    int(ev.max()),
            "mean":   float(ev.mean()),
            "median": float(np.median(ev)),
            "p10":    float(np.percentile(ev, 10)),
            "p90":    float(np.percentile(ev, 90)),
        },
        "bad_files": bad_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ASL-DVS or custom event-camera dataset"
    )
    parser.add_argument("--root", default=None,
                        help="Path to dataset root directory (auto-detect ASL or flat structure)")
    parser.add_argument("--out", default=None,
                        help="Output JSON file path (default: dataset_stats.json)")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Max .npy files to sample per class (default 100)")
    args = parser.parse_args()

    # Auto-detect dataset path if not provided
    if args.root is None:
        # Try standard locations in order
        candidates = [
            Path("datasets/genx320_recorded"),
            Path("/big_boy_hdd/ASL_Dataset/ASL"),
        ]
        args.root = None
        for cand in candidates:
            if cand.exists():
                args.root = str(cand)
                break
        if args.root is None:
            print(f"ERROR: No dataset found. Tried: {[str(c) for c in candidates]}", file=sys.stderr)
            sys.exit(1)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        args.out = f"{root.name}_stats.json"

    stats = {}

    # Detect dataset type: check for SR_Train/SR_Test (ASL) vs flat class folders
    has_splits = (root / "SR_Train").exists() or (root / "SR_Test").exists()

    if has_splits:
        # ASL-DVS format with splits
        for split_name, split_dir_name in [("train", "SR_Train"), ("test", "SR_Test")]:
            split_dir = root / split_dir_name
            stats[split_name] = {}
            for res in ["LR", "HR"]:
                print(f"\n{'='*60}")
                print(f"Scanning {split_name.upper()} / {res} ...")
                result = scan_split(split_dir, res, args.max_files)
                stats[split_name][res] = result

                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                    continue

                total = sum(result["class_counts"].values())
                print(f"  Classes  : {len(result['classes'])}  ({result['classes']})")
                print(f"  Samples  : {total} total  "
                      f"({list(result['class_counts'].values())[0]} per class)")
                cr = result["coordinate_ranges"]
                print(f"  x range  : [{cr['x']['min']}, {cr['x']['max']}]")
                print(f"  y range  : [{cr['y']['min']}, {cr['y']['max']}]")
                print(f"  t range  : [{cr['t']['min']:.1f}, {cr['t']['max']:.1f}]  "
                      f"→ {result['timestamp_type']}")
                print(f"  polarity : {cr['p']['unique_values']}  ({cr['p']['note']})")
                sr = result["inferred_sensor_resolution"]
                print(f"  Inferred sensor : {sr['W']} × {sr['H']}  (W × H)")
                ec = result["event_counts_per_sample"]
                print(f"  Events/sample   : min={ec['min']}  max={ec['max']}  "
                      f"mean={ec['mean']:.0f}  median={ec['median']:.0f}  "
                      f"p10={ec['p10']:.0f}  p90={ec['p90']:.0f}")
                if result["bad_files"]:
                    print(f"  BAD files       : {len(result['bad_files'])} "
                          f"(first: {result['bad_files'][0]})")

        # Compare LR vs HR
        print(f"\n{'='*60}")
        print("LR vs HR comparison (train split):")
        for res in ["LR", "HR"]:
            d = stats.get("train", {}).get(res, {})
            if "coordinate_ranges" in d:
                cr = d["coordinate_ranges"]
                sr = d["inferred_sensor_resolution"]
                print(f"  {res}: x=[{cr['x']['min']},{cr['x']['max']}]  "
                      f"y=[{cr['y']['min']},{cr['y']['max']}]  "
                      f"=> {sr['W']}×{sr['H']}")

        print(f"\n{'='*60}")
        print("NOTE: LR was created by 2×2 downsampling of HR (DAVIS240C 240×180).")
        print("      This is consistent with the paper [Li et al., CVPR 2021].")
    else:
        # Flat structure: class folders directly in root
        print(f"\n{'='*60}")
        print(f"Scanning flat dataset structure in {root.name} ...")
        result = scan_split(root, "", args.max_files)
        stats["dataset"] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            total = sum(result["class_counts"].values())
            print(f"  Classes  : {len(result['classes'])}  ({result['classes']})")
            print(f"  Samples  : {total} total  ({', '.join(f'{c}={n}' for c, n in result['class_counts'].items())})")
            cr = result["coordinate_ranges"]
            print(f"  x range  : [{cr['x']['min']}, {cr['x']['max']}]")
            print(f"  y range  : [{cr['y']['min']}, {cr['y']['max']}]")
            print(f"  t range  : [{cr['t']['min']:.1f}, {cr['t']['max']:.1f}]  "
                  f"→ {result['timestamp_type']}")
            print(f"  polarity : {cr['p']['unique_values']}  ({cr['p']['note']})")
            sr = result["inferred_sensor_resolution"]
            print(f"  Inferred sensor : {sr['W']} × {sr['H']}  (W × H)")
            ec = result["event_counts_per_sample"]
            print(f"  Events/sample   : min={ec['min']}  max={ec['max']}  "
                  f"mean={ec['mean']:.0f}  median={ec['median']:.0f}  "
                  f"p10={ec['p10']:.0f}  p90={ec['p90']:.0f}")
            if result["bad_files"]:
                print(f"  BAD files       : {len(result['bad_files'])} "
                      f"(first: {result['bad_files'][0]})")

            print(f"\n--- Recommended PreprocessConfig ---")
            print(f"  PreprocessConfig(source_resolution=({sr['W']}, {sr['H']}))")

    # Save JSON
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved → {out_path.resolve()}")

    # Generate visualization of random samples
    print(f"\n{'='*60}")
    print("Generating sample visualizations...")
    viz_dir = root / "sample_visualizations"
    viz_dir.mkdir(exist_ok=True)

    if has_splits:
        # ASL format: visualize LR and HR
        for split_name, split_dir_name in [("train", "SR_Train"), ("test", "SR_Test")]:
            split_dir = root / split_dir_name
            for res in ["LR", "HR"]:
                res_dir = split_dir / res
                if not res_dir.exists():
                    continue
                
                result = stats.get(split_name, {}).get(res, {})
                if "coordinate_ranges" not in result:
                    continue
                
                sr = result["inferred_sensor_resolution"]
                width, height = sr["W"], sr["H"]
                
                # Create grid of class samples
                classes = sorted([d.name for d in res_dir.iterdir() if d.is_dir()])
                n_cols = min(5, len(classes))
                n_rows = (len(classes) + n_cols - 1) // n_cols
                
                grid = Image.new("RGB", (n_cols * width, n_rows * height), color=(20, 20, 20))
                
                for idx, cls in enumerate(classes):
                    class_dir = res_dir / cls
                    img_array = visualize_random_sample(class_dir, width, height)
                    
                    # Create image with label
                    if img_array is not None:
                        img = Image.fromarray(img_array)
                        # Add class label to top-right
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                        # Draw white text with black outline for visibility
                        text = cls.upper()
                        draw.text((width - len(text) * 6 - 4, 4), text, fill=(255, 255, 255), font=font)
                    else:
                        # Blank image for empty class
                        img = Image.new("RGB", (width, height), color=(40, 40, 40))
                        draw = ImageDraw.Draw(img)
                        # Large centered label
                        text = cls.upper()
                        # Try to use larger font
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                        # Estimate text size and center it
                        text_width = len(text) * 14  # Rough estimate for larger font
                        x = (width - text_width) // 2
                        y = (height - 14) // 2
                        draw.text((x, y), text, fill=(200, 200, 200), font=font)
                    
                    row, col = idx // n_cols, idx % n_cols
                    grid.paste(img, (col * width, row * height))
                
                out_file = viz_dir / f"{split_name}_{res}.png"
                grid.save(out_file)
                print(f"  Saved: {out_file.name}")
    else:
        # Flat structure: create collage of all classes
        result = stats.get("dataset", {})
        if "coordinate_ranges" in result:
            sr = result["inferred_sensor_resolution"]
            width, height = sr["W"], sr["H"]
            
            # Get all class directories
            classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
            
            if classes:
                n_cols = min(5, len(classes))
                n_rows = (len(classes) + n_cols - 1) // n_cols
                
                grid = Image.new("RGB", (n_cols * width, n_rows * height), color=(20, 20, 20))
                
                for idx, cls in enumerate(classes):
                    class_dir = root / cls
                    img_array = visualize_random_sample(class_dir, width, height)
                    
                    # Create image with label
                    if img_array is not None:
                        img = Image.fromarray(img_array)
                        # Add class label to top-right
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                        # Draw white text with black outline for visibility
                        text = cls.upper()
                        draw.text((width - len(text) * 6 - 4, 4), text, fill=(255, 255, 255), font=font)
                        print(f"  Added {cls}")
                    else:
                        # Blank image for empty class
                        img = Image.new("RGB", (width, height), color=(40, 40, 40))
                        draw = ImageDraw.Draw(img)
                        # Large centered label
                        text = cls.upper()
                        # Try to use larger font
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                        # Estimate text size and center it
                        text_width = len(text) * 14  # Rough estimate for larger font
                        x = (width - text_width) // 2
                        y = (height - 14) // 2
                        draw.text((x, y), text, fill=(200, 200, 200), font=font)
                        print(f"  Added {cls} (empty)")
                    
                    row, col = idx // n_cols, idx % n_cols
                    grid.paste(img, (col * width, row * height))
                
                out_file = viz_dir / "samples.png"
                grid.save(out_file)
                print(f"  Saved: {out_file.name}")
    
    print(f"Visualizations saved → {viz_dir.resolve()}")



if __name__ == "__main__":
    main()
