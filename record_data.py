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

Web Viewfinder
--------------
Optionally enable a live web interface to preview camera feed before recording:
    python record_data.py --web --web-port 62079

Then open http://localhost:62079 to see:
    - Live event-frame visualization
    - Current event count and event rate
    - Recording status

The web interface runs in background and doesn't interfere with CLI recording control.

Usage
-----
    python record_data.py --out datasets/genx320_recorded \\
        --input-camera-config bias.json \\
        --samples-per-burst 20 --sample-duration-ms 200

    # With web viewfinder:
    python record_data.py --web --out datasets/genx320_recorded \\
        --input-camera-config bias.json
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response
from PIL import Image, ImageDraw

from utils import CLASSES, get_logger

logger = get_logger(__name__)

GENX320_RESOLUTION = (320, 320)


class EventDisplay:
    """
    Persistent event display with temporal decay.

    New events appear as bright white, then fade to grey over successive
    frames, against a black background. Both polarities are rendered
    identically (greyscale).
    """

    def __init__(self, sensor_wh: tuple[int, int], decay: float = 0.6):
        W, H = sensor_wh
        self.W = W
        self.H = H
        # Persistent float32 accumulator — values in [0, 1].
        self.canvas = np.zeros((H, W), dtype=np.float32)
        self.decay = decay

    def update(self, events_nx4: np.ndarray, quality: int) -> bytes:
        """
        Accumulate new events and render a JPEG frame.

        Each call first decays the existing canvas (bright → grey → black),
        then stamps new events as white.
        """
        # Decay existing pixels toward black.
        self.canvas *= self.decay

        # Stamp new events (both polarities) as bright white.
        if len(events_nx4) > 0:
            x = events_nx4[:, 1].astype(np.int32).clip(0, self.W - 1)
            y = events_nx4[:, 2].astype(np.int32).clip(0, self.H - 1)
            # Use event counts with log1p so dense areas are brighter.
            counts = np.zeros((self.H, self.W), dtype=np.float32)
            np.add.at(counts, (y, x), 1.0)
            counts = np.log1p(counts)
            scale = max(float(counts.max()), 1e-6)
            # Merge new events into canvas — take the brighter of old and new.
            self.canvas = np.maximum(self.canvas, counts / scale)

        grey = np.clip(self.canvas * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(grey, mode="L")

        buff = BytesIO()
        img.save(buff, format="JPEG", quality=quality, optimize=False)
        return buff.getvalue()


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_jpeg: Optional[bytes] = None
    status: str = "initializing"
    window_events: int = 0
    event_rate_eps: float = 0.0
    frozen: bool = False  # Freeze display during recording


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
    # Web viewfinder
    p.add_argument("--web", action="store_true", default=False,
                   help="Enable web viewfinder interface")
    p.add_argument("--web-host", default="0.0.0.0",
                   help="Web server host (default: 0.0.0.0)")
    p.add_argument("--web-port", type=int, default=62079,
                   help="Web server port (default: 62079)")
    p.add_argument("--jpeg-quality", type=int, default=70,
                   help="JPEG quality for stream frames (1-95, default 70)")
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


def record_burst(event_queue: queue.Queue, n_samples: int, sample_duration_ms: int,
                 gap_ms: int, delta_t_us: int,
                 state: Optional[SharedState] = None,
                 display: Optional[EventDisplay] = None,
                 args: Optional[argparse.Namespace] = None) -> list[np.ndarray]:
    """
    Record a burst of n_samples from the event queue.

    Returns a list of (N, 4) float32 arrays, one per sample.
    Timestamps within each sample are in microseconds from the camera.
    Optionally updates the display during recording.
    """
    sample_duration_us = sample_duration_ms * 1000
    gap_us = gap_ms * 1000
    samples = []

    accumulator = []
    accum_start_t = None
    in_gap = False
    gap_start_t = None

    while len(samples) < n_samples:
        try:
            # Get events from queue with timeout
            events = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if events is None or len(events) == 0:
            continue

        chunk_t_min = float(events[0, 0])
        chunk_t_max = float(events[-1, 0])

        # Update display during recording if enabled
        if state and display and args:
            jpeg_data = display.update(events, args.jpeg_quality)
            with state.lock:
                state.latest_jpeg = jpeg_data
                state.window_events = len(samples) * (sample_duration_ms // 10)

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

    # Create event queue for threading
    event_queue = queue.Queue(maxsize=100)

    # Initialize web viewfinder if enabled
    state = None
    display = None
    if args.web:
        state = SharedState()
        display = EventDisplay(GENX320_RESOLUTION)
        state.status = "ready"
        
        app = make_app(state)
        web_thread = threading.Thread(
            target=lambda: app.run(host=args.web_host, port=args.web_port,
                                   debug=False, use_reloader=False),
            daemon=True
        )
        web_thread.start()
        print(f"Web viewfinder started at http://{args.web_host}:{args.web_port}")

        # Start background camera thread
        camera_thread = threading.Thread(
            target=run_camera_loop,
            args=(iterator, event_queue, state, display, args),
            daemon=True
        )
        camera_thread.start()

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

            if state:
                state.status = f"ready for '{cls.upper()}'"

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
                    if state:
                        state.status = f"recording '{cls.upper()}'"
                        state.frozen = True
                    
                    print(f"\n  Recording {args.samples_per_burst} samples "
                          f"of '{cls.upper()}'...")
                    print(f"  Hold the gesture steady and move slightly.\n")

                    samples = record_burst(
                        event_queue,
                        n_samples=args.samples_per_burst,
                        sample_duration_ms=args.sample_duration_ms,
                        gap_ms=args.gap_ms,
                        delta_t_us=args.delta_t_us,
                        state=state,
                        display=display,
                        args=args,
                    )

                    if state:
                        state.frozen = False
                        state.status = f"ready for '{cls.upper()}'"

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


def run_camera_loop(iterator, event_queue: queue.Queue, state: SharedState,
                    display: EventDisplay, args: argparse.Namespace) -> None:
    """
    Background thread: continuously read events from camera and put them in queue.
    Updates display at least once per second.
    """
    try:
        last_display_update = time.time()
        last_t = None
        event_count = 0
        
        for events_struct in iterator:
            if events_struct is None or len(events_struct) == 0:
                continue

            events = structured_events_to_nx4(events_struct)
            event_count += len(events)

            # Put events in queue for both display and recording
            event_queue.put(events)

            # Update display at least once per second
            now = time.time()
            if (now - last_display_update >= 1.0 or len(events) > 0) and not state.frozen and display:
                jpeg_data = display.update(events, args.jpeg_quality)
                
                # Calculate event rate
                if last_t is not None and len(events) > 0:
                    dt = (events[-1, 0] - last_t) / 1_000_000
                    rate = event_count / dt if dt > 0 else 0
                else:
                    rate = 0

                with state.lock:
                    state.latest_jpeg = jpeg_data
                    state.window_events = event_count
                    state.event_rate_eps = rate
                
                last_display_update = now

            if len(events) > 0:
                last_t = events[-1, 0]

    except Exception as exc:
        logger.exception("Camera loop error: %s", exc)
        with state.lock:
            state.status = f"error: {exc}"


def make_app(state: SharedState) -> Flask:
    """Create Flask app for web viewfinder."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Record Data - Live Viewfinder</title>
  <style>
    body { font-family: sans-serif; margin: 16px; background: #111; color: #eee; }
    .container { max-width: 600px; margin: 0 auto; }
    .panel { background: #1a1a1a; padding: 12px; border: 1px solid #333; border-radius: 8px; }
    h1 { margin-top: 0; }
    .feed-container { position: relative; width: 100%; margin-bottom: 16px; }
    .feed-container img { width: 100%; display: block; border: 1px solid #333; background: #000; }
    .stats { font-size: 0.9em; }
    .stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2b2b2b; }
    .label { color: #bbb; }
    .value { font-weight: 700; color: #7CFC8A; }
    .status { padding: 8px; background: #0a0a0a; border-radius: 4px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Record Data - Live Viewfinder</h1>
    <div class="panel">
      <div class="feed-container">
        <img id="feed" src="/stream.mjpg" />
      </div>
      <div class="status">
        <div class="stat-row">
          <span class="label">Status:</span>
          <span class="value" id="status">initializing</span>
        </div>
      </div>
      <div class="stats">
        <div class="stat-row">
          <span class="label">Events in frame:</span>
          <span id="events">0</span>
        </div>
        <div class="stat-row">
          <span class="label">Event rate (events/s):</span>
          <span id="rate">0</span>
        </div>
      </div>
    </div>
  </div>
  <script>
    async function updateMetrics() {
      try {
        const resp = await fetch('/metrics');
        const data = await resp.json();
        document.getElementById('status').textContent = data.status;
        document.getElementById('events').textContent = data.window_events;
        document.getElementById('rate').textContent = Math.round(data.event_rate_eps);
      } catch (e) {
        console.error('Metrics fetch error:', e);
      }
    }
    setInterval(updateMetrics, 500);
  </script>
</body>
</html>
"""

    @app.route("/stream.mjpg")
    def mjpeg_stream():
        def generate():
            boundary = b"--frame\r\n"
            while True:
                with state.lock:
                    data = state.latest_jpeg
                if data is None:
                    time.sleep(0.02)
                    continue
                yield boundary
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                yield data
                yield b"\r\n"
                time.sleep(0.02)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/metrics")
    def metrics():
        with state.lock:
            payload = {
                "status": state.status,
                "window_events": state.window_events,
                "event_rate_eps": float(state.event_rate_eps),
            }
        return payload

    return app


if __name__ == "__main__":
    main()
