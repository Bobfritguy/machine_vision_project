#!/usr/bin/env python3
"""
infer_web_live.py – Live web dashboard for Prophesee GENX320 inference.

Runs a local HTTP server (default port 62078) that shows:
  - live event-frame visualisation
  - top prediction and confidence
  - debugging metrics (event counts/rate, inference latency, FPS, etc.)

Camera events are read via Metavision EventsIterator and converted to
shape (N, 4) float32 arrays with columns [t, x, y, p].
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify
from PIL import Image

from infer_live import GENX320_RESOLUTION, LiveInferencer
from utils import get_logger

logger = get_logger(__name__)


def resolve_events_iterator():
    """
    Return EventsIterator class from Metavision bindings using known import paths.
    """
    last_exc = None
    try:
        from metavision_core.event_io import EventsIterator  # type: ignore
        return EventsIterator
    except ImportError as exc:
        last_exc = exc

    try:
        from metavision_core.event_io.events_iterator import EventsIterator  # type: ignore
        return EventsIterator
    except ImportError as exc:
        last_exc = exc

    return None, last_exc


def metavision_import_diagnostics() -> str:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    mod = importlib.util.find_spec("metavision_core")
    return (
        f"python={sys.executable} "
        f"venv={in_venv} "
        f"metavision_core_found={mod is not None} "
        f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}"
    )


def structured_events_to_nx4(events_struct: np.ndarray) -> np.ndarray:
    """
    Convert Metavision structured events to float32 [N, 4] as [t, x, y, p].
    """
    if events_struct.dtype.names:
        required = {"t", "x", "y", "p"}
        names = set(events_struct.dtype.names)
        if not required.issubset(names):
            raise ValueError(f"Expected event fields {required}, got {names}")
        return np.stack(
            [events_struct["t"], events_struct["x"], events_struct["y"], events_struct["p"]],
            axis=1,
        ).astype(np.float32, copy=False)

    if events_struct.ndim == 2 and events_struct.shape[1] == 4:
        return events_struct.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported event array shape/dtype: {events_struct.shape}, {events_struct.dtype}")


def render_events_jpeg(events_nx4: np.ndarray, sensor_wh: tuple[int, int], quality: int) -> bytes:
    """
    Render a color event frame as JPEG bytes:
      - red channel: positive events
      - blue channel: negative events
    """
    W, H = sensor_wh
    pos = np.zeros((H, W), dtype=np.float32)
    neg = np.zeros((H, W), dtype=np.float32)

    x = events_nx4[:, 1].astype(np.int32).clip(0, W - 1)
    y = events_nx4[:, 2].astype(np.int32).clip(0, H - 1)
    p = events_nx4[:, 3]

    pos_mask = p > 0
    neg_mask = ~pos_mask
    np.add.at(pos, (y[pos_mask], x[pos_mask]), 1.0)
    np.add.at(neg, (y[neg_mask], x[neg_mask]), 1.0)

    pos = np.log1p(pos)
    neg = np.log1p(neg)

    scale = max(float(pos.max()), float(neg.max()), 1e-6)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((pos / scale) * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((neg / scale) * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    buff = BytesIO()
    img.save(buff, format="JPEG", quality=quality, optimize=False)
    return buff.getvalue()


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_jpeg: Optional[bytes] = None
    last_update_s: float = 0.0
    status: str = "starting"
    prediction: str = "-"
    confidence: float = 0.0
    topk: list[tuple[str, float]] = field(default_factory=list)
    window_events: int = 0
    event_rate_eps: float = 0.0
    inference_ms: float = 0.0
    inference_fps: float = 0.0
    total_windows: int = 0
    errors: int = 0
    message: str = ""


def parse_args():
    p = argparse.ArgumentParser(description="Live Prophesee X320 inference web dashboard")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=62078)
    p.add_argument("--delta-t-us", type=int, default=50000,
                   help="Camera integration window in microseconds (default 50000 = 50ms)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--crop-to-training-aspect", action="store_true")
    p.add_argument("--input-path", default="",
                   help="Metavision input path. Leave empty to open the live camera.")
    p.add_argument(
        "--input-camera-config",
        default="",
        help="Path to Metavision camera config JSON (same format as metavision_viewer --input-camera-config).",
    )
    p.add_argument("--jpeg-quality", type=int, default=70,
                   help="JPEG quality for stream frames (1-95, default 70).")
    return p.parse_args()


def make_app(state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>X320 Live ASL Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 16px; background: #111; color: #eee; }
    .wrap { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
    img { width: 100%; border: 1px solid #333; background: #000; }
    .panel { background: #1a1a1a; padding: 12px; border: 1px solid #333; border-radius: 8px; }
    .big { font-size: 2rem; font-weight: 700; }
    .muted { color: #bbb; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    td { padding: 4px 0; border-bottom: 1px solid #2b2b2b; }
    .ok { color: #7CFC8A; }
    .warn { color: #ffcc66; }
  </style>
</head>
<body>
  <h2>X320 Live ASL Dashboard</h2>
  <div class="wrap">
    <div class="panel">
      <img id="feed" src="/stream.mjpg" />
    </div>
    <div class="panel">
      <div>Status: <span id="status" class="muted">starting</span></div>
      <div class="big" id="pred">-</div>
      <div>Confidence: <span id="conf">0.00%</span></div>
      <table>
        <tr><td>Window events</td><td id="we">0</td></tr>
        <tr><td>Event rate (events/s)</td><td id="er">0</td></tr>
        <tr><td>Inference latency (ms)</td><td id="ims">0</td></tr>
        <tr><td>Inference FPS</td><td id="ifps">0</td></tr>
        <tr><td>Total windows</td><td id="tw">0</td></tr>
        <tr><td>Errors</td><td id="err">0</td></tr>
        <tr><td>Message</td><td id="msg">-</td></tr>
      </table>
      <div style="margin-top:8px;">
        <strong>Top-k</strong>
        <div id="topk" class="muted">-</div>
      </div>
    </div>
  </div>
  <script>
    async function tick() {
      try {
        const m = await fetch('/metrics').then(r => r.json());
        document.getElementById('status').textContent = m.status;
        document.getElementById('pred').textContent = m.prediction;
        document.getElementById('conf').textContent = (m.confidence * 100).toFixed(2) + '%';
        document.getElementById('we').textContent = m.window_events;
        document.getElementById('er').textContent = m.event_rate_eps.toFixed(0);
        document.getElementById('ims').textContent = m.inference_ms.toFixed(2);
        document.getElementById('ifps').textContent = m.inference_fps.toFixed(2);
        document.getElementById('tw').textContent = m.total_windows;
        document.getElementById('err').textContent = m.errors;
        document.getElementById('msg').textContent = m.message || '-';
        document.getElementById('topk').textContent =
          (m.topk || []).map(([c, s]) => `${c}:${(s*100).toFixed(2)}%`).join('  ');
      } catch (e) {}
    }
    setInterval(tick, 500);
    tick();
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
                "last_update_s": state.last_update_s,
                "status": state.status,
                "prediction": state.prediction,
                "confidence": state.confidence,
                "topk": state.topk,
                "window_events": state.window_events,
                "event_rate_eps": state.event_rate_eps,
                "inference_ms": state.inference_ms,
                "inference_fps": state.inference_fps,
                "total_windows": state.total_windows,
                "errors": state.errors,
                "message": state.message,
            }
        return jsonify(payload)

    return app


def run_camera_loop(args, state: SharedState) -> None:
    """
    Background loop:
      1) reads camera event windows from Metavision,
      2) runs model inference,
      3) updates latest frame and dashboard metrics.
    """
    resolved = resolve_events_iterator()
    if isinstance(resolved, tuple):
        EventsIterator, import_exc = resolved
    else:
        EventsIterator, import_exc = resolved, None

    if EventsIterator is None:
        diag = metavision_import_diagnostics()
        hint = "Install h5py in this environment (pip install h5py)." if isinstance(import_exc, ModuleNotFoundError) and str(getattr(import_exc, "name", "")) == "h5py" else ""
        with state.lock:
            state.status = "error"
            state.errors += 1
            state.message = (
                "Cannot import Metavision Python bindings. "
                "Use the same interpreter where Metavision is installed. "
                f"Diagnostics: {diag} ImportError={import_exc!r} {hint}".strip()
            )
        logger.error(state.message)
        return

    inferencer = LiveInferencer(
        checkpoint_path=args.checkpoint,
        crop_to_training_aspect=args.crop_to_training_aspect,
    )

    with state.lock:
        state.status = "running"
        state.message = "Camera loop started."

    iterator_kwargs = {"input_path": args.input_path, "delta_t": args.delta_t_us}
    iterator = None
    device = None
    try:
        if args.input_camera_config:
            cfg_path = str(Path(args.input_camera_config).expanduser().resolve())

            # First try direct EventsIterator kwargs (works on some SDK builds).
            iterator_candidates = [
                {"input_camera_config": cfg_path},
                {"camera_config": cfg_path},
                {"bias_file": cfg_path},
            ]
            last_exc = None
            for extra in iterator_candidates:
                try:
                    iterator = EventsIterator(**iterator_kwargs, **extra)
                    with state.lock:
                        state.message = f"Using camera config via iterator kwarg: {cfg_path}"
                    break
                except TypeError as exc:
                    last_exc = exc

            # If kwargs are not supported, apply biases through HAL device.
            if iterator is None:
                from metavision_core.event_io.raw_reader import initiate_device  # type: ignore
                device = initiate_device(path=args.input_path)
                i_ll_biases = device.get_i_ll_biases()
                if i_ll_biases is None:
                    raise RuntimeError("Device does not expose i_ll_biases facility.")

                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                biases = cfg.get("ll_biases_state", {}).get("bias", [])
                if not isinstance(biases, list) or not biases:
                    raise RuntimeError("Camera config has no ll_biases_state.bias entries.")

                applied = 0
                for item in biases:
                    name = item.get("name")
                    value = item.get("value")
                    if name is None or value is None:
                        continue

                    ok = False
                    for meth_name in ("set", "set_bias", "set_bias_value"):
                        if not hasattr(i_ll_biases, meth_name):
                            continue
                        meth = getattr(i_ll_biases, meth_name)
                        try:
                            ret = meth(str(name), int(value))
                            ok = (ret is None) or bool(ret)
                            if ok:
                                break
                        except TypeError:
                            continue
                    if not ok:
                        raise RuntimeError(f"Failed to apply bias {name}={value}.")
                    applied += 1

                if not hasattr(EventsIterator, "from_device"):
                    raise RuntimeError("EventsIterator.from_device is unavailable in this SDK.")
                iterator = EventsIterator.from_device(device=device, delta_t=args.delta_t_us)
                with state.lock:
                    state.message = f"Applied {applied} biases from {cfg_path} via HAL device."

        else:
            iterator = EventsIterator(**iterator_kwargs)

    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.errors += 1
            state.message = f"Camera init failed: {exc}"
        logger.exception("Camera initialization failed")
        return

    window_s = args.delta_t_us / 1e6

    try:
        for events_struct in iterator:
            if events_struct is None:
                continue

            events = structured_events_to_nx4(events_struct)
            n = int(len(events))
            if n == 0:
                continue

            start = time.perf_counter()
            pred = inferencer.predict(events)
            infer_ms = (time.perf_counter() - start) * 1000.0

            sorted_scores = sorted(
                pred["scores"].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            topk = sorted_scores[:args.top_k]

            frame_jpeg = render_events_jpeg(
                events,
                sensor_wh=GENX320_RESOLUTION,
                quality=max(1, min(95, args.jpeg_quality)),
            )

            with state.lock:
                state.latest_jpeg = frame_jpeg
                state.last_update_s = time.time()
                state.prediction = pred["predicted_class"]
                state.confidence = float(pred["confidence"])
                state.topk = [(k, float(v)) for k, v in topk]
                state.window_events = n
                state.event_rate_eps = n / window_s
                state.inference_ms = infer_ms
                state.inference_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
                state.total_windows += 1
                state.status = "running"
                state.message = ""
    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.errors += 1
            state.message = f"Camera loop failed: {exc}"
        logger.exception("Camera loop failed")


def main() -> None:
    args = parse_args()
    state = SharedState()

    worker = threading.Thread(target=run_camera_loop, args=(args, state), daemon=True)
    worker.start()

    app = make_app(state)
    logger.info("Starting dashboard on http://%s:%d", args.host, args.port)
    logger.info("Camera source: %r  |  delta_t=%dus", args.input_path, args.delta_t_us)
    if args.input_camera_config:
        logger.info("Camera config: %s", args.input_camera_config)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
