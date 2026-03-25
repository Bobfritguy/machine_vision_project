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
from flask import Flask, Response, jsonify, request
from PIL import Image, ImageDraw

from infer_live import GENX320_RESOLUTION, LiveInferencer
from utils import CLASSES, get_logger

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

    def update(self, events_nx4: np.ndarray, quality: int,
               roi: Optional[tuple[int, int, int, int]] = None) -> bytes:
        """
        Accumulate new events and render a JPEG frame.

        Each call first decays the existing canvas (bright → grey → black),
        then stamps new events as white. If an ROI is set, its border is
        drawn on top of the frame.
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

        # Draw ROI rectangle overlay if set.
        if roi is not None:
            x1, y1, x2, y2 = roi
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline=200, width=2)

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
    recording_active: bool = False
    recording_paused: bool = False
    spelled_text: str = ""
    appended_count: int = 0
    spelling_threshold: float = 0.85
    spelling_cooldown_s: float = 0.8
    spelling_min_streak: int = 2
    last_committed_letter: str = ""
    last_commit_time_s: float = 0.0
    streak_letter: str = ""
    streak_count: int = 0
    # ROI in sensor pixel coords (x1, y1, x2, y2) or None for full frame.
    roi: Optional[tuple[int, int, int, int]] = None


def parse_args():
    p = argparse.ArgumentParser(description="Live Prophesee X320 inference web dashboard")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=62078)
    p.add_argument("--delta-t-us", type=int, default=50000,
                   help="Camera read chunk in microseconds (default 50000 = 50ms for responsive display)")
    p.add_argument("--inference-window-us", type=int, default=200000,
                   help="Sliding window duration for model inference in microseconds "
                        "(default 200000 = 200ms, matching ASL-DVS training sample duration)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--crop-to-training-aspect", action="store_true", default=True,
                   help="Centre-crop 320×320 to 4:3 aspect to match training data (default: on)")
    p.add_argument("--no-crop-to-training-aspect", dest="crop_to_training_aspect",
                   action="store_false",
                   help="Disable centre-crop; use full square sensor FOV")
    p.add_argument("--input-path", default="",
                   help="Metavision input path. Leave empty to open the live camera.")
    p.add_argument(
        "--input-camera-config",
        default="",
        help="Path to Metavision camera config JSON (same format as metavision_viewer --input-camera-config).",
    )
    # -- Onboard noise filters (GenX320 / IMX636) --
    p.add_argument("--stc-threshold-us", type=int, default=None,
                   help="STC (Spatio-Temporal Contrast) filter threshold in microseconds. "
                        "Events closer in time than this at the same pixel are suppressed. "
                        "Recommended starting range: 2000-10000 us. Higher = more aggressive noise removal.")
    p.add_argument("--erc-rate", type=int, default=None,
                   help="ERC (Event Rate Controller) target max event rate in events/second. "
                        "Limits total throughput to match DAVIS240C-like rates. "
                        "Recommended: 2000000-5000000 (2-5 Mev/s).")
    p.add_argument("--afk-frequency", type=int, default=None, choices=[50, 60],
                   help="AFK (Anti-Flicker) filter frequency in Hz. "
                        "Set to 50 (Europe/Asia) or 60 (Americas) to filter indoor lighting flicker.")
    p.add_argument("--jpeg-quality", type=int, default=70,
                   help="JPEG quality for stream frames (1-95, default 70).")
    p.add_argument("--spelling-threshold", type=float, default=0.85,
                   help="Confidence threshold [0,1] to accept letters into spelled text.")
    p.add_argument("--spelling-cooldown-s", type=float, default=0.8,
                   help="Minimum seconds between accepted letters.")
    p.add_argument("--spelling-min-streak", type=int, default=2,
                   help="Consecutive high-confidence windows required before committing a letter.")
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
    .panel { background: #1a1a1a; padding: 12px; border: 1px solid #333; border-radius: 8px; }
    .big { font-size: 2rem; font-weight: 700; }
    .muted { color: #bbb; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    td { padding: 4px 0; border-bottom: 1px solid #2b2b2b; }
    .ok { color: #7CFC8A; }
    .warn { color: #ffcc66; }
    .feed-container { position: relative; width: 100%; }
    .feed-container img { width: 100%; display: block; border: 1px solid #333; background: #000; }
    .feed-container canvas {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      cursor: crosshair;
    }
    .roi-controls { margin-top: 6px; display: flex; gap: 6px; align-items: center; }
    .roi-controls button { cursor: pointer; }
    .roi-info { color: #bbb; font-size: 0.85em; }
  </style>
</head>
<body>
  <h2>X320 Live ASL Dashboard</h2>
  <div class="wrap">
    <div class="panel">
      <div class="feed-container">
        <img id="feed" src="/stream.mjpg" />
        <canvas id="roi-canvas"></canvas>
      </div>
      <div class="roi-controls">
        <button onclick="clearROI()">Clear ROI</button>
        <span class="roi-info" id="roi-info">No ROI set — click and drag on feed to draw</span>
      </div>
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
      <div style="margin-top:12px; border-top:1px solid #2b2b2b; padding-top:8px;">
        <strong>Spelling recorder</strong>
        <div>State: <span id="recstate" class="muted">idle</span></div>
        <div style="margin:6px 0; display:flex; gap:6px; flex-wrap:wrap;">
          <button onclick="recAction('start')">Record</button>
          <button onclick="recAction('pause')">Pause</button>
          <button onclick="recAction('clear_stop')">Clear/Stop</button>
        </div>
        <div style="word-break:break-all;">
          <span class="muted">Spelled:</span> <span id="spelled">-</span>
        </div>
      </div>
    </div>
  </div>
  <script>
    async function recAction(action) {
      try {
        await fetch('/recorder/' + action, { method: 'POST' });
      } catch (e) {}
    }

    // --- ROI drawing ---
    const roiCanvas = document.getElementById('roi-canvas');
    const roiCtx = roiCanvas.getContext('2d');
    const feedImg = document.getElementById('feed');
    let drawing = false, startX = 0, startY = 0;
    let currentROI = null;  // {x1, y1, x2, y2} normalised [0,1]

    function resizeCanvas() {
      roiCanvas.width = feedImg.clientWidth;
      roiCanvas.height = feedImg.clientHeight;
      drawROIOverlay();
    }
    feedImg.addEventListener('load', resizeCanvas);
    window.addEventListener('resize', resizeCanvas);

    function getNorm(e) {
      const r = roiCanvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
        y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))
      };
    }

    roiCanvas.addEventListener('mousedown', e => {
      const p = getNorm(e);
      startX = p.x; startY = p.y;
      drawing = true;
    });
    roiCanvas.addEventListener('mousemove', e => {
      if (!drawing) return;
      const p = getNorm(e);
      drawROIOverlay({
        x1: Math.min(startX, p.x), y1: Math.min(startY, p.y),
        x2: Math.max(startX, p.x), y2: Math.max(startY, p.y)
      });
    });
    roiCanvas.addEventListener('mouseup', async e => {
      if (!drawing) return;
      drawing = false;
      const p = getNorm(e);
      const roi = {
        x1: Math.min(startX, p.x), y1: Math.min(startY, p.y),
        x2: Math.max(startX, p.x), y2: Math.max(startY, p.y)
      };
      // Ignore tiny accidental clicks (less than 3% of frame).
      if ((roi.x2 - roi.x1) < 0.03 || (roi.y2 - roi.y1) < 0.03) return;
      currentROI = roi;
      drawROIOverlay(roi);
      try {
        await fetch('/roi/set', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(roi)
        });
      } catch (e) {}
      updateROIInfo();
    });

    function drawROIOverlay(roi) {
      roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
      const r = roi || currentROI;
      if (!r) return;
      const x = r.x1 * roiCanvas.width, y = r.y1 * roiCanvas.height;
      const w = (r.x2 - r.x1) * roiCanvas.width, h = (r.y2 - r.y1) * roiCanvas.height;
      // Semi-transparent dark overlay outside ROI.
      roiCtx.fillStyle = 'rgba(0,0,0,0.4)';
      roiCtx.fillRect(0, 0, roiCanvas.width, roiCanvas.height);
      roiCtx.clearRect(x, y, w, h);
      // Bright border.
      roiCtx.strokeStyle = '#00ff88';
      roiCtx.lineWidth = 2;
      roiCtx.strokeRect(x, y, w, h);
    }

    async function clearROI() {
      currentROI = null;
      roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
      try { await fetch('/roi/clear', { method: 'POST' }); } catch (e) {}
      updateROIInfo();
    }

    function updateROIInfo() {
      const el = document.getElementById('roi-info');
      if (currentROI) {
        const r = currentROI;
        el.textContent = `ROI: (${(r.x1*320)|0}, ${(r.y1*320)|0}) to (${(r.x2*320)|0}, ${(r.y2*320)|0})`;
      } else {
        el.textContent = 'No ROI set \u2014 click and drag on feed to draw';
      }
    }

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
        document.getElementById('recstate').textContent =
          m.recording_active ? (m.recording_paused ? 'paused' : 'recording') : 'idle';
        document.getElementById('spelled').textContent = m.spelled_text || '-';
        // Sync ROI from server (in case another client set it).
        if (m.roi && !drawing) {
          currentROI = {x1: m.roi[0]/320, y1: m.roi[1]/320, x2: m.roi[2]/320, y2: m.roi[3]/320};
          drawROIOverlay();
          updateROIInfo();
        } else if (!m.roi && !drawing && currentROI) {
          currentROI = null;
          roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
          updateROIInfo();
        }
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
                "recording_active": state.recording_active,
                "recording_paused": state.recording_paused,
                "spelled_text": state.spelled_text,
                "appended_count": state.appended_count,
                "spelling_threshold": state.spelling_threshold,
                "spelling_cooldown_s": state.spelling_cooldown_s,
                "spelling_min_streak": state.spelling_min_streak,
                "roi": list(state.roi) if state.roi else None,
            }
        return jsonify(payload)

    @app.route("/recorder/start", methods=["POST"])
    def recorder_start():
        with state.lock:
            state.recording_active = True
            state.recording_paused = False
        return jsonify({"ok": True})

    @app.route("/recorder/pause", methods=["POST"])
    def recorder_pause():
        with state.lock:
            if state.recording_active:
                state.recording_paused = True
        return jsonify({"ok": True})

    @app.route("/recorder/clear_stop", methods=["POST"])
    def recorder_clear_stop():
        with state.lock:
            state.recording_active = False
            state.recording_paused = False
            state.spelled_text = ""
            state.appended_count = 0
            state.last_committed_letter = ""
            state.last_commit_time_s = 0.0
            state.streak_letter = ""
            state.streak_count = 0
        return jsonify({"ok": True})

    @app.route("/roi/set", methods=["POST"])
    def roi_set():
        """Set ROI from normalised [0,1] coordinates sent by the frontend."""
        data = json.loads(request.get_data())
        W, H = GENX320_RESOLUTION
        x1 = int(round(float(data["x1"]) * W))
        y1 = int(round(float(data["y1"]) * H))
        x2 = int(round(float(data["x2"]) * W))
        y2 = int(round(float(data["y2"]) * H))
        # Clamp to sensor bounds and ensure x1<x2, y1<y2.
        x1, x2 = max(0, min(x1, x2)), min(W - 1, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H - 1, max(y1, y2))
        with state.lock:
            state.roi = (x1, y1, x2, y2)
        logger.info("ROI set: (%d, %d) -> (%d, %d)", x1, y1, x2, y2)
        return jsonify({"ok": True, "roi": [x1, y1, x2, y2]})

    @app.route("/roi/clear", methods=["POST"])
    def roi_clear():
        with state.lock:
            state.roi = None
        logger.info("ROI cleared")
        return jsonify({"ok": True})

    return app


def configure_hw_filters(device, args, state: SharedState) -> None:
    """
    Configure the GenX320 / IMX636 onboard noise filters via HAL facilities.

    These filters run on-chip and reduce noise before events reach the host,
    which is critical for bridging the DAVIS240C→GenX320 sensor gap.
    """
    filter_log = []

    # --- STC (Spatio-Temporal Contrast / Trail) filter ---
    # Suppresses isolated noise events: if two events at the same pixel are
    # closer together in time than the threshold, the second is kept and
    # isolated (noise) events are dropped.
    if args.stc_threshold_us is not None:
        try:
            noise_filter = device.get_i_noise_filter_module()
            if noise_filter is not None:
                # OpenEB ≥ 4.x API: I_NoiseFilterModule
                try:
                    noise_filter.enable_trail(args.stc_threshold_us)
                    filter_log.append(f"STC trail filter enabled: {args.stc_threshold_us} us (via I_NoiseFilterModule)")
                except AttributeError:
                    # Older API variant
                    noise_filter.set_stc_threshold(args.stc_threshold_us)
                    noise_filter.enable_stc(True)
                    filter_log.append(f"STC filter enabled: {args.stc_threshold_us} us (via set_stc_threshold)")
            else:
                raise AttributeError("get_i_noise_filter_module returned None")
        except (AttributeError, RuntimeError):
            try:
                # Fallback: older HAL with separate trail filter module
                trail = device.get_i_event_trail_filter_module()
                if trail is not None:
                    trail.set_threshold(args.stc_threshold_us)
                    trail.enable(True)
                    filter_log.append(f"STC trail filter enabled: {args.stc_threshold_us} us (via I_EventTrailFilterModule)")
                else:
                    filter_log.append("STC: trail filter facility not available on this device")
            except (AttributeError, RuntimeError) as exc:
                filter_log.append(f"STC: could not configure ({exc})")

    # --- ERC (Event Rate Controller) ---
    # Limits the maximum event rate to prevent sensor saturation and match
    # the lower event rates typical of DAVIS240C training data.
    if args.erc_rate is not None:
        try:
            erc = device.get_i_erc_module()
            if erc is not None:
                try:
                    erc.set_cd_event_rate(args.erc_rate)
                except AttributeError:
                    erc.set_event_rate(args.erc_rate)
                erc.enable(True)
                filter_log.append(f"ERC enabled: target {args.erc_rate:,} events/s")
            else:
                raise AttributeError("get_i_erc_module returned None")
        except (AttributeError, RuntimeError) as exc:
            filter_log.append(f"ERC: could not configure ({exc})")

    # --- AFK (Anti-Flicker) ---
    # Filters periodic events caused by 50/60 Hz artificial lighting.
    # The DAVIS240C training data may not contain this noise.
    if args.afk_frequency is not None:
        try:
            afk = device.get_i_antiflicker_module()
            if afk is not None:
                try:
                    afk.set_frequency_band(args.afk_frequency, args.afk_frequency)
                except (AttributeError, TypeError):
                    try:
                        afk.set_frequency(args.afk_frequency)
                    except AttributeError:
                        afk.set_filtering_mode(args.afk_frequency)
                afk.enable(True)
                filter_log.append(f"AFK enabled: {args.afk_frequency} Hz")
            else:
                raise AttributeError("get_i_antiflicker_module returned None")
        except (AttributeError, RuntimeError) as exc:
            filter_log.append(f"AFK: could not configure ({exc})")

    if filter_log:
        msg = " | ".join(filter_log)
        logger.info("HW filters: %s", msg)
        with state.lock:
            state.message = msg


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

    hw_filters_requested = any([
        args.stc_threshold_us is not None,
        args.erc_rate is not None,
        args.afk_frequency is not None,
    ])
    need_hal_device = args.input_camera_config or hw_filters_requested

    iterator_kwargs = {"input_path": args.input_path, "delta_t": args.delta_t_us}
    iterator = None
    device = None
    try:
        if need_hal_device:
            # We need direct HAL device access for bias config and/or HW filters.
            # Open device explicitly so we can configure it before streaming.
            from metavision_core.event_io.raw_reader import initiate_device  # type: ignore
            device = initiate_device(path=args.input_path)

            # Apply bias config JSON if provided.
            if args.input_camera_config:
                cfg_path = str(Path(args.input_camera_config).expanduser().resolve())
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

                logger.info("Applied %d biases from %s", applied, cfg_path)

            # Apply onboard HW filters (STC, ERC, AFK).
            if hw_filters_requested:
                configure_hw_filters(device, args, state)

            if not hasattr(EventsIterator, "from_device"):
                raise RuntimeError("EventsIterator.from_device is unavailable in this SDK.")
            iterator = EventsIterator.from_device(device=device, delta_t=args.delta_t_us)  # fast chunks

        else:
            iterator = EventsIterator(**iterator_kwargs)

    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.errors += 1
            state.message = f"Camera init failed: {exc}"
        logger.exception("Camera initialization failed")
        return

    chunk_s = args.delta_t_us / 1e6
    infer_window_us = args.inference_window_us

    # Persistent event display with temporal decay.
    display = EventDisplay(sensor_wh=GENX320_RESOLUTION, decay=0.6)

    # Rolling buffer: accumulate fast 50ms chunks into a sliding 200ms window.
    # Display updates every chunk (~20 FPS); inference runs on the full window.
    event_ring: list[np.ndarray] = []   # list of (N, 4) chunks
    ring_duration_us = 0                # total time span in the ring

    # EMA smoothing over prediction probabilities to suppress transient false positives.
    smooth_probs: Optional[np.ndarray] = None
    smoothing_alpha = 0.4  # weight for new predictions (lower = smoother)

    try:
        for events_struct in iterator:
            if events_struct is None:
                continue

            events = structured_events_to_nx4(events_struct)
            n = int(len(events))
            if n == 0:
                continue

            # Update display immediately with this chunk (responsive ~20 FPS).
            # All events are shown; the ROI rectangle is drawn on top.
            with state.lock:
                current_roi = state.roi
            frame_jpeg = display.update(
                events,
                quality=max(1, min(95, args.jpeg_quality)),
                roi=current_roi,
            )
            with state.lock:
                state.latest_jpeg = frame_jpeg
                state.last_update_s = time.time()

            # Append chunk to rolling buffer.
            event_ring.append(events)
            if n > 0:
                chunk_span = float(events[-1, 0] - events[0, 0])
                ring_duration_us += max(chunk_span, args.delta_t_us)

            # Evict old chunks to keep the ring at ~inference_window_us.
            while len(event_ring) > 1 and ring_duration_us > infer_window_us * 1.5:
                removed = event_ring.pop(0)
                if len(removed) > 0:
                    removed_span = float(removed[-1, 0] - removed[0, 0])
                    ring_duration_us -= max(removed_span, args.delta_t_us)

            # Run inference once the ring covers enough time.
            if ring_duration_us < infer_window_us * 0.8:
                continue

            window_events = np.concatenate(event_ring, axis=0)

            # If an ROI is set, filter to only events inside it for inference.
            # The display still shows all events — only the model input is cropped.
            if current_roi is not None:
                rx1, ry1, rx2, ry2 = current_roi
                mask = (
                    (window_events[:, 1] >= rx1) & (window_events[:, 1] <= rx2) &
                    (window_events[:, 2] >= ry1) & (window_events[:, 2] <= ry2)
                )
                window_events = window_events[mask]

            total_n = len(window_events)
            if total_n < 500:
                # Too few events — likely just noise, skip inference.
                continue

            start = time.perf_counter()
            pred = inferencer.predict(window_events)
            infer_ms = (time.perf_counter() - start) * 1000.0

            # EMA smooth the raw probabilities to reduce flicker.
            raw_probs = np.array([pred["scores"][c] for c in CLASSES], dtype=np.float32)
            if smooth_probs is None:
                smooth_probs = raw_probs
            else:
                smooth_probs = smoothing_alpha * raw_probs + (1 - smoothing_alpha) * smooth_probs

            smooth_idx = int(smooth_probs.argmax())
            smooth_pred = CLASSES[smooth_idx]
            smooth_conf = float(smooth_probs[smooth_idx])
            smooth_scores = {c: float(smooth_probs[i]) for i, c in enumerate(CLASSES)}

            sorted_scores = sorted(
                smooth_scores.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            topk = sorted_scores[:args.top_k]

            now_s = time.time()
            with state.lock:
                state.prediction = smooth_pred
                state.confidence = smooth_conf
                state.topk = [(k, float(v)) for k, v in topk]
                state.window_events = total_n
                state.event_rate_eps = total_n / (ring_duration_us / 1e6) if ring_duration_us > 0 else 0.0
                state.inference_ms = infer_ms
                state.inference_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
                state.total_windows += 1
                state.status = "running"
                state.message = ""

                # Spelling recorder: require sustained confidence + cooldown.
                if state.confidence >= state.spelling_threshold:
                    if state.prediction == state.streak_letter:
                        state.streak_count += 1
                    else:
                        state.streak_letter = state.prediction
                        state.streak_count = 1
                else:
                    state.streak_letter = ""
                    state.streak_count = 0

                if state.recording_active and not state.recording_paused:
                    ready = (
                        state.streak_count >= state.spelling_min_streak
                        and (now_s - state.last_commit_time_s) >= state.spelling_cooldown_s
                    )
                    if ready:
                        same_letter = state.streak_letter == state.last_committed_letter
                        # Allow doubles, but require extra dwell if repeating the same letter.
                        if (not same_letter) or ((now_s - state.last_commit_time_s) >= 2.0 * state.spelling_cooldown_s):
                            state.spelled_text += state.streak_letter
                            state.last_committed_letter = state.streak_letter
                            state.last_commit_time_s = now_s
                            state.appended_count += 1
    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.errors += 1
            state.message = f"Camera loop failed: {exc}"
        logger.exception("Camera loop failed")


def main() -> None:
    args = parse_args()
    state = SharedState()
    state.spelling_threshold = max(0.0, min(1.0, float(args.spelling_threshold)))
    state.spelling_cooldown_s = max(0.0, float(args.spelling_cooldown_s))
    state.spelling_min_streak = max(1, int(args.spelling_min_streak))

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
