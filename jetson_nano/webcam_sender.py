#!/usr/bin/env python3
"""Capture webcam frames, segment people with the stateful RVM ONNX, and stream them."""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


STATE_SHAPES = (
    (1, 8, 128, 128),
    (1, 10, 64, 64),
    (1, 20, 32, 32),
    (1, 64, 16, 16),
)
MODEL_FILENAME = "SLIM50_SVD50.onnx"


def find_default_model() -> Path | None:
    """Find the model beside a PyInstaller bundle or in the source checkout."""
    candidates = []
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(Path(bundle_dir) / "model" / MODEL_FILENAME)
    app_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            app_dir / "model" / MODEL_FILENAME,
            app_dir / MODEL_FILENAME,
            app_dir.parent / "compression" / "outputs" / "jetson_nano" / MODEL_FILENAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def make_session(model_path: str, provider: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required on Nano 1. Install a Jetson-compatible build."
        ) from exc

    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers: {available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    session = ort.InferenceSession(model_path, providers=providers)
    print("ONNX Runtime providers:", session.get_providers(), flush=True)
    return session


def segment(session, frame_bgr: np.ndarray, states: list[np.ndarray]):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    feed = {"image": image}
    feed.update({f"r{i + 1}": state for i, state in enumerate(states)})
    outputs = session.run(["logits", "r1o", "r2o", "r3o", "r4o"], feed)
    logits, *next_states = outputs
    logits = logits[0, 0]
    # The model was trained with a 256x256 input; return a mask aligned to the webcam frame.
    logits = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (logits >= 0).astype(np.uint8) * 255
    return mask, next_states


def send_packet(sock: socket.socket, frame: np.ndarray, mask: np.ndarray, seq: int) -> None:
    ok_frame, frame_buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    ok_mask, mask_buf = cv2.imencode(".png", mask)
    if not ok_frame or not ok_mask:
        raise RuntimeError("Could not encode webcam frame or segmentation mask")
    frame_bytes, mask_bytes = frame_buf.tobytes(), mask_buf.tobytes()
    header = json.dumps(
        {
            "seq": seq,
            "time_ns": time.time_ns(),
            "frame_bytes": len(frame_bytes),
            "mask_bytes": len(mask_bytes),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    payload = struct.pack("!I", len(header)) + header + frame_bytes + mask_bytes
    sock.sendall(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="IP address of Nano 2")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--camera", default="0",
        help="Webcam index (0) or Linux device path (/dev/video0)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to ONNX model (defaults to bundled SLIM50_SVD50.onnx)",
    )
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--preview", action="store_true", help="Show webcam and person mask")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser() if args.model else find_default_model()
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {MODEL_FILENAME}. Place it under model/ beside the app, "
            "or pass --model /path/to/SLIM50_SVD50.onnx."
        )
    session = make_session(str(model_path), args.provider)
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam source {args.camera}")
    # Keep the newest frame available instead of building a latency backlog.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    states = [np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES]
    seq = 0
    report_started = time.perf_counter()
    try:
        print(
            f"Model: {model_path}\n"
            f"Using webcam {args.camera}; connecting to Nano 2 at {args.host}:{args.port} ...",
            flush=True,
        )
        with socket.create_connection((args.host, args.port)) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Connected. Press Ctrl+C to stop.", flush=True)
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames")
                mask, states = segment(session, frame, states)
                send_packet(sock, frame, mask, seq)
                seq += 1
                if args.preview:
                    overlay = frame.copy()
                    person = mask > 0
                    green = np.zeros_like(frame)
                    green[:, :] = (0, 255, 0)
                    overlay[person] = cv2.addWeighted(
                        frame[person], 0.55, green[person], 0.45, 0
                    )
                    cv2.imshow("Nano 1 webcam - segmented person", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if seq % 30 == 0:
                    elapsed = max(time.perf_counter() - report_started, 1e-6)
                    print(f"Sent {seq} frames; recent rate={30 / elapsed:.1f} FPS", flush=True)
                    report_started = time.perf_counter()
    except KeyboardInterrupt:
        print("Stopping webcam stream.", flush=True)
    finally:
        camera.release()
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
