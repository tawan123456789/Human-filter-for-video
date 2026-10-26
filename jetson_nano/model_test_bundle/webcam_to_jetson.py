#!/usr/bin/env python3
"""Segment webcam frames with Model A/B and stream frame+mask to a Jetson Nano receiver."""
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
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "model_a_svd50": ROOT / "models" / "model_a_svd50_fp16.onnx",
    "model_b_slim50_svd50": ROOT / "models" / "model_b_slim50_svd50_fp16.onnx",
}
INPUT_SIZE = 256


def make_session(path: Path, provider: str):
    if not path.is_file():
        raise FileNotFoundError(f"Model is missing: {path}")
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider unavailable; installed providers: {available}")
    active = "CUDAExecutionProvider" if provider in ("auto", "cuda") and "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    providers = [active]
    if active != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(path), providers=providers)
    print(f"ONNX Runtime providers: {session.get_providers()}", flush=True)
    return session


def initial_states(session):
    states = {}
    for spec in session.get_inputs()[1:]:
        shape = [int(dim) if isinstance(dim, (int, np.integer)) and int(dim) > 0 else 1 for dim in spec.shape]
        states[spec.name] = np.zeros(shape, dtype=np.float32)
    return states


def segment(session, frame_bgr, states):
    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = (rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    feeds = {inputs[0].name: image, **states}
    values = session.run([item.name for item in outputs], feeds)
    logits = np.asarray(values[0], dtype=np.float32)[0, 0]
    next_states = {spec.name: value for spec, value in zip(inputs[1:], values[1:])}
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    return (probability >= 0.5).astype(np.uint8) * 255, next_states


def send_packet(sock, frame, mask, seq, variant):
    ok_frame, frame_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok_mask, mask_buf = cv2.imencode(".png", mask)
    if not ok_frame or not ok_mask:
        raise RuntimeError("Could not encode webcam frame or mask")
    frame_bytes, mask_bytes = frame_buf.tobytes(), mask_buf.tobytes()
    header = json.dumps({
        "seq": seq, "time_ns": time.time_ns(), "variant": variant,
        "input_size": INPUT_SIZE, "frame_bytes": len(frame_bytes), "mask_bytes": len(mask_bytes),
    }, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header)) + header + frame_bytes + mask_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="IP address of the Nano running stream_receiver.py")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera", default="0", help="Webcam index or device path")
    parser.add_argument("--variant", choices=tuple(MODEL_PATHS), default="model_b_slim50_svd50")
    parser.add_argument("--model", type=Path, help="Optional ONNX path override")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve() if args.model else MODEL_PATHS[args.variant]
    session = make_session(model_path, args.provider)
    states = initial_states(session)
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_source, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam source {args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    seq, report_at = 0, time.perf_counter()
    try:
        print(f"Model: {model_path}\nStreaming to {args.host}:{args.port}", flush=True)
        with socket.create_connection((args.host, args.port), timeout=10) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames")
                mask, states = segment(session, frame, states)
                send_packet(sock, frame, mask, seq, args.variant)
                seq += 1
                if args.preview:
                    overlay = frame.copy()
                    fg = mask > 0
                    green = np.zeros_like(frame)
                    green[:] = (0, 255, 0)
                    overlay[fg] = cv2.addWeighted(frame[fg], 0.55, green[fg], 0.45, 0)
                    cv2.imshow("Webcam segmentation sent to Jetson", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if seq % 30 == 0:
                    now = time.perf_counter()
                    print(f"Sent {seq} frames; stream rate={30 / max(now - report_at, 1e-6):.1f} FPS", flush=True)
                    report_at = now
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
