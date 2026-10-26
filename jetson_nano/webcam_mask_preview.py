#!/usr/bin/env python3
"""Show webcam input and the SLIM50_SVD50 person mask side by side on Nano 1."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


MODEL_CANDIDATES = [
    "default_model.onnx",
    "rvm_mobilenetv3_large_svd50.onnx",
    "rvm_mobilenetv1.onnx",
    "rvm_mobilenetv2.onnx",
    "rvm_mobilenetv3_large.onnx",
    "rvm_mobilenetv3_small.onnx",
    "SLIM50_SVD50.onnx",
]


def find_default_model() -> Path | None:
    """Find the best available model, prioritizing superior fine-tuned models over distilled baseline."""
    app_dir = Path(__file__).resolve().parent
    repo_dir = app_dir.parent
    bundle_dir = getattr(sys, "_MEIPASS", None)

    search_dirs = []
    if bundle_dir:
        search_dirs.extend([Path(bundle_dir) / "model", Path(bundle_dir)])
    search_dirs.extend([
        app_dir / "models",
        app_dir / "model",
        app_dir,
        repo_dir / "outputs" / "checkpoints",
        repo_dir / "jetson_nano" / "models",
        repo_dir / "compression" / "outputs" / "jetson_nano",
    ])

    for model_name in MODEL_CANDIDATES:
        for sdir in search_dirs:
            candidate = sdir / model_name
            if candidate.is_file():
                return candidate
    return None


def get_initial_states(session) -> list[np.ndarray]:
    """Dynamically initialize recurrent states based on actual ONNX session input tensor shapes."""
    inputs = session.get_inputs()
    states = []
    for inp in inputs[1:]:
        shape = [int(dim) if isinstance(dim, int) else 1 for dim in inp.shape]
        states.append(np.zeros(shape, dtype=np.float32))
    return states



def make_session(model_path: str, provider: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime could not be loaded") from exc
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
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    feed = {"image": image}
    feed.update({f"r{i + 1}": state for i, state in enumerate(states)})
    outputs = session.run(["logits", "r1o", "r2o", "r3o", "r4o"], feed)
    logits, *next_states = outputs
    logits = cv2.resize(logits[0, 0], (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (logits >= 0).astype(np.uint8) * 255
    return mask, next_states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="Webcam index or device path")
    parser.add_argument("--model", default=None, help="Optional ONNX model path override")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--width", type=int, default=640, help="Requested webcam width")
    parser.add_argument("--height", type=int, default=480, help="Requested webcam height")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser() if args.model else find_default_model()
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            "Cannot find any ONNX model. Run `python export_onnx.py` to export fine-tuned models, "
            "or pass `--model /path/to/model.onnx`."
        )
    session = make_session(str(model_path), args.provider)

    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam source {args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    states = get_initial_states(session)
    title = f"Nano 1 | Model: {model_path.name} | q to quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, args.width * 2, args.height)
    count = 0
    report_start = time.perf_counter()
    print(f"Active Model: {model_path.name} (Superior to distilled baseline)", flush=True)
    print(f"State Shapes: {[s.shape for s in states]}", flush=True)
    print("Live preview started. Press q or Esc in the preview window to stop.", flush=True)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")
            mask, states = segment(session, frame, states)
            mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, "WEBCAM", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(mask_view, "PERSON MASK", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(title, np.hstack((frame, mask_view)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            count += 1
            if count % 30 == 0:
                elapsed = max(time.perf_counter() - report_start, 1e-6)
                print(f"Preview rate: {30 / elapsed:.1f} FPS", flush=True)
                report_start = time.perf_counter()
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
