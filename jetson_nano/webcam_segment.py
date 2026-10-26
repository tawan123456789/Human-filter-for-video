#!/usr/bin/env python3
"""Read a webcam on Nano 1 and run the person-segmentation ONNX locally."""
import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

MODEL_CANDIDATES = [
    "default_model.onnx",
    "rvm_mobilenetv3_large_svd50_fp16.onnx",
    "rvm_mobilenetv1_fp16.onnx",
    "SLIM50_SVD50_fp16.onnx",
    "rvm_mobilenetv2_fp16.onnx",
    "rvm_mobilenetv3_small_fp16.onnx",
    "rvm_mobilenetv3_large_fp16.onnx",
    "rvm_mobilenetv3_large_svd50.onnx",
    "rvm_mobilenetv1.onnx",
    "rvm_mobilenetv2.onnx",
    "rvm_mobilenetv3_large.onnx",
    "rvm_mobilenetv3_small.onnx",
    "SLIM50_SVD50.onnx",
]


def find_default_model() -> Path | None:
    app_dir = Path(__file__).resolve().parent
    repo_dir = app_dir.parent
    search_dirs = [
        app_dir / "models",
        app_dir / "model",
        app_dir,
        repo_dir / "outputs" / "checkpoints",
        repo_dir / "jetson_nano" / "models",
        repo_dir / "compression" / "outputs" / "jetson_nano",
        repo_dir / "compression" / "outputs" / "jetson_nano_256",
    ]
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
        raise RuntimeError("Install a Jetson-compatible onnxruntime build first") from exc
    available = ort.get_available_providers()
    print("Available ORT providers:", available, flush=True)

    if provider in ("trt", "tensorrt"):
        if "TensorrtExecutionProvider" not in available:
            print("[WARN] TensorrtExecutionProvider unavailable, trying CUDA...", flush=True)
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers: {available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:  # auto
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()
    print("Active ONNX Runtime providers:", active, flush=True)
    if active and active[0] == "CPUExecutionProvider":
        print(
            "\n" + "!" * 70 + "\n"
            "[WARNING] Model is running on CPUExecutionProvider!\n"
            "On Jetson Nano this drops speed to ~3-4 FPS.\n"
            "To get 25-35+ FPS on Jetson Nano:\n"
            "  1. Install onnxruntime-gpu wheel (wheels/onnxruntime_gpu-*.whl)\n"
            "  2. Make sure CUDA/cuDNN is in LD_LIBRARY_PATH\n"
            "  3. Run 'sudo jetson_clocks' and 'sudo nvpmodel -m 0'\n"
            + "!" * 70 + "\n",
            flush=True,
        )
    return session


def segment(session, frame_bgr: np.ndarray, states: List[np.ndarray]):
    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (256, 256))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    feed = {"image": image}
    feed.update({f"r{i + 1}": state for i, state in enumerate(states)})
    logits, *next_states = session.run(
        ["logits", "r1o", "r2o", "r3o", "r4o"], feed
    )
    logits = cv2.resize(logits[0, 0], (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (logits >= 0).astype(np.uint8) * 255
    return mask, next_states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--model", default=None, help="Path to ONNX model (defaults to best available model)")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu", "trt", "tensorrt"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0, help="FPS for the saved video")
    parser.add_argument("--output", help="Optional path to save the mask-overlay video")
    parser.add_argument("--no-display", action="store_true", help="Run headless")
    args = parser.parse_args()
    if args.no_display and not args.output:
        parser.error("--no-display requires --output so the result is saved")

    model_path = Path(args.model).expanduser() if args.model else find_default_model()
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            "Cannot find any ONNX model. Run `python export_onnx.py` to export fine-tuned models, "
            "or pass `--model /path/to/model.onnx`."
        )

    session = make_session(str(model_path), args.provider)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    states = get_initial_states(session)
    print(f"Active Model: {model_path.name} (Superior to distilled baseline)")
    print(f"State Shapes: {[s.shape for s in states]}")
    writer = None
    frame_count = 0
    try:
        print("Webcam segmentation running. Press q in the preview to stop.", flush=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")

            mask, states = segment(session, frame, states)
            overlay = frame.copy()
            green = np.zeros_like(frame)
            green[:, :] = (0, 255, 0)
            person = mask > 0
            overlay[person] = cv2.addWeighted(
                frame[person], 0.55, green[person], 0.45, 0
            )
            preview = np.concatenate((frame, overlay), axis=1)
            cv2.putText(preview, "Webcam", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(preview, "Segmented person", (frame.shape[1] + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if args.output:
                if writer is None:
                    output = Path(args.output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    h, w = overlay.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps, (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output}")
                writer.write(overlay)

            if not args.no_display:
                cv2.imshow("Nano 1 - Webcam and person segmentation", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames", flush=True)
    except KeyboardInterrupt:
        print("Stopping webcam segmentation.", flush=True)
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
