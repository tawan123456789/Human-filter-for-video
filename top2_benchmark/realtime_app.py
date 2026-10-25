"""Real-time, one-model-at-a-time webcam tester for the three ONNX FP16 models.

Run from the repository root with: python -m top2_benchmark.realtime_app
Keys: 1/2/3 or Left/Right switch model, m toggle overlay/mask, q/ESC quit.
"""
from __future__ import annotations

import argparse
import time
import os
import sys
from pathlib import Path

import cv2
import numpy as np
# On Windows the CUDA/cuDNN DLLs shipped with PyTorch can be shared with ORT.
# Jetson uses its system JetPack CUDA libraries and does not need PyTorch here.
if __import__("os").name == "nt":
    import torch  # noqa: F401
import onnxruntime as ort


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
FINAL = ROOT / "compression" / "outputs" / "jetson_nano_256"
MODEL_PATHS = {
    "Original baseline": FINAL / "baseline" / "baseline_256_fp16.onnx",
    "SVD50 + KD": FINAL / "model_a_svd50" / "model_a_svd50_256_fp16.onnx",
    "SLIM50 + SVD50 + KD": FINAL / "model_b_slim50_svd50" / "model_b_slim50_svd50_256_fp16.onnx",
}
MODEL_SIZE = 256
COLORS = [(70, 210, 120), (255, 180, 65), (210, 110, 240)]  # BGR


def make_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return ort.InferenceSession(str(path), providers=providers)


def initial_states(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    states = {}
    for spec in session.get_inputs()[1:]:
        shape = [int(dim) if isinstance(dim, (int, np.integer)) and int(dim) > 0 else 1 for dim in spec.shape]
        states[spec.name] = np.zeros(shape, dtype=np.float32)
    return states


def color_panel(frame: np.ndarray, prob: np.ndarray, title: str, latency: float,
                fps: float, display_fps: float, foreground_pct: float,
                confidence: float, color: tuple[int, int, int], mask_only: bool,
                provider: str, model_index: int, model_count: int) -> np.ndarray:
    height, width = frame.shape[:2]
    prob_full = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = prob_full >= 0.5
    if mask_only:
        panel = np.zeros_like(frame)
        panel[mask] = color
    else:
        panel = frame.copy()
        tint = np.empty_like(frame)
        tint[:] = color
        panel[mask] = cv2.addWeighted(frame, 0.48, tint, 0.52, 0)[mask]
    cv2.rectangle(panel, (0, 0), (width, 130), (20, 24, 30), -1)
    cv2.putText(panel, f"{model_index + 1}/{model_count}  {title}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Inference: {latency:.1f} ms   |   Model FPS: {fps:.1f}   |   Live FPS: {display_fps:.1f}", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (220, 230, 238), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Foreground: {foreground_pct:.1f}%   |   Mean foreground confidence: {confidence:.3f}", (14, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (220, 230, 238), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Provider: {provider}   |   1-3/Left-Right: switch   m: view   q: quit", (14, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 195, 210), 1, cv2.LINE_AA)
    return panel


def run(camera: int, width: int, height: int, provider: str | None) -> None:
    available = ort.get_available_providers()
    if provider:
        if provider not in available:
            raise RuntimeError(f"Provider {provider} unavailable. Installed: {available}")
        active = provider
    else:
        active = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if active == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    providers = [active]
    if active != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    print(f"ONNX Runtime providers: {available}; selected: {active}")
    sessions = {name: make_session(path, providers) for name, path in MODEL_PATHS.items()}
    states = {name: initial_states(session) for name, session in sessions.items()}
    input_names = {name: [spec.name for spec in session.get_inputs()] for name, session in sessions.items()}
    output_names = {name: [spec.name for spec in session.get_outputs()] for name, session in sessions.items()}

    backend = cv2.CAP_DSHOW if os.name == "nt" else getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}; check camera permissions or --camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    names = list(sessions)
    selected_index = 0
    latency_ema = {name: 0.0 for name in sessions}
    fps_ema = {name: 0.0 for name in sessions}
    mask_only = False
    last_frame_time = time.perf_counter()
    window = "RVM ONNX FP16 - realtime single-model tester"
    print("Keys: 1/2/3 or Left/Right = switch model, m = overlay/mask, q or ESC = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame unavailable")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
            image = (rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
            name = names[selected_index]
            session = sessions[name]
            feed = {input_names[name][0]: image}
            feed.update(states[name])
            t0 = time.perf_counter()
            outputs = session.run(output_names[name], feed)
            elapsed = (time.perf_counter() - t0) * 1000.0
            # ONNX output 0 is segmentation logits; the remaining four are recurrent states.
            states[name] = {key: value for key, value in zip(input_names[name][1:], outputs[1:])}
            latency_ema[name] = elapsed if latency_ema[name] == 0 else 0.8 * latency_ema[name] + 0.2 * elapsed
            instantaneous_fps = 1000.0 / max(elapsed, 1e-6)
            fps_ema[name] = instantaneous_fps if fps_ema[name] == 0 else 0.8 * fps_ema[name] + 0.2 * instantaneous_fps
            logits = np.asarray(outputs[0], dtype=np.float32)[0, 0]
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
            prob_full = cv2.resize(prob, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = prob_full >= 0.5
            foreground_pct = float(mask.mean() * 100.0)
            confidence = float(prob_full[mask].mean()) if mask.any() else 0.0

            now = time.perf_counter()
            live_fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            canvas = color_panel(frame, prob, name, latency_ema[name], fps_ema[name], live_fps,
                                 foreground_pct, confidence, COLORS[selected_index], mask_only,
                                 active, selected_index, len(names))
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mask_only = not mask_only
            if key in (ord("1"), ord("2"), ord("3")):
                next_index = key - ord("1")
                if next_index < len(names) and next_index != selected_index:
                    selected_index = next_index
                    states[names[selected_index]] = initial_states(sessions[names[selected_index]])
                    latency_ema[names[selected_index]] = 0.0
                    fps_ema[names[selected_index]] = 0.0
                    last_frame_time = time.perf_counter()
            if key in (81, 2424832):  # Left arrow across OpenCV backends
                selected_index = (selected_index - 1) % len(names)
                states[names[selected_index]] = initial_states(sessions[names[selected_index]])
                latency_ema[names[selected_index]] = 0.0
                fps_ema[names[selected_index]] = 0.0
                last_frame_time = time.perf_counter()
            if key in (83, 2555904):  # Right arrow across OpenCV backends
                selected_index = (selected_index + 1) % len(names)
                states[names[selected_index]] = initial_states(sessions[names[selected_index]])
                latency_ema[names[selected_index]] = 0.0
                fps_ema[names[selected_index]] = 0.0
                last_frame_time = time.perf_counter()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--provider", choices=ort.get_available_providers(), default=None,
                        help="ORT provider; default automatically chooses CUDA when available, otherwise CPU")
    args = parser.parse_args()
    run(args.camera, args.width, args.height, args.provider)


if __name__ == "__main__":
    main()
