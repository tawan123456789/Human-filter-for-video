#!/usr/bin/env python3
"""Read a webcam on Nano 1 and run the person-segmentation ONNX locally."""
import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

STATE_SHAPES = (
    (1, 8, 128, 128),
    (1, 10, 64, 64),
    (1, 20, 32, 32),
    (1, 64, 16, 16),
)


def make_session(model_path: str, provider: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install a Jetson-compatible onnxruntime build first") from exc
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
    parser.add_argument("--model", required=True, help="Path to SLIM50_SVD50 ONNX")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0, help="FPS for the saved video")
    parser.add_argument("--output", help="Optional path to save the mask-overlay video")
    parser.add_argument("--no-display", action="store_true", help="Run headless")
    args = parser.parse_args()
    if args.no_display and not args.output:
        parser.error("--no-display requires --output so the result is saved")

    session = make_session(args.model, args.provider)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    states = [np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES]
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
