#!/usr/bin/env python3
"""Interactive webcam tester for the newly fine-tuned FP16 ONNX models."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Model A — SVD50 + KD": ROOT / "models" / "model_a_svd50_fp16.onnx",
    "Model B — SLIM50 + SVD50 + KD": ROOT / "models" / "model_b_slim50_svd50_fp16.onnx",
}

st.set_page_config(page_title="RVM FP16 Nano Model Tester", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="Loading ONNX model...")
def load_session(model_path: str, preferred_provider: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preferred_provider == "CUDA":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDA provider unavailable. Available providers: {available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif preferred_provider == "CPU":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
    session = ort.InferenceSession(model_path, providers=providers)
    return session, session.get_providers()


def make_initial_states(session):
    states = []
    for spec in session.get_inputs()[1:]:
        shape = [int(dim) if isinstance(dim, (int, np.integer)) and int(dim) > 0 else 1 for dim in spec.shape]
        states.append(np.zeros(shape, dtype=np.float32))
    return states


def open_camera(index: int):
    backend = cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def apply_effect(rgb, probability, threshold, effect):
    mask = probability >= threshold
    if effect == "Binary mask":
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    if effect == "Green background":
        bg = np.zeros_like(rgb)
        bg[:] = (0, 190, 45)
        return np.where(mask[..., None], rgb, bg).astype(np.uint8)
    if effect == "Blur background":
        bg = cv2.GaussianBlur(rgb, (35, 35), 0)
        return np.where(mask[..., None], rgb, bg).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([40, 230, 120])).astype(np.uint8)
    return overlay


def clear_camera():
    cap = st.session_state.pop("nano_test_cap", None)
    if cap is not None:
        cap.release()
    st.session_state.pop("nano_test_camera", None)
    st.session_state.pop("nano_test_model", None)
    st.session_state.pop("nano_test_states", None)


@st.fragment(run_every=0.1)
def show_live(camera_index, session, model_name, threshold, effect):
    cap = st.session_state.get("nano_test_cap")
    if (cap is None or st.session_state.get("nano_test_camera") != camera_index
            or st.session_state.get("nano_test_model") != model_name
            or not cap.isOpened()):
        clear_camera()
        cap = open_camera(camera_index)
        if not cap.isOpened():
            st.error(f"Cannot open webcam {camera_index}. Check the camera or choose another index.")
            return
        st.session_state["nano_test_cap"] = cap
        st.session_state["nano_test_camera"] = camera_index
        st.session_state["nano_test_model"] = model_name
        st.session_state["nano_test_states"] = make_initial_states(session)

    ok, frame = cap.read()
    if not ok:
        st.warning("Webcam is open but did not return a frame.")
        return

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    input_specs, output_specs = session.get_inputs(), session.get_outputs()
    feeds = {input_specs[0].name: image}
    feeds.update({spec.name: value for spec, value in zip(input_specs[1:], st.session_state["nano_test_states"])})

    start = time.perf_counter()
    outputs = session.run([spec.name for spec in output_specs], feeds)
    latency_ms = (time.perf_counter() - start) * 1000.0
    st.session_state["nano_test_states"] = outputs[1:]
    logits = cv2.resize(np.asarray(outputs[0], dtype=np.float32)[0, 0], (width, height),
                        interpolation=cv2.INTER_LINEAR)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    result = apply_effect(rgb, probability, threshold, effect)

    left, right = st.columns(2)
    with left:
        st.image(rgb, caption=f"Webcam input · {width}×{height}", use_container_width=True)
    with right:
        st.image(result, caption=f"{model_name} · {effect}", use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Inference latency", f"{latency_ms:.1f} ms")
    m2.metric("Model FPS", f"{1000.0 / latency_ms:.1f}" if latency_ms else "—")
    m3.metric("Input size", "256 × 256")


st.title("RVM FP16 · Jetson Nano webcam test")
st.caption("ทดสอบโมเดล fine-tune ใหม่ Model A และ Model B ผ่าน ONNX Runtime")

with st.sidebar:
    model_name = st.selectbox("Model", list(MODEL_PATHS))
    camera_index = st.selectbox("Webcam input", list(range(6)), index=0)
    provider = st.selectbox("ONNX Runtime provider", ["Auto", "CUDA", "CPU"])
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
    effect = st.selectbox("Preview effect", ["Mask overlay", "Binary mask", "Green background", "Blur background"])
    run_live = st.toggle("Start realtime webcam", value=False)
    st.caption(f"Model: {MODEL_PATHS[model_name].name}")

model_path = MODEL_PATHS[model_name]
if not model_path.is_file():
    st.error(f"Missing model file: {model_path}")
    st.stop()
try:
    session, active_providers = load_session(str(model_path), provider)
    st.caption(f"ONNX Runtime providers: {', '.join(active_providers)}")
except Exception as exc:
    clear_camera()
    st.error(f"Could not load model: {exc}")
    st.stop()

if run_live:
    show_live(camera_index, session, model_name, threshold, effect)
else:
    clear_camera()
    st.info("Choose a webcam and turn on Start realtime webcam.")
