"""Real-time SLIM50_SVD50 webcam segmentation with Streamlit."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = [
    ROOT / "compression" / "outputs" / "jetson_nano" / "SLIM50_SVD50.onnx",
    ROOT / "compression" / "outputs" / "exports" / "SLIM50_SVD50" / "stateful_fp32.onnx",
]
MODEL_PATH = next((path for path in MODEL_PATHS if path.is_file()), None)
STATE_SHAPES = (
    (1, 8, 128, 128),
    (1, 10, 64, 64),
    (1, 20, 32, 32),
    (1, 64, 16, 16),
)

st.set_page_config(page_title="SLIM50_SVD50 Live Segmentation", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="Loading SLIM50_SVD50 ONNX...")
def load_session(model_path: str, preferred_provider: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preferred_provider == "CUDA":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDA provider is unavailable. Available providers: {available}"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif preferred_provider == "CPU":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    session = ort.InferenceSession(model_path, providers=providers)
    return session, session.get_providers()


def open_camera(camera_index: int):
    backend = cv2.CAP_DSHOW if os.name == "nt" else getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def apply_effect(rgb: np.ndarray, probability: np.ndarray, threshold: float, effect: str):
    mask = probability >= threshold
    if effect == "Binary mask":
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    if effect == "Green background":
        background = np.zeros_like(rgb)
        background[:] = (0, 190, 45)
        return np.where(mask[..., None], rgb, background).astype(np.uint8)
    if effect == "Blur background":
        background = cv2.GaussianBlur(rgb, (35, 35), 0)
        return np.where(mask[..., None], rgb, background).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([40, 230, 120])
    ).astype(np.uint8)
    return overlay


def clear_camera():
    cap = st.session_state.pop("live_cap", None)
    if cap is not None:
        cap.release()
    st.session_state.pop("live_camera_index", None)
    st.session_state.pop("live_states", None)


@st.fragment(run_every=0.1)
def show_live(camera_index: int, session, threshold: float, effect: str):
    current_index = st.session_state.get("live_camera_index")
    cap = st.session_state.get("live_cap")
    if cap is None or current_index != camera_index or not cap.isOpened():
        clear_camera()
        cap = open_camera(camera_index)
        if not cap.isOpened():
            st.error(
                f"Cannot open webcam {camera_index}. Connect a camera or select another index."
            )
            return
        st.session_state["live_cap"] = cap
        st.session_state["live_camera_index"] = camera_index
        st.session_state["live_states"] = [
            np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES
        ]

    ok, frame_bgr = cap.read()
    if not ok:
        st.warning("Webcam is open but did not return a frame.")
        return

    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    states = st.session_state["live_states"]
    feeds = {"image": image}
    feeds.update({f"r{i + 1}": state for i, state in enumerate(states)})

    start = time.perf_counter()
    logits, *next_states = session.run(
        ["logits", "r1o", "r2o", "r3o", "r4o"], feeds
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    st.session_state["live_states"] = next_states

    logits = cv2.resize(
        logits[0, 0].astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    result = apply_effect(frame_rgb, probability, threshold, effect)

    left, right = st.columns(2)
    with left:
        st.image(frame_rgb, caption=f"Webcam input · {width}×{height}", use_container_width=True)
    with right:
        st.image(result, caption=f"SLIM50_SVD50 · {effect}", use_container_width=True)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Inference latency", f"{latency_ms:.1f} ms")
    metric2.metric("Model FPS", f"{1000.0 / latency_ms:.1f}" if latency_ms else "—")
    metric3.metric("Model input", "256 × 256")


st.title("SLIM50_SVD50 · Real-time human segmentation")
st.caption("เลือก webcam แล้วเริ่มสตรีมเพื่อดูภาพต้นฉบับและผล segmentation แบบสด")

if MODEL_PATH is None:
    st.error("ไม่พบไฟล์ SLIM50_SVD50.onnx ใน compression/outputs")
    st.stop()

with st.sidebar:
    st.header("ตั้งค่า")
    camera_index = st.selectbox("Webcam input", options=list(range(6)), index=0)
    provider_choice = st.selectbox("ONNX Runtime provider", ["Auto", "CUDA", "CPU"])
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
    effect = st.selectbox(
        "แสดงผล",
        ["Mask overlay", "Binary mask", "Green background", "Blur background"],
    )
    run_live = st.toggle("เริ่ม webcam แบบ realtime", value=False)
    st.caption(f"Model: {MODEL_PATH.name}")

try:
    session, active_providers = load_session(str(MODEL_PATH), provider_choice)
    st.caption(f"ONNX Runtime: {', '.join(active_providers)}")
except Exception as exc:
    clear_camera()
    st.error(f"โหลดโมเดลไม่สำเร็จ: {exc}")
    st.stop()

if run_live:
    show_live(camera_index, session, threshold, effect)
else:
    clear_camera()
    st.info("เลือกหมายเลข webcam แล้วเปิดสวิตช์ “เริ่ม webcam แบบ realtime”")
