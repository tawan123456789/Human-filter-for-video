"""Jetson Nano & Edge · Real-Time Human Matting & Segmentation with Streamlit.

Supports fine-tuned high-accuracy RVM models that outperform the distilled/compressed baseline:
  - RVM-MobileNetV1 (Fastest GPU: 185.6 FPS, IoU 93.18%) - Recommended for Jetson CUDA
  - RVM-MobileNetV2 (Edge Optimized: 9.67 MB, IoU 92.69%)
  - RVM-MobileNetV3-Large (Highest Precision: IoU 93.54%)
  - RVM-MobileNetV3-Small (Fastest CPU: 33.9 FPS, 6.39 MB, IoU 91.28%)
  - SLIM50_SVD50 (Distilled Baseline: IoU 91.16%, INT8 87.75%)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

# Known model registry for detailed comparative metrics
MODEL_INFO_REGISTRY = {
    "default_model.onnx": {
        "title": "Default Jetson Model (MobileNetV3-Large Distilled SVD50)",
        "params": "3.40 M",
        "size_mb": 13.6,
        "iou": 93.23,
        "dice": 96.20,
        "gpu_fps": 106.5,
        "cpu_fps": 18.2,
        "badge": "🏆 Default Model · Distilled 3-Large (IoU 93.23%, +2.07% over SLIM50)",
        "description": "Active default model: Distilled from MobileNetV3-Large with SVD Factorization (saves 9.2% params).",
    },
    "rvm_mobilenetv3_large_svd50.onnx": {
        "title": "RVM-MobileNetV3-Large (Distilled SVD50 - Superior)",
        "params": "3.40 M",
        "size_mb": 13.6,
        "iou": 93.23,
        "dice": 96.20,
        "gpu_fps": 106.5,
        "cpu_fps": 18.2,
        "badge": "🌟 Best Distilled 3-Large (+2.07% IoU over SLIM50_SVD50)",
        "description": "Distilled from MobileNetV3-Large using Knowledge Distillation and Low-Rank SVD (0.50). Retains full ConvGRU states for flicker-free video matting.",
    },
    "rvm_mobilenetv1.onnx": {
        "title": "RVM-MobileNetV1 (Fine-tuned Full Model)",
        "params": "4.21 M",
        "size_mb": 16.8,
        "iou": 93.18,
        "dice": 96.47,
        "gpu_fps": 185.6,
        "cpu_fps": 20.6,
        "badge": "⚡ Champion GPU FPS (185.6) · +2.02% IoU over SLIM50",
        "description": "Standard depthwise separable conv without residual branching overhead. Highest throughput on Jetson Nano CUDA.",
    },
    "rvm_mobilenetv2.onnx": {
        "title": "RVM-MobileNetV2 (Fine-tuned)",
        "params": "2.42 M",
        "size_mb": 9.67,
        "iou": 92.69,
        "dice": 96.19,
        "gpu_fps": 134.7,
        "cpu_fps": 22.8,
        "badge": "📱 Best Embedded Architecture (9.67 MB) · +1.53% IoU over Distilled",
        "description": "Inverted residuals with linear bottlenecks. Engineered for mobile DSP/GPU edge devices.",
    },
    "rvm_mobilenetv3_large.onnx": {
        "title": "RVM-MobileNetV3-Large (Official Pretrained)",
        "params": "3.74 M",
        "size_mb": 15.0,
        "iou": 93.54,
        "dice": 96.66,
        "gpu_fps": 110.0,
        "cpu_fps": 18.0,
        "badge": "💎 Highest Accuracy (IoU 93.54%) · +2.38% IoU over Distilled",
        "description": "Official RVM large architecture with full receptive field for fine hair and edge details.",
    },
    "rvm_mobilenetv3_small.onnx": {
        "title": "RVM-MobileNetV3-Small (Fine-tuned)",
        "params": "1.59 M",
        "size_mb": 6.39,
        "iou": 91.28,
        "dice": 95.43,
        "gpu_fps": 126.8,
        "cpu_fps": 33.9,
        "badge": "🪶 Ultra Light (6.39 MB) · Fastest CPU (33.9 FPS)",
        "description": "Minimal FLOPs architecture. Best for CPU-only edge boards like Raspberry Pi or Jetson CPU fallback.",
    },
    "SLIM50_SVD50.onnx": {
        "title": "SLIM50_SVD50 (Distilled / Compressed Baseline)",
        "params": "3.29 M",
        "size_mb": 13.4,
        "iou": 91.16,
        "dice": 95.05,
        "gpu_fps": 112.0,
        "cpu_fps": 47.5,
        "badge": "🧪 Distilled Baseline (IoU 91.16% / INT8 87.75%)",
        "description": "Compressed via channel slimming (0.50) + SVD factorization (0.50) + Knowledge Distillation.",
    },
}

st.set_page_config(
    page_title="Jetson Nano · Real-Time Human Matting",
    page_icon="🎥",
    layout="wide",
)


def discover_models() -> dict[str, Path]:
    """Find all available stateful ONNX models in the workspace."""
    search_dirs = [
        ROOT / "jetson_nano" / "models",
        ROOT / "outputs" / "checkpoints",
        ROOT / "compression" / "outputs" / "jetson_nano",
        ROOT / "compression" / "outputs" / "exports" / "SLIM50_SVD50",
    ]
    found = {}
    for sdir in search_dirs:
        if sdir.is_dir():
            for p in sorted(sdir.glob("*.onnx")):
                if p.name not in found:
                    found[p.name] = p
    return found


@st.cache_resource(show_spinner="Loading ONNX model session...")
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


def get_initial_states(session) -> list[np.ndarray]:
    """Dynamically initialize recurrent states based on session inputs."""
    states = []
    for inp in session.get_inputs()[1:]:
        shape = [int(dim) if isinstance(dim, int) else 1 for dim in inp.shape]
        states.append(np.zeros(shape, dtype=np.float32))
    return states


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
    if effect == "Cyberpunk Neon":
        neon_bg = np.zeros_like(rgb)
        h, w = rgb.shape[:2]
        neon_bg[:, :] = [20, 10, 40]  # dark violet
        # Add subtle horizontal grid lines
        neon_bg[::20, :] = [0, 220, 255]
        return np.where(mask[..., None], rgb, neon_bg).astype(np.uint8)
    if effect == "Black Cutout":
        background = np.zeros_like(rgb)
        return np.where(mask[..., None], rgb, background).astype(np.uint8)

    # Default: Mask overlay
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
    st.session_state.pop("live_model_name", None)


@st.fragment(run_every=0.08)
def show_live(camera_index: int, session, threshold: float, effect: str, model_name: str):
    current_index = st.session_state.get("live_camera_index")
    current_model = st.session_state.get("live_model_name")
    cap = st.session_state.get("live_cap")

    if cap is None or current_index != camera_index or current_model != model_name or not cap.isOpened():
        clear_camera()
        cap = open_camera(camera_index)
        if not cap.isOpened():
            st.error(
                f"Cannot open webcam {camera_index}. Connect a camera or select another index."
            )
            return
        st.session_state["live_cap"] = cap
        st.session_state["live_camera_index"] = camera_index
        st.session_state["live_model_name"] = model_name
        st.session_state["live_states"] = get_initial_states(session)

    ok, frame_bgr = cap.read()
    if not ok:
        st.warning("Webcam is open but did not return a frame.")
        return

    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    states = st.session_state.get("live_states")
    if states is None or len(states) != len(session.get_inputs()) - 1:
        states = get_initial_states(session)

    feeds = {"image": image}
    feeds.update({f"r{i + 1}": state for i, state in enumerate(states)})

    output_names = [out.name for out in session.get_outputs()]
    start = time.perf_counter()
    outputs = session.run(output_names, feeds)
    latency_ms = (time.perf_counter() - start) * 1000.0

    logits = outputs[0]
    st.session_state["live_states"] = outputs[1:]

    logits = cv2.resize(
        logits[0, 0].astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    result = apply_effect(frame_rgb, probability, threshold, effect)

    left, right = st.columns(2)
    with left:
        st.image(frame_rgb, caption=f"Webcam Input · {width}×{height}", use_container_width=True)
    with right:
        st.image(result, caption=f"{model_name} · {effect}", use_container_width=True)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Inference Latency", f"{latency_ms:.1f} ms")
    metric2.metric("Inference FPS", f"{1000.0 / latency_ms:.1f}" if latency_ms else "—")
    metric3.metric("Input Size", "256 × 256")
    metric4.metric("State Tensors", f"{len(st.session_state['live_states'])} recurrent states")


# -------------------------------------------------------------
# MAIN APP UI
# -------------------------------------------------------------

st.title("🎥 Jetson Nano & Edge · Real-Time Human Matting")
st.markdown(
    "แอปพลิเคชันสำหรับสตรีมมิ่งเว็บแคมและตัดพื้นหลังคนแบบเรียลไทม์บน Jetson Nano "
    "โดยใช้โมเดล **RVM Fine-tuned Encoders** ซึ่งให้ความแม่นยำ (IoU) และความเร็วสูงกว่าโมเดลที่ผ่านการกลั่น (Distilled Baseline)"
)

available_models = discover_models()

if not available_models:
    st.error(
        "❌ ไม่พบไฟล์โมเดล ONNX ในโปรเจกต์ กรุณารันคำสั่ง `python export_onnx.py --variant all` เพื่อ Export โมเดลที่ Fine-tuned ไว้ล่วงหน้า"
    )
    st.stop()

# Organize model display list, putting best fine-tuned models first
model_keys = list(available_models.keys())
priority_order = [
    "default_model.onnx",
    "rvm_mobilenetv3_large_svd50.onnx",
    "rvm_mobilenetv1.onnx",
    "rvm_mobilenetv2.onnx",
    "rvm_mobilenetv3_large.onnx",
    "rvm_mobilenetv3_small.onnx",
    "SLIM50_SVD50.onnx",
]
sorted_keys = sorted(
    model_keys,
    key=lambda k: priority_order.index(k) if k in priority_order else 99,
)

with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")

    selected_model_name = st.selectbox(
        "🧠 เลือกโมเดล ONNX",
        options=sorted_keys,
        format_func=lambda k: f"{MODEL_INFO_REGISTRY.get(k, {}).get('title', k)}",
        index=0,
    )
    selected_model_path = available_models[selected_model_name]

    provider_choice = st.selectbox(
        "🚀 Execution Provider",
        ["Auto", "CUDA", "CPU"],
        help="CUDA แนะนำสำหรับ Jetson Nano GPU / Nvidia Laptop; CPU สำหรับทดสอบทั่วไป",
    )
    threshold = st.slider("🎯 Mask Threshold", 0.1, 0.9, 0.5, 0.05)
    effect = st.selectbox(
        "🎨 เอฟเฟกต์พื้นหลัง",
        [
            "Green background",
            "Blur background",
            "Mask overlay",
            "Cyberpunk Neon",
            "Black Cutout",
            "Binary mask",
        ],
    )
    camera_index = st.selectbox("📷 Webcam Index", options=list(range(6)), index=0)
    run_live = st.toggle("🔴 เริ่มทำงาน Webcam แบบเรียลไทม์", value=False)

# Display Model Info Card
info = MODEL_INFO_REGISTRY.get(selected_model_name, {})
if info:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Test IoU", f"{info.get('iou', 0):.2f}%", delta=f"{info.get('iou', 0) - 91.16:+.2f}% vs Distilled" if info.get('iou') else None)
    col_b.metric("Dice Score", f"{info.get('dice', 0):.2f}%")
    col_c.metric("GPU Speed", f"{info.get('gpu_fps', 0)} FPS")
    col_d.metric("Model Size", f"{selected_model_path.stat().st_size / 1e6:.2f} MB")

    st.success(f"**จุดเด่น:** {info.get('badge', '')} — {info.get('description', '')}")

try:
    session, active_providers = load_session(str(selected_model_path), provider_choice)
    st.caption(f"⚡ Model Path: `{selected_model_path}` | Active Providers: `{', '.join(active_providers)}`")
except Exception as exc:
    clear_camera()
    st.error(f"โหลดโมเดลไม่สำเร็จ: {exc}")
    st.stop()

if run_live:
    show_live(camera_index, session, threshold, effect, selected_model_name)
else:
    clear_camera()
    st.info("💡 เลือกโมเดลและตั้งค่าที่แถบด้านซ้าย แล้วเปิดสวิตช์ **“เริ่มทำงาน Webcam แบบเรียลไทม์”** เพื่อเริ่มประมวลผล")
