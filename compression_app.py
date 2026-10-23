"""Compare every trained and exported RVM compression candidate."""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "compression" / "outputs"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_supervisely_best.pth"
RVM_SOURCE = ROOT / "third_party" / "RobustVideoMatting"
MODEL_SIZE = 256

st.set_page_config(page_title="RVM Compression Lab", page_icon="🎛️", layout="wide")
st.markdown("""
<style>
  .stApp { background: #0b1020; }
  [data-testid="stMetric"] { background: #111a2d; border: 1px solid #263653;
    padding: 14px; border-radius: 12px; }
  .small-note { color: #9db0cc; }
</style>
""", unsafe_allow_html=True)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def model_catalog():
    """Build the list from artifacts on disk so incomplete exports stay visible."""
    items = []
    baseline = read_json(OUT / "baseline.json", {}) or {}
    baseline_metrics = baseline.get("validation", {})
    base_path = OUT / "exports" / "C0_segment_fp32_state.pth"
    if base_path.is_file():
        items.append({
            "key": "C0|pytorch_fp32", "label": "C0 · Original FP32",
            "run_id": "C0", "format": "pytorch_fp32", "path": base_path,
            "size": base_path.stat().st_size, "val_iou": baseline_metrics.get("iou"),
            "val_dice": baseline_metrics.get("dice"), "speed": None,
            "description": "Fine-tuned MobileNetV3-Large baseline, segmentation-only state.",
        })

    for result_path in sorted((OUT / "runs").glob("*/result.json")):
        result = read_json(result_path, {}) or {}
        run_id = result.get("run_id", result_path.parent.name)
        path = result_path.parent / "deploy_fp32_state.pth"
        if not path.is_file():
            continue
        spec = result.get("spec", {})
        label = f"{run_id} · FP32"
        items.append({
            "key": f"{run_id}|pytorch_fp32", "label": label,
            "run_id": run_id, "format": "pytorch_fp32", "path": path,
            "size": path.stat().st_size, "val_iou": result.get("val_iou"),
            "val_dice": result.get("val_dice"), "speed": result.get("fps"),
            "description": f"Compression operations: {spec.get('ops', []) or 'none (control)'}",
        })

    export_labels = {
        "fp16.json": ("pytorch_fp16", "FP16"),
        "onnx_fp32.json": ("onnx_fp32", "ONNX FP32"),
        "int8.json": ("onnx_int8_qdq", "ONNX INT8"),
    }
    for meta_path in sorted((OUT / "exports").glob("*/*.json")):
        if meta_path.name not in export_labels:
            continue
        meta = read_json(meta_path, {}) or {}
        path = Path(meta.get("path", ""))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            continue
        run_id = meta.get("run_id", meta_path.parent.name)
        fmt, fmt_label = export_labels[meta_path.name]
        speed = meta.get("cpu_fps") if fmt == "onnx_int8_qdq" else meta.get("fps")
        items.append({
            "key": f"{run_id}|{fmt}", "label": f"{run_id} · {fmt_label}",
            "run_id": run_id, "format": fmt, "path": path,
            "size": meta.get("deploy_bytes", path.stat().st_size),
            "val_iou": meta.get("val_iou"), "val_dice": meta.get("val_dice"),
            "speed": speed,
            "description": meta.get("note", "Exported compression artifact."),
        })
    return items


@st.cache_resource(show_spinner=False)
def load_runtime(key: str, device_name: str):
    item = next((x for x in model_catalog() if x["key"] == key), None)
    if item is None:
        raise FileNotFoundError(f"Model artifact disappeared: {key}")
    if item["format"].startswith("onnx_"):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(item["path"]), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        return {"kind": "onnx", "session": session, "item": item}

    from compression.quantization import load_variant
    args = SimpleNamespace(checkpoint=CHECKPOINT, rvm_source=RVM_SOURCE,
                            output_dir=OUT)
    model = load_variant(args, item["run_id"])
    if item["format"] == "pytorch_fp16":
        weights = torch.load(item["path"], map_location="cpu", weights_only=True)
        model.half()
        model.load_state_dict(weights, strict=True)
        # CPU kernels use FP32; weights retain the rounding of the FP16 artifact.
        if device_name == "cpu":
            model.float()
    model.eval().to(torch.device(device_name))
    return {"kind": "torch", "model": model, "item": item,
            "dtype": next(model.parameters()).dtype}


def initial_states(runtime):
    if runtime["kind"] == "torch":
        return [None, None, None, None]
    states = []
    for input_meta in runtime["session"].get_inputs()[1:]:
        shape = tuple(int(v) if isinstance(v, int) else 1 for v in input_meta.shape)
        states.append(np.zeros(shape, dtype=np.float32))
    return states


def infer_frame(runtime, rgb: np.ndarray, states, device_name: str):
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    x_np = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    item = runtime["item"]
    if runtime["kind"] == "torch":
        model = runtime["model"]
        dtype = runtime["dtype"]
        x = torch.from_numpy(x_np).to(device_name, dtype=dtype)
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits, *new_states = model(x, *states, segmentation_pass=True)
            probability = logits.float().sigmoid()[0, 0].cpu().numpy()
        if device_name == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000.0
        next_states = new_states
    else:
        session = runtime["session"]
        feeds = {"image": x_np}
        feeds.update({f"r{i+1}": state for i, state in enumerate(states)})
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        latency = (time.perf_counter() - start) * 1000.0
        logits, *next_states = outputs
        logits = np.asarray(logits, dtype=np.float32)
        probability = (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))[0, 0]
    probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
    return probability, next_states, latency


def make_view(rgb, probability, threshold, effect):
    mask = probability >= threshold
    if effect == "Binary mask":
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    if effect == "Green background":
        bg = np.zeros_like(rgb)
        bg[:] = (0, 210, 40)
    elif effect == "Blur background":
        bg = cv2.GaussianBlur(rgb, (35, 35), 0)
    elif effect == "Mask overlay":
        overlay = rgb.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([40, 230, 120])).astype(np.uint8)
        return overlay
    else:
        return rgb
    return np.where(mask[..., None], rgb, bg).astype(np.uint8)


def format_iou(value):
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def artifact_table(items):
    rows = []
    for item in items:
        rows.append({
            "Model": item["label"], "Format": item["format"],
            "File size (MB)": round(item["size"] / 1_000_000, 2),
            "Val IoU": format_iou(item.get("val_iou")),
            "Val Dice": format_iou(item.get("val_dice")),
            "Recorded speed (FPS)": round(float(item["speed"]), 1) if item.get("speed") else "—",
        })
    return pd.DataFrame(rows)


st.title("🎛️ RVM Compression Lab")
st.caption("เปรียบเทียบทุก checkpoint และ export ที่มีใน compression/outputs โดยใช้ validation metrics ที่บันทึกจาก pipeline")

items = model_catalog()
if not items:
    st.error("ยังไม่พบ model artifacts ใน compression/outputs กรุณารัน compression search และ quantization ก่อน")
    st.stop()

frozen = read_json(OUT / "reports" / "final_test.json", {}) or {}
selection = frozen.get("frozen_selection", {})
test_metrics = frozen.get("test", {})
if selection:
    st.success(
        f"Frozen candidate: {selection.get('id')} · {selection.get('format')} · "
        f"{selection.get('bytes', 0) / 1_000_000:.2f} MB · "
        f"Validation IoU {format_iou(selection.get('val_iou'))} · "
        f"Test audit IoU {format_iou(test_metrics.get('iou'))}"
    )

with st.expander("ผล validation ของทุก model", expanded=True):
    st.dataframe(artifact_table(items), use_container_width=True, hide_index=True)
    st.caption("FPS เป็นค่าที่บันทึกจาก benchmark ของแต่ละ format/hardware; เปรียบเทียบตรงกันได้เมื่อ platform เดียวกัน")

labels = [x["label"] for x in items]
by_label = {x["label"]: x for x in items}
default_labels = labels

with st.sidebar:
    st.header("เปรียบเทียบ")
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_name = st.selectbox("PyTorch device", device_options, index=0)
    if device_name == "cuda":
        st.caption(torch.cuda.get_device_name(0))
    selected_labels = st.multiselect("Models", labels, default=default_labels)
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
    effect = st.selectbox("Preview", ["Mask overlay", "Green background", "Blur background", "Binary mask", "Original"])
    st.caption("ONNX INT8/FP32 uses CPUExecutionProvider. FP16 on CPU runs with FP32 kernels using FP16-rounded weights.")

if not selected_labels:
    st.info("เลือกอย่างน้อยหนึ่ง model จากแถบด้านซ้าย")
    st.stop()
selected = [by_label[label] for label in selected_labels]
tab_image, tab_video = st.tabs(["🖼️ Compare image", "🎞️ Compare video"])

with tab_image:
    uploaded = st.file_uploader("อัปโหลดภาพ", type=["png", "jpg", "jpeg", "webp", "bmp"], key="image_upload")
    camera_image = st.camera_input("หรือถ่ายภาพจากกล้อง", key="camera_upload")
    source = camera_image or uploaded
    if source is not None:
        pil = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        rgb = np.asarray(pil)
        st.image(rgb, caption=f"Input · {pil.width}×{pil.height}", use_container_width=True)
        col_count = min(4, len(selected))
        cols = st.columns(col_count)
        metrics = []
        progress = st.progress(0, text="กำลังประมวลผลภาพ...")
        errors = []
        for index, item in enumerate(selected, start=1):
            try:
                runtime = load_runtime(item["key"], device_name)
                probability, _, latency = infer_frame(runtime, rgb, initial_states(runtime), device_name)
                view = make_view(rgb, probability, threshold, effect)
                with cols[(index - 1) % col_count]:
                    st.image(view, caption=f"{item['label']} · {latency:.1f} ms", use_container_width=True)
                metrics.append({"Model": item["label"], "Latency (ms)": round(latency, 2),
                                "FPS (single-frame reciprocal)": round(1000 / latency, 1) if latency else None,
                                "Val IoU": format_iou(item.get("val_iou")),
                                "File MB": round(item["size"] / 1_000_000, 2)})
            except Exception as exc:
                errors.append(f"{item['label']}: {exc}")
            progress.progress(index / len(selected), text=f"ประมวลผล {index}/{len(selected)}")
        progress.empty()
        if metrics:
            st.subheader("Inference comparison")
            st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        for error in errors:
            st.warning(error)

with tab_video:
    st.caption("ระบบสุ่มประมวลผลเป็นช่วงเฟรมเพื่อให้เทียบหลาย model ได้เร็ว โดยคง recurrent state ระหว่างเฟรมตัวอย่างที่ประมวลผล")
    video = st.file_uploader("อัปโหลดวิดีโอ", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    max_frames = st.slider("จำนวนเฟรมตัวอย่างสูงสุด", 8, 120, 40, 8)
    if video is not None:
        run_video = st.button("เปรียบเทียบวิดีโอ", type="primary")
        if run_video:
            temp_input = None
            temp_output = None
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix or ".mp4", delete=False) as temp:
                    temp.write(video.getvalue())
                    temp_input = temp.name
                cap = cv2.VideoCapture(temp_input)
                if not cap.isOpened():
                    raise RuntimeError("เปิดไฟล์วิดีโอไม่สำเร็จ")
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                stride = max(1, math.ceil(frame_count / max_frames)) if frame_count else 1
                runtimes = {}
                states = {}
                load_errors = []
                with st.spinner(f"โหลด {len(selected)} model..."):
                    for item in selected:
                        try:
                            runtimes[item["key"]] = load_runtime(item["key"], device_name)
                            states[item["key"]] = initial_states(runtimes[item["key"]])
                        except Exception as exc:
                            load_errors.append(f"{item['label']}: {exc}")
                if not runtimes:
                    raise RuntimeError("โหลด model ไม่สำเร็จ")

                ret, first = cap.read()
                if not ret:
                    raise RuntimeError("ไม่พบเฟรมในวิดีโอ")
                h, w = first.shape[:2]
                panel_w = min(480, w)
                panel_h = max(1, round(h * panel_w / w))
                grid_cols = min(3, len(runtimes))
                grid_rows = math.ceil(len(runtimes) / grid_cols)
                canvas_w, canvas_h = panel_w * grid_cols, panel_h * grid_rows
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
                    temp_output = temp.name
                out_fps = max(1.0, source_fps / stride)
                writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                         (canvas_w, canvas_h))
                if not writer.isOpened():
                    raise RuntimeError("เปิดตัวเขียน MP4 ไม่สำเร็จ")
                total = frame_count if frame_count > 0 else max_frames * stride
                progress = st.progress(0, text="กำลังเปรียบเทียบวิดีโอ...")
                frame_idx = 0
                rendered = 0
                frame = first
                while True:
                    if frame_idx % stride == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        panels = []
                        for item in selected:
                            runtime = runtimes.get(item["key"])
                            if runtime is None:
                                continue
                            probability, states[item["key"]], _ = infer_frame(
                                runtime, rgb_frame, states[item["key"]], device_name)
                            preview = make_view(rgb_frame, probability, threshold, effect)
                            preview = cv2.resize(preview, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                            cv2.rectangle(preview, (0, 0), (panel_w, 30), (12, 18, 31), -1)
                            cv2.putText(preview, item["label"][:42], (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.48, (245, 248, 255), 1, cv2.LINE_AA)
                            panels.append(preview)
                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        for i, panel in enumerate(panels):
                            y0, x0 = (i // grid_cols) * panel_h, (i % grid_cols) * panel_w
                            canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel
                        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                        rendered += 1
                    frame_idx += 1
                    if frame_count and frame_idx >= frame_count:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % max(1, stride) == 0:
                        progress.progress(min(1.0, frame_idx / max(1, total)),
                                          text=f"ประมวลผลเฟรมตัวอย่าง · {rendered}/{max_frames}")
                writer.release()
                cap.release()
                progress.empty()
                for error in load_errors:
                    st.warning(error)
                st.success(f"สร้าง preview แล้ว {rendered} เฟรม (stride {stride}, ประมาณ {out_fps:.1f} FPS)")
                st.video(Path(temp_output).read_bytes())
            except Exception as exc:
                st.error(str(exc))
            finally:
                if 'cap' in locals():
                    cap.release()
                if 'writer' in locals():
                    writer.release()
                for temp_path in (temp_input, temp_output):
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink(missing_ok=True)

