"""Streamlit app for side-by-side testing of the three final ONNX FP16 RVMs."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "compression" / "outputs" / "final_fp16"
DATA = ROOT / "data" / "supervisely"
MANIFEST = FINAL / "midterm_split_manifest.json"
MODELS = {
    "Original baseline": FINAL / "baseline" / "baseline_fp16.onnx",
    "SVD50 + KD": FINAL / "model_a_svd50" / "model_a_svd50_fp16.onnx",
    "SLIM50 + SVD50 + KD": FINAL / "model_b_slim50_svd50" / "model_b_slim50_svd50_fp16.onnx",
}
SIZE = 256


@st.cache_resource(show_spinner="Loading ONNX models...")
def get_sessions(provider: str):
    available = ort.get_available_providers()
    selected = provider if provider in available else "CPUExecutionProvider"
    sessions = {}
    for name, path in MODELS.items():
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model missing: {path}")
        sessions[name] = ort.InferenceSession(str(path), providers=[selected, "CPUExecutionProvider"])
    return sessions, selected, available


def resize_rgb(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    resized = Image.fromarray(original).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return original, x


def resize_mask(mask: Image.Image) -> np.ndarray:
    gray = np.asarray(mask.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=np.uint8)
    return gray > (0 if gray.max() <= 1 else 127)


def zero_states(session):
    states = []
    for item in session.get_inputs()[1:]:
        shape = [int(x) if isinstance(x, (int, np.integer)) and int(x) > 0 else 1 for x in item.shape]
        states.append(np.zeros(shape, dtype=np.float32))
    return states


def infer_one(session, x: np.ndarray, gt: np.ndarray | None, repeats: int):
    states = zero_states(session)
    # Warm-up on a copy of the zero-state path; benchmark carries states across frames.
    session.run(None, {"image": x, **{f"r{i+1}": s for i, s in enumerate(states)}})
    times = []
    logits = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = session.run(None, {"image": x, **{f"r{i+1}": s for i, s in enumerate(states)}})
        times.append((time.perf_counter() - start) * 1000)
        logits, states = output[0], output[1:]
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -80, 80)))
    pred = prob >= 0.5
    metrics = {"IoU": None, "Dice": None, "Precision": None, "Recall": None}
    if gt is not None:
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        metrics = {
            "IoU": float(tp / max(1, tp + fp + fn)),
            "Dice": float(2 * tp / max(1, 2 * tp + fp + fn)),
            "Precision": float(tp / max(1, tp + fp)),
            "Recall": float(tp / max(1, tp + fn)),
        }
    return pred, metrics, float(np.mean(times)), float(np.percentile(times, 95))


def pair_for_id(sample_id: str):
    # The repository's paired dataset uses image/mask stems as IDs.
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        image = DATA / "images" / f"{sample_id}{ext}"
        if image.is_file():
            for mext in (".png", ".jpg", ".jpeg", ".bmp"):
                mask = DATA / "masks" / f"{sample_id}{mext}"
                if mask.is_file():
                    return image, mask
    return None


def overlay(original: np.ndarray, pred: np.ndarray) -> Image.Image:
    mask_img = Image.fromarray((pred.astype(np.uint8) * 255)).resize((original.shape[1], original.shape[0]), Image.Resampling.NEAREST)
    m = np.asarray(mask_img) > 127
    result = original.astype(np.float32).copy()
    green = np.array([40, 220, 100], dtype=np.float32)
    result[m] = result[m] * 0.55 + green * 0.45
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


st.set_page_config(page_title="RVM ONNX Model Comparison", layout="wide")
st.title("RVM ONNX FP16 · เปรียบเทียบ 3 โมเดล")
st.caption("ประมวลผลภาพที่ 256×256, RGB [0,1], reset recurrent states ที่ภาพเริ่มต้น และใช้ threshold sigmoid 0.5 เหมือนกันทุกโมเดล")

available = ort.get_available_providers()
provider_options = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in available]
if not provider_options:
    provider_options = available
chosen_provider = st.sidebar.selectbox("ONNX Runtime provider", provider_options)
repeat_count = st.sidebar.slider("วัด latency ต่อภาพ (รอบ, stateful)", 1, 30, 5)
sessions, active_provider, providers = get_sessions(chosen_provider)
st.sidebar.write(f"Active provider: `{active_provider}`")
st.sidebar.write(f"Installed providers: {', '.join(providers)}")
results_path = FINAL / "comparison" / "results.csv"
if results_path.is_file():
    benchmark_rows = pd.read_csv(results_path)
    st.subheader("ผล validation และขนาด artifact ที่บันทึกไว้")
    cols = [c for c in ("variant", "val_iou", "val_dice", "params", "structural_reduction_pct", "onnx_fp16_bytes") if c in benchmark_rows.columns]
    st.dataframe(benchmark_rows[cols].rename(columns={
        "variant": "Model", "val_iou": "Validation IoU", "val_dice": "Validation Dice",
        "params": "Parameters", "structural_reduction_pct": "Parameter reduction (%)",
        "onnx_fp16_bytes": "ONNX FP16 size (bytes)"
    }), hide_index=True, use_container_width=True)

tab_sample, tab_upload, tab_validate = st.tabs(["Validation sample", "Upload image", "Evaluate validation split"])

with tab_sample:
    if MANIFEST.is_file():
        ids = json.loads(MANIFEST.read_text(encoding="utf-8")).get("val", [])
        selected_id = st.selectbox("Validation image", ids)
        pair = pair_for_id(selected_id)
        if pair:
            image_path, mask_path = pair
            image = Image.open(image_path)
            gt = resize_mask(Image.open(mask_path))
            if st.button("Compare all models", key="sample_compare"):
                original, x = resize_rgb(image)
                cols = st.columns(3)
                rows = []
                for col, (name, session) in zip(cols, sessions.items()):
                    pred, metrics, mean_ms, p95_ms = infer_one(session, x, gt, repeat_count)
                    col.subheader(name)
                    col.image(overlay(original, pred), caption="Green = predicted foreground", use_container_width=True)
                    rows.append({"Model": name, **metrics, "Mean latency (ms)": mean_ms, "P95 latency (ms)": p95_ms})
                st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(precision=4), use_container_width=True)
        else:
            st.warning(f"ไม่พบคู่ภาพ/Mask ใน data/supervisely สำหรับ {selected_id}")
    else:
        st.error(f"ไม่พบ split manifest: {MANIFEST}")

with tab_upload:
    image_file = st.file_uploader("เลือกรูปภาพ", type=["png", "jpg", "jpeg", "webp", "bmp"], key="upload_image")
    mask_file = st.file_uploader("Ground truth mask (ถ้ามี)", type=["png", "jpg", "jpeg", "bmp"], key="upload_mask")
    if image_file and st.button("Compare uploaded image", key="upload_compare"):
        image = Image.open(io.BytesIO(image_file.getvalue()))
        original, x = resize_rgb(image)
        gt = resize_mask(Image.open(io.BytesIO(mask_file.getvalue()))) if mask_file else None
        cols = st.columns(3)
        rows = []
        for col, (name, session) in zip(cols, sessions.items()):
            pred, metrics, mean_ms, p95_ms = infer_one(session, x, gt, repeat_count)
            col.subheader(name)
            col.image(overlay(original, pred), caption="Green = predicted foreground", use_container_width=True)
            rows.append({"Model": name, **metrics, "Mean latency (ms)": mean_ms, "P95 latency (ms)": p95_ms})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(precision=4), use_container_width=True)
        if gt is None:
            st.info("อัปโหลด mask เพื่อแสดง IoU, Dice, Precision และ Recall")

with tab_validate:
    st.write("รัน validation split เดียวกันตาม `midterm_split_manifest.json`; รายงาน metric เฉลี่ยรายภาพและ latency เฉลี่ยต่อภาพ")
    if st.button("Run full validation comparison"):
        ids = json.loads(MANIFEST.read_text(encoding="utf-8"))["val"]
        sums = {name: {k: [] for k in ("IoU", "Dice", "Precision", "Recall", "Mean latency (ms)", "P95 latency (ms)")} for name in sessions}
        progress = st.progress(0.0)
        status = st.empty()
        missing = 0
        for index, sample_id in enumerate(ids):
            pair = pair_for_id(sample_id)
            if pair is None:
                missing += 1
                continue
            image_path, mask_path = pair
            original, x = resize_rgb(Image.open(image_path))
            gt = resize_mask(Image.open(mask_path))
            for name, session in sessions.items():
                _, metrics, mean_ms, p95_ms = infer_one(session, x, gt, 1)
                for key, value in metrics.items():
                    sums[name][key].append(value)
                sums[name]["Mean latency (ms)"].append(mean_ms)
                sums[name]["P95 latency (ms)"].append(p95_ms)
            progress.progress((index + 1) / len(ids))
            status.write(f"Processed {index + 1}/{len(ids)} validation images")
        rows = [{"Model": name, "Images": len(sums[name]["IoU"]), **{k: float(np.mean(v)) if v else None for k, v in sums[name].items()}} for name in sessions]
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(precision=4), use_container_width=True)
        if missing:
            st.warning(f"ข้าม {missing} ภาพที่ไม่พบคู่ image/mask")

