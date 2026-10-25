"""Create Jetson Nano (JetPack 4/CUDA 10.2) compatible ONNX FP16 models.

The Jetson Nano's bundled ONNX Runtime 1.11 supports ONNX opset 16, so these
graphs use opset 16 rather than the workstation graphs' opset 17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from .core import DEFAULT_CHECKPOINT, ROOT, atomic_json
from .quantization import convert_onnx_fp16, export_stateful_onnx, load_variant


SOURCE = ROOT / "compression" / "outputs" / "final_fp16"
VARIANTS = [
    ("C0", "baseline", "baseline_512_fp16.onnx"),
    ("SVD50", "model_a_svd50", "model_a_svd50_512_fp16.onnx"),
    ("SLIM50_SVD50", "model_b_slim50_svd50", "model_b_slim50_svd50_512_fp16.onnx"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, choices=(256, 512))
    args = parser.parse_args()
    size = args.size
    args.output_dir = SOURCE
    args.checkpoint = DEFAULT_CHECKPOINT
    args.rvm_source = ROOT / "third_party" / "RobustVideoMatting"
    destination = ROOT / "compression" / "outputs" / f"jetson_nano_{size}"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, folder, stem in VARIANTS:
        filename = stem.replace("_512", f"_{size}")
        print(f"Export {run_id} ({size}x{size}, opset 16)", flush=True)
        model = load_variant(args, run_id).cpu().eval()
        out = destination / folder
        out.mkdir(parents=True, exist_ok=True)
        fp32 = out / filename.replace("_fp16.onnx", "_fp32.onnx")
        fp16 = out / filename
        export_stateful_onnx(model, fp32, resolution=size, opset_version=16)
        conversion = convert_onnx_fp16(fp32, fp16)
        graph = onnx.load(str(fp16))
        onnx.checker.check_model(graph)
        opsets = {item.domain or "ai.onnx": item.version for item in graph.opset_import}
        if opsets.get("ai.onnx") != 16:
            raise RuntimeError(f"Unexpected ONNX opset for {fp16.name}: {opsets}")
        session = ort.InferenceSession(str(fp16), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider did not initialize for {fp16}: {session.get_providers()}")
        inputs = session.get_inputs()
        feed = {}
        for spec in inputs:
            shape = [int(x) if isinstance(x, (int, np.integer)) and int(x) > 0 else 1 for x in spec.shape]
            feed[spec.name] = np.zeros(shape, dtype=np.float32)
        feed[inputs[0].name] = np.random.default_rng(size).random((1, 3, size, size), dtype=np.float32)
        outputs = session.run(None, feed)
        row = {
            "run_id": run_id,
            "input_shape": inputs[0].shape,
            "opset": opsets,
            "provider": session.get_providers()[0],
            "logits_shape": outputs[0].shape,
            "artifact_bytes": fp16.stat().st_size,
            "conversion": conversion,
            "onnx_fp32_path": str(fp32),
            "onnx_fp16_path": str(fp16),
        }
        atomic_json(row, out / "jetson_export.json")
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    atomic_json(rows, destination / "exports.json")
    print(f"Jetson Nano models are ready under {destination}", flush=True)


if __name__ == "__main__":
    main()
