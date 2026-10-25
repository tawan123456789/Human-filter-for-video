"""Export and evaluate all final RVM variants as ONNX FP16 at 512x512."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .core import (DEFAULT_CHECKPOINT, ROOT, atomic_json, common_parser,
                   loaders, param_count)
from .quantization import (check_onnx_parity, convert_onnx_fp16,
                           create_ort_session, export_stateful_onnx,
                           load_variant, ort_dataset_iou)


SOURCE = ROOT / "compression" / "outputs" / "final_fp16"
DEST = ROOT / "compression" / "outputs" / "final_512"
VARIANTS = [
    ("C0", "baseline", "baseline_512"),
    ("SVD50", "model_a_svd50", "model_a_svd50_512"),
    ("SLIM50_SVD50", "model_b_slim50_svd50", "model_b_slim50_svd50_512"),
]


def main() -> None:
    parser = common_parser(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.size = 512
    args.batch_size = 1
    args.output_dir = SOURCE  # Load the already-trained best checkpoints.
    args.checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINT)
    args.rvm_source = args.rvm_source or ROOT / "third_party" / "RobustVideoMatting"
    args.split_manifest = SOURCE / "midterm_split_manifest.json"

    # The same frozen validation IDs are resized to 512 for a fair, reproducible comparison.
    datasets, manifest = loaders(args, DEST)
    summary = []
    for run_id, folder, stem in VARIANTS:
        print(f"\n[{run_id}] Load best checkpoint and export 512x512 graph", flush=True)
        model = load_variant(args, run_id).cpu().eval()
        target = DEST / folder
        target.mkdir(parents=True, exist_ok=True)
        fp32_path = target / f"{stem}_fp32.onnx"
        fp16_path = target / f"{stem}_fp16.onnx"
        export_stateful_onnx(model, fp32_path, resolution=512, opset_version=args.opset)
        conversion = convert_onnx_fp16(fp32_path, fp16_path)

        session = create_ort_session(fp16_path, require_cuda=True)
        parity = check_onnx_parity(model, session, 512, atol=None)
        metrics = ort_dataset_iou(session, datasets["val"])

        # Each measured request advances RVM recurrent states, as in streaming inference.
        input_specs = session.get_inputs()
        image = np.random.default_rng(20261024).random((1, 3, 512, 512), dtype=np.float32)
        states = [np.zeros(tuple(int(dim) for dim in spec.shape), dtype=np.float32)
                  for spec in input_specs[1:]]

        def infer():
            nonlocal states
            outputs = session.run(None, {"image": image, **{f"r{i+1}": s for i, s in enumerate(states)}})
            states = outputs[1:]

        for _ in range(args.warmup):
            infer()
        durations = []
        for _ in range(args.iters):
            started = time.perf_counter()
            infer()
            durations.append((time.perf_counter() - started) * 1000.0)
        durations = np.asarray(durations)
        timing = {
            "provider": session.get_providers()[0],
            "latency_ms_p50": float(np.percentile(durations, 50)),
            "latency_ms_p95": float(np.percentile(durations, 95)),
            "latency_ms_mean": float(durations.mean()),
            "fps_from_mean_latency": float(1000.0 / durations.mean()),
            "warmup": args.warmup,
            "iterations": args.iters,
            "stateful": True,
        }
        result = {
            "variant": run_id,
            "input_size": [512, 512],
            "validation_split_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
            "validation": metrics,
            "stateful_5_frame_parity": parity,
            "benchmark": timing,
            "conversion": conversion,
            "onnx_fp16_bytes": fp16_path.stat().st_size,
            "parameter_count": param_count(model),
            "onnx_fp32_path": str(fp32_path),
            "onnx_fp16_path": str(fp16_path),
        }
        atomic_json(result, target / "metrics_512.json")
        summary.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del session, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(summary, DEST / "comparison_512.json")
    print(f"\nSaved 512x512 models and measurements under {DEST}", flush=True)


if __name__ == "__main__":
    main()
