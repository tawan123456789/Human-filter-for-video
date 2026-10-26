"""Run a same-device FP32 streaming benchmark for ResNet50 and the four MobileNet RVM graphs.

For MobileNetV1/V2/V3-Small only the graph is benchmarked because their
fine-tuned checkpoints are not present in this checkout. Weight values do not
change the dense inference graph. MobileNetV3-Large uses the saved fine-tuned
checkpoint. This program does not calculate accuracy metrics.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from compression.core import load_rvm, param_count
from rvm_backbones import build_rvm


ROOT = Path(__file__).resolve().parents[1]
LARGE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_supervisely_best.pth"
DEFAULT_OUTPUT = ROOT / "compression" / "outputs" / "backbone_gpu_retest_20261027" / "fps.json"


@torch.inference_mode()
def benchmark(model, device, size=256, warmup=20, iters=100):
    model.eval().to(device)
    image = torch.rand((1, 3, size, size), device=device)
    state = [None] * 4
    for _ in range(warmup):
        _, *state = model(image, *state, segmentation_pass=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _, *state = model(image, *state, segmentation_pass=True)
        torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    mean = float(np.mean(elapsed))
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iters,
        "stateful": True,
        "latency_scope": "FP32 model only; input tensor already resident on GPU",
        "latency_ms_mean": mean,
        "latency_ms_p50": float(np.percentile(elapsed, 50)),
        "latency_ms_p95": float(np.percentile(elapsed, 95)),
        "fps_from_mean_latency": 1000.0 / mean,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda:0")
    if not LARGE_CHECKPOINT.is_file():
        raise FileNotFoundError(LARGE_CHECKPOINT)
    hub_source = Path(torch.hub.get_dir()) / "PeterL1n_RobustVideoMatting_master"
    if not (hub_source / "hubconf.py").is_file():
        raise FileNotFoundError(f"Cached official RVM source missing: {hub_source}")
    model_factories = {
        "RVM-ResNet50": lambda: torch.hub.load(
            str(hub_source), "resnet50", source="local",
            pretrained=True, progress=False,
        ),
        "RVM-MobileNetV3-Large official pretrained": lambda: torch.hub.load(
            str(hub_source), "mobilenetv3", source="local",
            pretrained=True, progress=False,
        ),
        "RVM-MobileNetV3-Large": lambda: load_rvm(
            LARGE_CHECKPOINT, rvm_source=ROOT / "third_party" / "RobustVideoMatting",
            segmentation_only=True,
        ),
        "RVM-MobileNetV3-Small": lambda: build_rvm("mobilenetv3_small", pretrained_backbone=False),
        "RVM-MobileNetV2": lambda: build_rvm("mobilenetv2", pretrained_backbone=False),
        "RVM-MobileNetV1": lambda: build_rvm("mobilenetv1", pretrained_backbone=False),
    }
    result = {
        "date": "2026-10-27",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "input_size": [256, 256],
        "batch_size": 1,
        "precision": "FP32",
        "test_split_used": False,
        "models": {},
    }
    for name, factory in model_factories.items():
        model = factory().to(device).eval()
        result["models"][name] = {
            "parameters": param_count(model),
            "weights": {
                "RVM-ResNet50": "official pretrained",
                "RVM-MobileNetV3-Large official pretrained": "official pretrained",
                "RVM-MobileNetV3-Large": "fine-tuned checkpoint",
            }.get(name, "architecture-only benchmark; no fine-tuned checkpoint available"),
            "benchmark": benchmark(model, device),
        }
        print(name, json.dumps(result["models"][name], indent=2))
        del model
        torch.cuda.empty_cache()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
