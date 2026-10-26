"""Compare official pretrained RVM ResNet50 and MobileNetV3-Large baselines.

Uses the frozen Midterm validation manifest and the same segmentation metrics
as compression.core. It never evaluates the test split.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.core import common_parser, loaders, load_rvm, param_count


ROOT = Path(__file__).resolve().parents[1]
RVM_SOURCE = ROOT / "third_party" / "RobustVideoMatting"
SPLIT_MANIFEST = ROOT / "compression" / "outputs" / "final_fp16" / "midterm_split_manifest.json"
FINETUNED_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_supervisely_best.pth"
OUTPUT = ROOT / "compression" / "outputs" / "baseline_backbone_resnet50" / "result.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    image_iou, image_dice, image_bce = [], [], []
    tp_total = fp_total = fn_total = 0
    pixel_count = 0
    for image, mask, _ in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, *_ = model(image, segmentation_pass=True)
        if logits.shape != mask.shape:
            raise RuntimeError(f"Output shape {logits.shape} != mask shape {mask.shape}")
        logits = logits.float()
        pred, gt = logits.sigmoid() >= 0.5, mask >= 0.5
        dims = (1, 2, 3)
        tp = (pred & gt).sum(dims).float()
        fp = (pred & ~gt).sum(dims).float()
        fn = (~pred & gt).sum(dims).float()
        union = tp + fp + fn
        image_iou.extend(torch.where(union > 0, tp / union.clamp_min(1), 1).cpu().tolist())
        denominator = 2 * tp + fp + fn
        image_dice.extend(torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 1).cpu().tolist())
        image_bce.extend(F.binary_cross_entropy_with_logits(logits, mask, reduction="none").mean(dims).cpu().tolist())
        tp_total += tp.sum().item()
        fp_total += fp.sum().item()
        fn_total += fn.sum().item()
        pixel_count += mask.numel()
    return {
        "count": len(image_iou),
        "iou_mean_per_image": float(np.mean(image_iou)),
        "dice_mean_per_image": float(np.mean(image_dice)),
        "bce_mean_per_image": float(np.mean(image_bce)),
        "precision_global": tp_total / max(1, tp_total + fp_total),
        "recall_global": tp_total / max(1, tp_total + fn_total),
        "global_iou": tp_total / max(1, tp_total + fp_total + fn_total),
    }


@torch.inference_mode()
def benchmark(model, device, size, warmup=20, iters=100):
    model.eval().to(device)
    image = torch.rand((1, 3, size, size), device=device)
    state = [None] * 4
    for _ in range(warmup):
        _, *state = model(image, *state, segmentation_pass=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, *state = model(image, *state, segmentation_pass=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "warmup_iterations": warmup,
        "measured_iterations": iters,
        "stateful": True,
        "timing_scope": "model inference only; tensor already resident on GPU",
        "latency_ms_mean": float(np.mean(durations)),
        "latency_ms_p50": float(np.percentile(durations, 50)),
        "latency_ms_p95": float(np.percentile(durations, 95)),
        "fps_from_mean_latency": float(1000 / np.mean(durations)),
    }
    if device.type == "cuda":
        result["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return result


def main():
    parser = common_parser(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(f"Frozen split manifest missing: {SPLIT_MANIFEST}")
    if not FINETUNED_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Fine-tuned reference checkpoint missing: {FINETUNED_CHECKPOINT}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    # Reuse the cached official source, downloading only official ResNet50 weights if absent.
    hub_source = Path(torch.hub.get_dir()) / "PeterL1n_RobustVideoMatting_master"
    if not (hub_source / "hubconf.py").is_file():
        raise FileNotFoundError(f"Cached Torch Hub RVM source missing: {hub_source}")
    validation_loader, manifest = loaders(
        argparse.Namespace(
            data_dir=ROOT / "data" / "supervisely",
            images_dir=None,
            masks_dir=None,
            split_manifest=SPLIT_MANIFEST,
            seed=42,
            size=args.size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
        ),
        OUTPUT.parent,
    )
    models = {
        "RVM-ResNet50 official pretrained": torch.hub.load(
            str(hub_source), "resnet50", source="local", pretrained=True, progress=True
        ),
        "RVM-MobileNetV3-Large official pretrained": torch.hub.load(
            str(hub_source), "mobilenetv3", source="local", pretrained=True, progress=False
        ),
        "RVM-MobileNetV3-Large Supervisely fine-tuned": load_rvm(
            FINETUNED_CHECKPOINT, rvm_source=RVM_SOURCE, segmentation_only=True
        ),
    }
    checkpoint_paths = {
        "RVM-ResNet50 official pretrained": Path(torch.hub.get_dir()) / "checkpoints" / "rvm_resnet50.pth",
        "RVM-MobileNetV3-Large official pretrained": Path(torch.hub.get_dir()) / "checkpoints" / "rvm_mobilenetv3.pth",
        "RVM-MobileNetV3-Large Supervisely fine-tuned": FINETUNED_CHECKPOINT,
    }
    results = {
        "experiment": "official RVM baseline backbone comparison",
        "date": "2026-10-27",
        "validation_only": True,
        "test_evaluated": False,
        "input_size": [args.size, args.size],
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "split_counts": {"train": 2133, "validation": 266, "test": 268},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "models": {},
    }
    for name, model in models.items():
        model = model.to(device).eval()
        metrics = evaluate(model, validation_loader["val"], device)
        timing = benchmark(model, device, args.size, args.warmup, args.iters)
        checkpoint = checkpoint_paths[name]
        results["models"][name] = {
            "parameters": param_count(model),
            "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            **metrics,
            "benchmark": timing,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(name, json.dumps(results["models"][name], indent=2))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
