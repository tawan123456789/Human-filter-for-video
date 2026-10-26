#!/usr/bin/env python3
"""Diagnostic & Benchmark tool for Jetson Nano.

Checks whether GPU (CUDA/TensorRT) is actually active and measures real inference FPS.
Run on Jetson Nano with: python3 jetson_nano/check_jetson_env.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("=" * 70)
    print(" Jetson Nano Inference Environment Diagnostic & Speed Test")
    print("=" * 70)

    # 1. System info
    print(f"[+] Python executable : {sys.executable}")
    print(f"[+] Python version    : {sys.version.split()[0]}")
    print(f"[+] Platform          : {sys.platform}")

    # 2. Check ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"[+] ONNX Runtime ver  : {ort.__version__}")
        available = ort.get_available_providers()
        print(f"[+] Available Providers: {available}")
    except ImportError as e:
        print(f"[-] ONNX Runtime NOT installed: {e}")
        print("    Install the provided ARM64 wheel: pip3 install jetson_nano/wheels/onnxruntime_gpu-*.whl")
        return 1

    # Check CUDA provider
    has_cuda = "CUDAExecutionProvider" in available
    has_trt = "TensorrtExecutionProvider" in available

    if not has_cuda and not has_trt:
        print("\n" + "!" * 70)
        print("[-] DANGER: Neither CUDA nor TensorRT execution provider is available!")
        print("    This is why your Jetson Nano gets 3-4 FPS!")
        print("    Fixes:")
        print("    1. Install onnxruntime-gpu wheel (NOT plain onnxruntime):")
        print("       pip3 install jetson_nano/wheels/onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl")
        print("    2. Make sure CUDA library paths are exported:")
        print("       export PATH=/usr/local/cuda/bin:$PATH")
        print("       export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print("!" * 70 + "\n")
    else:
        print(f"[V] GPU Provider ready: CUDA={has_cuda}, TensorRT={has_trt}")

    # 3. Model Benchmark
    model_path = ROOT / "jetson_nano" / "models" / "default_model.onnx"
    if not model_path.is_file():
        model_path = ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_svd50_fp16.onnx"

    if not model_path.is_file():
        print(f"[-] No model found at {model_path}")
        return 1

    print(f"\n[+] Testing Model: {model_path.name} ({model_path.stat().st_size / 1e6:.2f} MB)")

    # Session creation
    providers = []
    if has_trt:
        providers.append("TensorrtExecutionProvider")
    if has_cuda:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(str(model_path), providers=providers)
    active = sess.get_providers()
    print(f"[+] Active Provider    : {active}")

    # Build dummy input
    inputs = sess.get_inputs()
    feed = {}
    for inp in inputs:
        shape = [int(x) if isinstance(x, int) and x > 0 else 1 for x in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=np.float32)

    # Warmup
    print("[+] Warming up (10 runs)...")
    for _ in range(10):
        outs = sess.run(None, feed)
        # feed back recurrent states
        for i, s_name in enumerate(["r1", "r2", "r3", "r4"]):
            if s_name in feed and i + 1 < len(outs):
                feed[s_name] = outs[i + 1]

    # Benchmark 50 iterations
    iters = 50
    print(f"[+] Measuring latency over {iters} recurrent frames...")
    durations = []
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = sess.run(None, feed)
        durations.append((time.perf_counter() - t0) * 1000)
        for i, s_name in enumerate(["r1", "r2", "r3", "r4"]):
            if s_name in feed and i + 1 < len(outs):
                feed[s_name] = outs[i + 1]

    mean_ms = float(np.mean(durations))
    p50_ms = float(np.median(durations))
    p95_ms = float(np.percentile(durations, 95))
    fps = 1000.0 / mean_ms

    print("-" * 50)
    print(f"  Mean Latency : {mean_ms:.2f} ms")
    print(f"  p50 Latency  : {p50_ms:.2f} ms")
    print(f"  p95 Latency  : {p95_ms:.2f} ms")
    print(f"  Inference FPS: {fps:.2f} FPS")
    print("-" * 50)

    if fps < 10:
        print("[!] Warning: FPS is below 10. Check that CPU clocks are maxed via 'sudo jetson_clocks'")
    else:
        print(f"[V] Excellent! Your Jetson Nano is achieving {fps:.1f} FPS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
