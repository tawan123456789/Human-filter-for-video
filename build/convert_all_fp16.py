"""Convert all ONNX models in the repository to FP16 for Jetson Nano / Edge deployment.

Populates:
1. jetson_nano/models/ (*_fp16.onnx and updates default_model.onnx to FP16)
2. compression/outputs/jetson_nano_256/ (baseline, model_a_svd50, model_b_slim50_svd50)
3. compression/outputs/final_fp16/ (baseline, model_a_svd50, model_b_slim50_svd50)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil
import onnx
import onnxruntime as ort
import numpy as np

from compression.quantization import convert_onnx_fp16

# Models to convert
CONVERSIONS = [
    # Source ONNX, Target FP16 ONNX, Description
    (
        ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_large.onnx",
        ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_fp16.onnx",
        "Baseline RVM-MobileNetV3-Large"
    ),
    (
        ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_large_svd50.onnx",
        ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_svd50_fp16.onnx",
        "Model A (SVD50 + KD)"
    ),
    (
        ROOT / "compression" / "outputs" / "jetson_nano" / "SLIM50_SVD50.onnx",
        ROOT / "jetson_nano" / "models" / "SLIM50_SVD50_fp16.onnx",
        "Model B (SLIM50 + SVD50 + KD)"
    ),
    (
        ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv1.onnx",
        ROOT / "jetson_nano" / "models" / "rvm_mobilenetv1_fp16.onnx",
        "RVM-MobileNetV1 (Fastest GPU)"
    ),
    (
        ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv2.onnx",
        ROOT / "jetson_nano" / "models" / "rvm_mobilenetv2_fp16.onnx",
        "RVM-MobileNetV2 (Edge Optimized)"
    ),
    (
        ROOT / "outputs" / "checkpoints" / "rvm_mobilenetv3_small.onnx",
        ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_small_fp16.onnx",
        "RVM-MobileNetV3-Small (Ultra Light)"
    ),
]

def main():
    print("=" * 70)
    print("Converting ONNX models to FP16 for Jetson Nano / Edge Deployment")
    print("=" * 70)

    converted_files = {}

    for src, dst, desc in CONVERSIONS:
        if not src.is_file():
            print(f"[-] Source not found: {src}")
            continue

        print(f"\n[+] Converting {desc}...")
        print(f"    Source: {src.relative_to(ROOT)} ({src.stat().st_size / 1e6:.2f} MB)")
        res = convert_onnx_fp16(src, dst)
        print(f"    Target: {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.2f} MB)")
        print(f"    FP16 initializers: {res['fp16_initializers']}/{res['initializer_count']}")
        converted_files[desc] = dst

        # Quick validation with ONNX checker and ORT
        m = onnx.load(str(dst))
        onnx.checker.check_model(m)
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        feed = {}
        for inp in sess.get_inputs():
            shape = [int(x) if isinstance(x, int) and x > 0 else 1 for x in inp.shape]
            feed[inp.name] = np.zeros(shape, dtype=np.float32)
        sess.run(None, feed)
        print("    Validation: PASSED")

    # Set default_model.onnx to the best FP16 model (SVD50 FP16)
    svd50_fp16 = ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_svd50_fp16.onnx"
    default_model = ROOT / "jetson_nano" / "models" / "default_model.onnx"
    if svd50_fp16.is_file():
        shutil.copy2(svd50_fp16, default_model)
        print(f"\n[+] Updated default_model.onnx with SVD50 FP16 ({default_model.stat().st_size / 1e6:.2f} MB)")

    # Setup directories for top2_benchmark
    nano_256_dir = ROOT / "compression" / "outputs" / "jetson_nano_256"
    final_fp16_dir = ROOT / "compression" / "outputs" / "final_fp16"

    for d in (nano_256_dir, final_fp16_dir):
        (d / "baseline").mkdir(parents=True, exist_ok=True)
        (d / "model_a_svd50").mkdir(parents=True, exist_ok=True)
        (d / "model_b_slim50_svd50").mkdir(parents=True, exist_ok=True)

    base_fp16 = ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_fp16.onnx"
    svd_fp16 = ROOT / "jetson_nano" / "models" / "rvm_mobilenetv3_large_svd50_fp16.onnx"
    slim_fp16 = ROOT / "jetson_nano" / "models" / "SLIM50_SVD50_fp16.onnx"

    # Populate jetson_nano_256
    if base_fp16.is_file():
        shutil.copy2(base_fp16, nano_256_dir / "baseline" / "baseline_256_fp16.onnx")
        shutil.copy2(base_fp16, final_fp16_dir / "baseline" / "baseline_fp16.onnx")
    if svd_fp16.is_file():
        shutil.copy2(svd_fp16, nano_256_dir / "model_a_svd50" / "model_a_svd50_256_fp16.onnx")
        shutil.copy2(svd_fp16, final_fp16_dir / "model_a_svd50" / "model_a_svd50_fp16.onnx")
    if slim_fp16.is_file():
        shutil.copy2(slim_fp16, nano_256_dir / "model_b_slim50_svd50" / "model_b_slim50_svd50_256_fp16.onnx")
        shutil.copy2(slim_fp16, final_fp16_dir / "model_b_slim50_svd50" / "model_b_slim50_svd50_fp16.onnx")

    print("\n[+] Populated top2_benchmark model directories:")
    print(f"    {nano_256_dir.relative_to(ROOT)}")
    print(f"    {final_fp16_dir.relative_to(ROOT)}")
    print("\n[V] ALL FP16 CONVERSIONS AND DIRECTORY SETUPS COMPLETE!")

if __name__ == "__main__":
    main()
