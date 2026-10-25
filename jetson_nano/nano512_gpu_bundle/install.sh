#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.8 for ARM64 is required (set PYTHON_BIN to its path)." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'
"$PYTHON_BIN" -m venv "$ROOT/.venv"
# Install all bundled Python wheels offline; ORT wheel is the CUDA 10.2 ARM64 build.
"$ROOT/.venv/bin/python" -m pip install --no-index --find-links "$ROOT/wheels" \
  numpy==1.24.4 opencv-python==4.10.0.84 protobuf==4.25.5 flatbuffers==23.5.26
"$ROOT/.venv/bin/python" -m pip install --no-index --no-deps \
  "$ROOT/wheels/onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl"
"$ROOT/.venv/bin/python" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("ONNX Runtime", ort.__version__, "providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is missing; check JetPack/CUDA/cuDNN compatibility")
PY
echo "Installed. Run from bundle root with .venv/bin/python -m top2_benchmark.realtime_app --provider CUDAExecutionProvider"
