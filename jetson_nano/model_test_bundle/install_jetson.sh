#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.8}"
WHEEL="wheels/onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.8 was not found. Set PYTHON_BIN to a Python 3.8 executable." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'

"$PYTHON_BIN" -m pip install --upgrade 'pip<25' 'setuptools<76' wheel
"$PYTHON_BIN" -m pip install 'numpy==1.24.4' 'protobuf==4.25.5' flatbuffers
"$PYTHON_BIN" -m pip install "$WHEEL"
"$PYTHON_BIN" -m pip install -r requirements-app.txt

"$PYTHON_BIN" - <<'PY'
import onnxruntime as ort
print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())
PY
echo "Install complete. See README_TH.md for run commands."
