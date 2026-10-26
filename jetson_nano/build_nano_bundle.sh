#!/usr/bin/env bash
set -euo pipefail

# Build on the target Jetson (or matching Linux ARM64 system). PyInstaller does
# not cross-compile Windows/x86 binaries into Linux/aarch64 executables.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-$REPO_DIR/jetson_nano/models/rvm_mobilenetv1.onnx}"
if [[ ! -f "$MODEL" ]]; then
  MODEL="$REPO_DIR/jetson_nano/models/default_model.onnx"
fi
if [[ ! -f "$MODEL" ]]; then
  MODEL="$REPO_DIR/compression/outputs/jetson_nano/SLIM50_SVD50.onnx"
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Build this package on a Linux ARM64 / Jetson Nano device (found $(uname -m))." >&2
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)'; then
  echo "PyInstaller requires Python 3.8 or newer. Current JetPack images may use Python 3.6." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL (Run python export_onnx.py first)" >&2
  exit 1
fi


python3 -m pip install --user pyinstaller
python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name nano1_webcam_sender \
  --distpath "$REPO_DIR/dist" \
  --workpath "$REPO_DIR/build/pyinstaller" \
  --add-data "$MODEL:model" \
  --collect-all onnxruntime \
  --collect-all cv2 \
  --collect-all numpy \
  "$SCRIPT_DIR/webcam_sender.py"

echo
echo "Bundle created at: $REPO_DIR/dist/nano1_webcam_sender"
echo "Copy the whole directory to Nano 1; it contains the ONNX model."
