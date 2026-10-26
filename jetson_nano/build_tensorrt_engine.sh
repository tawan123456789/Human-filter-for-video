#!/usr/bin/env bash
# ==============================================================================
# Build TensorRT Engine on Jetson Nano using trtexec
# Compiles ONNX FP16 into a native TensorRT (.engine) for 25-35+ FPS
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${DIR}/models"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

if [ ! -f "${TRTEXEC}" ]; then
  echo "[-] trtexec not found at ${TRTEXEC}."
  echo "    Ensure JetPack (TensorRT) is installed on your Jetson Nano."
  exit 1
fi

MODEL_ONNX="${1:-${MODEL_DIR}/rvm_mobilenetv3_large_svd50_fp16.onnx}"
if [ ! -f "${MODEL_ONNX}" ]; then
  echo "[-] Model ONNX file not found: ${MODEL_ONNX}"
  echo "    Usage: bash build_tensorrt_engine.sh [path/to/model.onnx]"
  exit 1
fi

ENGINE_OUT="${MODEL_ONNX%.onnx}.engine"

echo "=========================================================="
echo " Building TensorRT Engine"
echo " Input ONNX : ${MODEL_ONNX}"
echo " Output TRT : ${ENGINE_OUT}"
echo "=========================================================="

"${TRTEXEC}" \
  --onnx="${MODEL_ONNX}" \
  --saveEngine="${ENGINE_OUT}" \
  --fp16 \
  --workspace=512 \
  --verbose

echo ""
echo "[V] TensorRT engine created successfully at:"
echo "    ${ENGINE_OUT}"
