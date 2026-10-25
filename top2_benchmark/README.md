# ONNX FP16 model comparison app

Streamlit app for testing the same three final artifacts side by side:

1. Original RVM baseline (`baseline/baseline_fp16.onnx`)
2. SVD50 + KD (`model_a_svd50/model_a_svd50_fp16.onnx`)
3. SLIM50 + SVD50 + KD (`model_b_slim50_svd50/model_b_slim50_svd50_fp16.onnx`)

The app applies the repository evaluation preprocessing: RGB, resize to 256×256 using bilinear interpolation, values in `[0,1]`, mask resize with nearest-neighbor, and a sigmoid threshold of 0.5. Each image starts with zero recurrent states; repeated latency runs carry recurrent states forward. Latency includes ONNX Runtime input/output handling and is measured on the selected provider. It is not comparable across different providers or devices.

## Run on Windows PowerShell

From the repository root, using the project's virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run .\top2_benchmark\app.py
```

The app uses `CUDAExecutionProvider` when installed and selected; otherwise it uses CPU. The current project environment may only have CPU ONNX Runtime. The validation tab uses the frozen validation IDs from `compression/outputs/final_fp16/midterm_split_manifest.json`. Test images are not evaluated.

## Real-time webcam comparison

To recreate all three 512×512 ONNX FP16 models and their GPU validation/latency measurements:

```powershell
.\.venv\Scripts\python.exe -m compression.export_512 --rvm-source .\third_party\RobustVideoMatting --device cuda --warmup 20 --iters 100
```

Artifacts and metrics are written under `compression/outputs/final_512/`. The export uses the existing best checkpoints and the frozen validation split; it does not retrain the models.

Use the OpenCV app to run one selected model at a time at 512×512 input. Switch between the baseline, SVD50 + KD, and SLIM50 + SVD50 + KD without restarting:

```powershell
python -m top2_benchmark.realtime_app
```

Press `1`, `2`, or `3` (or Left/Right) to switch models, `m` to switch between a colored mask and an overlay, and `q` or `Esc` to quit. Switching resets that model's recurrent state so it resumes from a clean state. The live panel reports smoothed inference latency, model FPS, display FPS, foreground coverage, and mean foreground confidence. IoU/Dice need a ground-truth mask and therefore are not reported for a live camera.

This workstation has an NVIDIA GeForce RTX 5070 Ti and uses ONNX Runtime GPU 1.30.0 with `CUDAExecutionProvider`. To set up a matching environment, remove the CPU package and install the CUDA build:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv\Scripts\python.exe -m pip install -r .\top2_benchmark\requirements-gpu.txt
```

The GPU wheel needs compatible CUDA 13 and cuDNN 9 runtime DLLs. The app imports the CUDA-enabled PyTorch package first so ONNX Runtime can find its bundled DLLs. Confirm availability with `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`. Select a camera with `--camera 1`, or force CPU with `--provider CPUExecutionProvider`.
