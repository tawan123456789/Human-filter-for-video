# RVM-MobileNetV3-Large: Local GPU Compression Search

This directory is a **drop-in addition** to the existing [Human-filter-for-video](https://github.com/tawan123456789/Human-filter-for-video) project. No Midterm files are modified. The pipeline compresses your **fine-tuned original RVM MobileNetV3-Large**, rather than swapping to a smaller backbone. It targets the *smallest actually deployable model file* subject to validation IoU falling by at most **10 percentage points** versus the measured original validation IoU.

> Status: experiment artifacts are retained under outputs/. The consolidated report is in EXPERIMENT_RESULTS.md. Historical test results from the old compression split are not verified because the split overlaps the Midterm training IDs; no Jetson Nano benchmark is recorded.

## Results and trade-offs

All recorded model families, compression candidates, precision exports, benchmark values, and audit limitations are consolidated in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md). Use that report as the canonical results record; this README keeps the workflow and reproduction instructions.

## 0. Windows 11 / PowerShell local setup

The latest recovery run uses train-only Gaussian blur/noise; validation and test preprocessing stays deterministic and unmodified. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for augmentation settings and the runs that used them.

```powershell
cd Human-filter-for-video
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Use a CUDA-enabled PyTorch+torchvision build compatible with your GPU.
# Install/verify your working torch first, then:
python -m pip install numpy pillow tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Obtain the upstream source once for offline use (RVM uses an upstream GPL-3.0 implementation):

```powershell
git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
```

**Don't** use `pip install -r requirements-quant.txt` to replace your working GPU PyTorch installation. The ONNX packages are optional and CPU-only.

### Required local paths

```text
Human-filter-for-video/
  compression/                                  <-- add this directory
  data/supervisely/images/                     <-- images from prior project
  data/supervisely/masks/                      <-- annotated masks
  outputs/checkpoints/rvm_mobilenetv3_supervisely_best.pth
  third_party/RobustVideoMatting/              <-- upstream source
```

Checkpoint file should be the **fine-tuned V3-Large state**, optionally wrapped in `model_state_dict`. The loader refuses missing core weights. If using different paths, override `--checkpoint`, `--data-dir`, `--images-dir`, `--masks-dir`, `--rvm-source` on all relevant commands.

## 1. Check everything before using GPU hours

```powershell
python -m compression.doctor --rvm-source third_party/RobustVideoMatting --device cuda
```

Checks CUDA, original checkpoint, image/mask pairs, split manifest, segmentation-only output parity including 4 recurrent states.

**Important:** `compression/outputs/split_manifest.json` is a fixed train/validation/test split for the compression search and differs from the reconstructed Midterm split. Their results are not directly comparable; see the consolidated report for the split audit and leakage caveat. Never change a manifest after running experiments.

## 2. Launch fast experiment set (resumable)

```powershell
python -m compression.search --profile compression/configs/quick.json --rvm-source third_party/RobustVideoMatting --device cuda --epochs 8 --batch-size 8
```

`quick.json`: CONTROL, UP40, UP70, SVD50, SLIM75, SLIM50, SLIM50_SVD50. For all candidates, CPU/main GPU benchmarking is **stateful** (carries four recurrent tensors frame-to-frame). CUDA AMP training and Knowledge Distillation recovery are enabled by default when spec has `kd:true`.

For more aggressive search:

```powershell
python -m compression.search --profile compression/configs/full.json --rvm-source third_party/RobustVideoMatting --device cuda --epochs 8
```

Or run/restart one candidate:

```powershell
python -m compression.train --experiment SLIM50 --rvm-source third_party/RobustVideoMatting --epochs 8
# To intentionally overwrite prior run data, append --force
```

The training code automatically saves `last.pt`, `best.pt`, `history.csv`, `config.json`, `deploy_fp32_state.pth`, `result.json`. If interrupted, repeat the identical command to resume. **Changing any optimizer, dataset or run settings invalidates the saved run.** To save RAM while debugging: `--batch-size 2 --workers 0 --skip-benchmark` and use a separate `--output-dir`.

## 3. Quantize best models

```powershell
python -m pip install -r compression/requirements-quant.txt
python -m compression.quantization --run-id C0 --format both --rvm-source third_party/RobustVideoMatting
python -m compression.quantization --run-id SLIM50 --format both --rvm-source third_party/RobustVideoMatting
```

- `fp16` is a **GPU half-precision state_dict**, not a ready CPU INT8 executable.
- `int8` exports a **stateful** ONNX model with `image,r1,r2,r3,r4` inputs and `logits,r1o,r2o,r3o,r4o` outputs. Compares PyTorch and ONNX outputs over 2 recurrent frames and uses **train-only** still images with zero/propagated state to calibrate Conv QDQ operators. Validates IoU on validation split only; benchmarks INT8 vs ONNX FP32 on the same CPU execution provider.
- INT8 may fail to export/execute for certain PyTorch/ONNX/ORT versions; the command will raise an error, **not** claim compression success.
- INT8/FP16 footprint is measured from actual exported files, never estimated from bit width.

## 4. Evaluate and freeze the final result

```powershell
python -m compression.evaluate --run-id SLIM50 --rvm-source third_party/RobustVideoMatting
python -m compression.finalize --confirm-final --rvm-source third_party/RobustVideoMatting
```

`finalize` chooses the **smallest valid file using validation data only**, freezes the choice in `reports/frozen_selection.json`, then evaluates test only once and writes `reports/final_test.json`. It refuses to run again if that file already exists. The test metric is an audit, *not a model-selection signal*. The 10-point rule uses **the measured baseline on the same partition**. Video temporal quality still needs separate labeled video testing.

## 5. Experiment IDs and interpretation

| ID prefix | Technique | Number of params decreases? | Dense model storage decreases? |
|---|---|---|---|
| `CONTROL` | Further fine-tuning without compression | No | No |
| `UP` | Global unstructured magnitude pruning with persistent masks | No | Usually no |
| `SLIM` | Reduced decoder widths, including ConvGRU state channels | **Yes** | **Yes** |
| `SVD` | Dense convolution low-rank factorization | **Yes** | **Yes** |
| `SLIM*_SVD*` | Hybrid structural compression | **Yes** | **Yes** |
| `INT8` | Static quantization of stateful ONNX Conv operators | N/A | Measure actual file |

Unstructured pruning reports **eligible-weight sparsity**; it does not claim speed or file shrinkage unless observed. Structural slimming is overlap-initialized and recovers through fine-tuning and KD. Weights-only INT4, dependency-graph backbone channel pruning, and QAT are **not implemented**: do not present them as completed experiments.

## 6. Recorded outputs

```text
compression/outputs/
├── split_manifest.json
├── baseline.json
├── search_results.csv
├── runs/<RUN_ID>/
│   ├── config.json
│   ├── last.pt
│   ├── best.pt
│   ├── history.csv
│   ├── deploy_fp32_state.pth
│   └── result.json
├── exports/C0_segment_fp32_state.pth
├── exports/<RUN_ID>/
│   ├── state_fp16.pth
│   ├── stateful_fp32.onnx
│   ├── stateful_int8.onnx
│   ├── onnx_fp32.json
│   ├── fp16.json
│   └── int8.json
├── reports/frozen_selection.json
├── reports/final_test.json
└── logs/<FAILED_RUN>_error.txt
```

Add `compression/outputs/` to `.gitignore` in the **original** repository, or ensure large outputs/checkpoints are excluded before committing. Original app/runtime does not automatically load a slimmed or SVD-modified architecture: deploy using its `spec` and reconstruct transformations as in `load_variant` in `quantization.py`, or deploy a validated ONNX graph with its explicit recurrent I/O.

## Dataset and metric cautions

- Ground Truth must be human-labeled binary masks. The webcam demo's model-generated reference mask is not Ground Truth.
- Evaluation measures **mean per-image IoU** (empty mask pairs count as 1 for exact empty prediction). Also stores Dice/global IoU where available.
- Historical metrics from different model runs are listed in the canonical report; the selection threshold is always `measured_baseline_validation_iou - 0.10`.
- Fixed 256×256 inputs and recurrent downsample_ratio 1.0 are assumed; benchmarks with different resolution or preprocessing cannot be compared directly.
- This project works with upstream RVM under its GPL-3.0 conditions.
