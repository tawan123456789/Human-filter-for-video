# RVM-MobileNetV3-Large: Local GPU Compression Search

This directory is a **drop-in addition** to the existing [Human-filter-for-video](https://github.com/tawan123456789/Human-filter-for-video) project. No Midterm files are modified. The pipeline compresses your **fine-tuned original RVM MobileNetV3-Large**, rather than swapping to a smaller backbone. It targets the *smallest actually deployable model file* subject to validation IoU falling by at most **10 percentage points** versus the measured original validation IoU.

> Status: full train search, FP16/ONNX FP32/ONNX INT8 exports, validation, and one final test audit are recorded in `outputs/`. The run used the fixed 2,133 / 266 / 268 train/validation/test split. The results and target-specific trade-offs are summarized below; FPS values are hardware-specific and are not Jetson Nano benchmarks.

## Results and trade-offs (2026-10-21)

The selection gate was **Val IoU ≥ 0.81995**, calculated as the measured C0 validation IoU (0.91995) minus 10 percentage points. All 7 trained variants and all 14 deploy candidates passed this gate. File size is measured from the actual deploy artifact; savings below use the original C0 PyTorch FP32 checkpoint (15,211,668 bytes) as the reference.

### Train search: structural compression beats unstructured pruning for file size

| Candidate | Method | Params | Deploy FP32 | Size saved | Val IoU / Dice | GPU FPS* | Trade-off |
|---|---|---:|---:|---:|---:|---:|---|
| CONTROL | Fine-tune control | 3,747,757 | 15.210 MB | 0.0% | **0.9331 / 0.9625** | 116.9 | Highest search IoU; essentially the same file size as C0, so it is a quality reference rather than a compression win. |
| UP40 | Unstructured prune 40% | 3,747,757 | 15.210 MB | 0.0% | 0.9316 / 0.9618 | 116.6 | Removed 1.49M eligible weights, but dense deploy storage did not shrink. Not worthwhile when the goal is a smaller ordinary checkpoint. |
| UP70 | Unstructured prune 70% | 3,747,757 | 15.210 MB | 0.0% | 0.8999 / 0.9407 | 102.8 | Removed 2.60M eligible weights, yet did not shrink the file and lost about 3.32 IoU points vs CONTROL. Weakest storage/quality trade-off. |
| SVD50 | Low-rank convolution factorization | 3,397,417 | 13.814 MB | 9.2% | **0.9323 / 0.9620** | 106.5 | **Best FP32 accuracy/size balance:** saves 9.2% while losing only 0.08 IoU points vs CONTROL. Choose when keeping quality near the top matters. |
| SLIM75 | Channel slimming 0.75 | 3,638,956 | 14.775 MB | 2.9% | 0.9129 / 0.9512 | 110.6 | Small file reduction for about 2.02 IoU points below CONTROL; SVD50 is a better trade if accuracy is important. |
| SLIM50 | Channel slimming 0.50 | 3,551,287 | 14.423 MB | 5.2% | 0.9085 / 0.9477 | 113.8 | More savings than SLIM75, with a larger quality drop; the hybrid below saves more and scores higher. |
| SLIM50_SVD50 | Slim 0.50 + SVD 0.50 | **3,293,525** | **13.396 MB** | **11.9%** | 0.9116 / 0.9505 | 112.0 | **Smallest trained FP32 model.** Best when structural size reduction matters more than the last ~2.07 IoU points vs SVD50. |

### Export formats: compare within the same runtime

| Model / format | File | Val IoU / Dice | Benchmark on this run | Trade-off |
|---|---:|---:|---|---|
| C0 · PyTorch FP32 | 15.212 MB | 0.9200 / 0.9538 | — | Original checkpoint and size reference. |
| C0 · PyTorch FP16 | 7.663 MB | 0.9200 / 0.9538 | GPU: 97.1 FPS · p50 10.36 / p95 15.05 ms | Nearly preserves baseline validation quality while cutting file size about 49.6%. |
| SLIM50_SVD50 · PyTorch FP16 | 6.758 MB | 0.9116 / 0.9505 | GPU: 103.2 FPS · p50 9.49 / p95 14.75 ms | **Best balanced compact FP16 candidate in this run:** 55.6% smaller than C0 FP32 and about 11.8% smaller than C0 FP16, with a 0.84-point IoU difference vs C0 FP16. |
| C0 · ONNX FP32 | 15.009 MB | 0.9200 / — | CPU: 30.9 FPS · p50 28.07 / p95 58.76 ms | Portable stateful graph; almost no storage reduction vs original C0. |
| SLIM50_SVD50 · ONNX FP32 | 13.197 MB | 0.9116 / — | CPU: **47.5 FPS** · p50 19.66 / p95 33.16 ms | Highest measured CPU throughput among these ONNX exports, with 13.2% less storage than C0 FP32. Best choice to investigate for CPU ONNX deployment. |
| C0 · ONNX INT8 QDQ | 4.075 MB | 0.8856 / 0.9314 | CPU: 24.4 FPS · p50 39.11 / p95 59.98 ms | 73.2% less storage than C0 FP32, but lower IoU and slower than its ONNX FP32 counterpart on the recorded CPU benchmark. |
| SLIM50_SVD50 · ONNX INT8 QDQ | **3.641 MB** | 0.8775 / 0.9263 | CPU: 35.0 FPS · p50 25.59 / p95 51.63 ms | **Smallest valid artifact**; selected by validation-only freeze. Saves 76.1% vs C0 FP32, at a 4.24-point Val IoU difference vs C0. On the recorded CPU benchmark it is slower than the same model's ONNX FP32. |

`—` means the corresponding metric/benchmark was not recorded for that export. PyTorch FP16 timings were measured on the RTX 5070 Ti Laptop GPU; ONNX timings were measured on the local CPU. Compare FPS only within the same device and execution provider; these numbers do **not** predict Jetson Nano speed.

### Which one is most worthwhile?

- **Best accuracy/size trade in trained FP32:** `SVD50` — 13.814 MB and Val IoU 0.9323, only 0.08 points below CONTROL while saving 9.2%.
- **Best compact FP16 starting point:** `SLIM50_SVD50` — 6.758 MB, Val IoU 0.9116 / Dice 0.9505. It gives up a little quality for a much smaller file; the measured RTX speed is not a Nano result.
- **Best when minimum file size is the priority:** `SLIM50_SVD50 ONNX INT8 QDQ` — 3.641 MB. This is the frozen selection for the experiment's validation gate, not a universal fastest model.
- **Best measured CPU ONNX throughput:** `SLIM50_SVD50 ONNX FP32` — 47.5 FPS on the local CPU benchmark; its graph is larger than INT8, but it ran faster in this test.
- **Usually not worth it for file-size reduction:** `UP40` and `UP70`. Unstructured sparsity did not reduce the dense checkpoint file. Use pruning only with a sparse-aware runtime and a measured device benefit.
- **Jetson Nano:** treat `SLIM50_SVD50` FP16 as a candidate to benchmark on-device; do not infer Nano performance from these FPS numbers. Nano is compute capability 5.3, and the TensorRT 8.2.5 matrix lists FP16 but not INT8 for SM 5.3. Build the TensorRT engine on Nano and verify the graph/operators and recurrent-state path there. [Nano compute capability](https://developer.nvidia.com/cuda/gpus/legacy) · [TensorRT 8.2.5 support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-825/pdf/TensorRT-Support-Matrix-Guide.pdf) · [JetPack 4.6.6 / L4T 32.7.6 support](https://developer.nvidia.com/embedded/linux-tegra-r3276).

### Final test audit

The validation-only selection was `SLIM50_SVD50 ONNX INT8 QDQ`. On the held-out 268-image test split it scored **IoU 0.8841 / Dice 0.9341**, versus C0 baseline **IoU 0.9214 / Dice 0.9560**: a **3.72 percentage-point** IoU drop, inside the 10-point limit. The test set was touched once; do not use this test result to select a different model. This run uses a new fixed split and is not directly comparable to the historical 93.54% test IoU elsewhere in the repository. Video temporal quality still requires a separate labeled-video evaluation.

## 0. Windows 11 / PowerShell local setup

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

**Important:** `compression/outputs/split_manifest.json` will be generated once from sorted image stems using seed 42. It is a *new* train/val/test split, not necessarily the same image IDs that produced the README's original 93.54% IoU. To reproduce previous results exactly, supply the original split mapping with `--split-manifest PATH` (JSON keys: `train`,`val`,`test` of image stems). Never change the manifest after running experiments.

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
- Baseline 93.54% cited in the repo is a past **test** estimate. The selection threshold is always `measured_baseline_validation_iou - 0.10`.
- Fixed 256×256 inputs and recurrent downsample_ratio 1.0 are assumed; benchmarks with different resolution or preprocessing cannot be compared directly.
- This project works with upstream RVM under its GPL-3.0 conditions.
