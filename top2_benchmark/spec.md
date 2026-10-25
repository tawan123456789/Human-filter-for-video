# TASK: Implement and Benchmark Two Compressed ONNX FP16 Models for the AI for Edge Computing Final Project

Repository: https://github.com/tawan123456789/Human-filter-for-video

You are a Senior Machine Learning Engineer specializing in PyTorch, Model Compression, Knowledge Distillation, ONNX Runtime, CUDA, and Edge AI deployment.

**Your task is to implement, execute, validate, benchmark, and document TWO compressed models that satisfy the course's Final Project criteria.**

This is an implementation task, not a planning-only task. Inspect the existing repository, reuse its implementations, modify the necessary files, execute the available experiments, and produce reproducible artifacts.

Do not stop after writing a plan or example scripts.

---

## 1. Final Project Requirements

The course evaluates:

1. Source and target datasets.
2. Transfer Learning.
3. Model Compression.
4. Model Optimization.
5. Benchmarking against the Midterm models.
6. Summary report.
7. Complete source code and evidence of implementation.

Federated Learning is optional and is not required.

The goal is to demonstrate real structural compression of an existing fine-tuned RVM model, followed by FP16 optimization for Edge AI deployment.

**We must deliver exactly TWO final compressed model variants, plus an original baseline for comparison.**

Both final models MUST use ONNX FP16.

DO NOT use ONNX INT8 for these final experiments.

## 2. Required Models

### Model A: SVD50 + Knowledge Distillation + ONNX FP16

Base architecture:
- Robust Video Matting (RVM).
- MobileNetV3-Large backbone.
- Start from the existing fine-tuned checkpoint, not from random weights.

Compression:
- Low-Rank Approximation.
- SVD with ratio 0.50, following the existing repository's implementation.
- Apply compression only to layers supported by the current operator implementation.
- Preserve architectural and recurrent-state compatibility.
- Recover performance using Knowledge Distillation.

Optimization:
- Export the trained model to ONNX FP32 first.
- Convert the resulting graph to ONNX FP16.
- Use FP32 input/output interfaces if necessary for ONNX Runtime compatibility.
- Execute using ONNX Runtime CUDAExecutionProvider.

Objective:
- Preserve segmentation quality as closely as possible to the original model.
- Achieve real structural parameter reduction.
- Demonstrate the accuracy-oriented compression approach.

### Model B: SLIM50_SVD50 + Knowledge Distillation + ONNX FP16

Base architecture:
- Use the SAME original RVM-MobileNetV3-Large checkpoint as Model A.

Compression:
- Channel Slimming with ratio 0.50.
- SVD Low-Rank Approximation with ratio 0.50.
- Apply both methods in the order defined in the existing configuration.
- Recover performance using Knowledge Distillation.

Optimization:
- Export to ONNX FP32.
- Convert to ONNX FP16.
- Maintain the same external input/output convention as Model A.
- Execute using the same ONNX Runtime CUDAExecutionProvider.

Objective:
- Achieve greater structural compression than Model A.
- Preserve acceptable segmentation quality.
- Demonstrate the compression-oriented approach.

### Original Baseline

Use the existing fine-tuned RVM-MobileNetV3-Large checkpoint:

`outputs/checkpoints/rvm_mobilenetv3_supervisely_best.pth`

Check that the file exists and that the weights are compatible with the model.

Use this checkpoint as the common starting point and teacher for Knowledge Distillation.

Evaluate the original FP32 model and also export an ONNX FP16 baseline for a fair comparison under the same runtime.

The original baseline is a reference, NOT a third final compressed model.

---

## 3. Mandatory Repository Inspection

Before changing code, inspect:

- `README.md`
- `compression/README.md`
- `compression/core.py`
- `compression/operators.py`
- `compression/train.py`
- `compression/quantization.py`
- `compression/search.py`
- `compression/doctor.py`
- `compression/configs/quick.json`
- `compression/configs/full.json`
- Existing tests and result artifacts.
- Relevant Midterm training scripts and dataset splitting code.
- Existing checkpoints and local dataset availability.

Identify and reuse existing implementations.

The repository already includes experiment identifiers:

- `SVD50`
- `SLIM50_SVD50`

These are already configured to use Knowledge Distillation.

Do not unnecessarily rewrite the training framework or change the models' compression definitions.

Preserve existing Midterm code and experimental results.

Place all new final experiment outputs in a clearly separated directory, for example:

`compression/outputs/final_fp16/`

Do not silently overwrite prior checkpoints or historical experiment results.

---

## 4. Dataset and Data Leakage Audit — CRITICAL

Before training or evaluating, audit the dataset split.

The existing Midterm and Compression pipelines may generate different splits despite both using seed 42.

One may use PyTorch's random permutation while another uses NumPy's random permutation.

These procedures do NOT necessarily produce the same samples.

This can cause test contamination if the original teacher checkpoint was trained on images later assigned to the Compression Test set.

Tasks:

1. Inspect how the original fine-tuned teacher checkpoint was trained.
2. Recover the original training, validation, and test image IDs if possible.
3. Compare those IDs with the Compression split manifest.
4. Identify any overlap between the teacher's training data and the proposed final test set.
5. Generate a machine-readable leakage audit.
6. Reuse a verified common split for all final models and the original baseline.
7. Never regenerate splits silently between experiments.
8. Save image IDs and split hashes.

If the original teacher training IDs cannot be established, do not claim that a newly randomized test split is uncontaminated.

Instead, identify a verifiably untouched external test set, or provide another scientifically valid solution such as reconstructing/retraining the baseline using a controlled split.

If a leakage-free evaluation cannot be established, clearly mark final test metrics as NOT VERIFIED.

Never hide this limitation.

Keep all preprocessing identical across models.

Use the existing Supervise.ly person images and corresponding binary segmentation masks where available.

Document:
- Dataset source.
- Number of samples.
- Input resolution.
- Mask preprocessing.
- Train/validation/test allocation.
- Dataset limitations.
- Why this source/target data is relevant to video background segmentation.

Training and model selection must never use test-set results.

---

## 5. Training and Knowledge Distillation

Reuse the repository's existing compression and KD recovery pipeline.

For BOTH models:

1. Load the same fine-tuned teacher checkpoint.
2. Construct the corresponding compressed student.
3. Measure parameters before and after structural compression.
4. Validate successful weight initialization and model forward pass.
5. Train using the existing segmentation loss and KD implementation.
6. Use the same training dataset, validation dataset, resolution, and evaluation protocol.
7. Select the best checkpoint using validation metrics only.
8. Save complete training configuration and history.

Inspect the actual implementation of Knowledge Distillation.

Confirm that:
- The teacher weights are frozen.
- The teacher and student output tensors are compatible.
- The KD loss is correctly calculated.
- Segmentation supervision is also retained.
- The selected KD method and weights are documented.
- The code does not silently skip KD.

Do not claim Knowledge Distillation was applied without verifying the actual training path.

Use the existing hyperparameters as the initial configuration.

Do not change the compression ratios without documenting why.

Use:
- Seed 42.
- CUDA training when available.
- AMP where supported.
- Gradient clipping if appropriate.
- Checkpoint saving and resuming.
- Early stopping based on validation.

Reuse existing completed checkpoints if provenance, configuration, dataset split, and model architecture can be verified.

Otherwise retrain the affected models.

Do not start from random weights as a fallback.

---

## 6. ONNX FP16 Export — REQUIRED

The existing `compression/quantization.py` supports ONNX FP32 and INT8, but the final project requires ONNX FP16.

Extend the existing export pipeline to support an explicit format such as:

`--format onnx-fp16`

Do not replace existing functionality.

Use ONNX-compatible FP16 conversion, such as `onnxconverter-common` or an equivalent verified method.

Required export sequence:

PyTorch FP32 compressed model
→ Stateful ONNX FP32
→ ONNX FP16 conversion
→ ONNX graph validation
→ ONNX Runtime execution
→ Numerical parity verification
→ Validation benchmark

Preserve the recurrent-state interface:

Inputs:
- image
- r1
- r2
- r3
- r4

Outputs:
- logits
- r1o
- r2o
- r3o
- r4o

Requirements:

1. Preserve all four recurrent states.
2. Do not convert the RVM into an unrelated stateless segmentation network.
3. Verify input and output dimensions.
4. Verify initial zero-state and propagated-state inference.
5. Test multiple consecutive frames.
6. Check numerical parity against the corresponding PyTorch model.
7. Measure IoU and Dice after ONNX FP16 conversion.
8. Validate numerical stability and finite outputs.
9. Check that the actual ONNX graph contains meaningful FP16 computation/weights.
10. Save the final ONNX file and export metadata.

Keeping external I/O tensors in FP32 is acceptable if necessary for compatibility.

However, document any FP32 operators remaining in the graph.

Do not label an entirely FP32 model as FP16.

If a particular ONNX operator cannot safely run in FP16, preserve that operator in FP32 and record the mixed-precision exception.

Do not silently fall back to CPUExecutionProvider when reporting CUDA performance.

Report the actual active execution providers and provider assignments where available.

If CUDAExecutionProvider is unavailable, report an execution blocker rather than inventing benchmark results.

---

## 7. Quality Acceptance Criteria

The primary constraint is:

**Maximum allowed IoU decrease = 10 percentage points from the verified original baseline.**

Formula:

`IoU_drop_pp = (IoU_baseline - IoU_compressed) * 100`

Pass condition:

`IoU_compressed >= IoU_baseline - 0.10`

Use a fresh baseline measurement on the same verified validation split.

For reference, the previous experiment reported:

- Baseline Validation IoU approximately 0.91995.
- Previous threshold approximately 0.81995.

These values are historical references, not hardcoded thresholds.

Recalculate if the verified dataset split changes.

Both final compressed models must:

- Pass the validation IoU constraint.
- Have fewer structural parameters than the original model.
- Produce a valid executable ONNX FP16 artifact.
- Have reproducible training and export metadata.
- Pass stateful inference and numerical integrity checks.

If a model fails, report why.

Do not mark a failed run as successful.

Do not use test-set scores to tune the model.

After the two model configurations are frozen and pass validation, perform final testing on a verified untouched test set.

---

## 8. Benchmarking Requirements

Benchmark the following three variants:

A. Original RVM-MobileNetV3-Large ONNX FP16 — fair runtime baseline.

B. SVD50 + KD ONNX FP16.

C. SLIM50_SVD50 + KD ONNX FP16.

Also retain the original PyTorch FP32 Midterm reference as a separate historical comparison.

Use:
- Exactly the same input resolution: 256 × 256.
- Same GPU.
- Same ONNX Runtime and execution provider.
- Same batch size of 1.
- Same warm-up procedure.
- Same measured iteration count.
- Same recurrent-state execution procedure.
- Same preprocessing and evaluation metrics.

Use at least 20 warm-up iterations and 100 measured iterations as a starting point, configurable from the CLI.

Use appropriate synchronization for accurate GPU latency measurements.

Benchmark actual stateful inference with recurrent states carried forward across frames.

Also report whether the timing excludes or includes CPU-to-GPU transfers.

Required metrics:

### Accuracy
- Validation IoU.
- Test IoU, only if verified leakage-free.
- Validation Dice.
- Test Dice, only if verified leakage-free.
- Precision.
- Recall.
- IoU decrease in percentage points.

### Model complexity
- Number of parameters.
- Structural parameter reduction (%).
- FP32 checkpoint size (MB).
- ONNX FP16 artifact size (MB).
- ONNX FP16 size reduction (%).
- FLOPs or MACs if they can be measured reliably.

### Inference performance
- Mean latency (ms).
- p50 latency (ms).
- p95 latency (ms).
- FPS.
- Peak GPU memory (MB).
- Device and runtime information.

### Numerical integrity
- PyTorch vs ONNX maximum absolute error.
- Mean absolute error.
- Recurrent-state parity checks.
- FP16 operator/initializer verification.

Calculate file sizes from actual artifacts.

Do not estimate real file sizes by multiplying parameters by two bytes.

Avoid comparing GPU FPS against CPU FPS as though they were equivalent.

Clearly separate measured results from historical results.

---

## 9. Required Results and Visualizations

Generate:

1. Final comparison table for all three reference/compressed variants.
2. Model size comparison chart.
3. IoU comparison chart.
4. FPS comparison chart.
5. Accuracy vs model size scatter plot.
6. Latency comparison chart.
7. Before/after segmentation examples using identical images.
8. Training and validation loss curves.
9. Summary of structural compression and FP16 optimization.

Use validation metrics for development comparisons.

Only use test metrics in the final locked evaluation.

All figures must originate from actual result files.

Never fabricate results or fill missing metrics with guessed values.

---

## 10. Required Output Files

Create a clearly organized final directory.

Suggested structure:

compression/outputs/final_fp16/
- audit/
  - dataset_split_audit.json
  - checkpoint_provenance.json
- baseline/
  - baseline_fp16.onnx
  - metrics.json
- model_a_svd50/
  - best.pt
  - model_a_fp16.onnx
  - train_config.json
  - training_history.csv
  - export_validation.json
  - benchmark.json
- model_b_slim50_svd50/
  - best.pt
  - model_b_fp16.onnx
  - train_config.json
  - training_history.csv
  - export_validation.json
  - benchmark.json
- comparison/
  - results.csv
  - results.json
  - figures/
  - qualitative_samples/
- FINAL_REPORT.md
- REPRODUCE.md

Adapt this structure to repository conventions where necessary.

Do not duplicate huge checkpoint files unnecessarily.

If the GitHub repository excludes large binaries, retain local artifact paths and provide appropriate release instructions.

---

## 11. Summary Report

Generate a detailed `FINAL_REPORT.md` suitable for the course final submission.

Write explanatory sections in Thai, keeping technical terms in English.

Required sections:

1. Project Background and Objectives.
2. Source and Target Dataset.
3. Original RVM Architecture.
4. Transfer Learning and Initial Checkpoint.
5. Model A: SVD50 + KD.
6. Model B: Channel Slimming + SVD50 + KD.
7. Model Optimization with ONNX FP16.
8. Training Settings.
9. Benchmark Methodology.
10. Experimental Results.
11. Accuracy–Model Size–Latency Trade-off.
12. Comparison with Midterm Models.
13. Edge Deployment Considerations.
14. Limitations and Threats to Validity.
15. Conclusion.
16. Reproduction Instructions.

Explain the role of every compression operation and why the two models are being compared.

Make it clear that both final models are derived from an existing foundation/pretrained architecture and are not merely downloaded ready-made models.

Explain the original fine-tuning and compression recovery process.

Do not confuse binary segmentation IoU with video temporal consistency or alpha-matting performance.

If video-level annotated evaluation is unavailable, state this limitation explicitly.

---

## 12. Tests and Reproducibility

Add or extend automated tests for:

- Dataset split overlap.
- Model loading.
- Structural parameter reduction.
- KD loss execution.
- ONNX graph validation.
- FP16 graph inspection.
- ONNX Runtime inference.
- Recurrent-state dimensions and propagation.
- Numerical parity.
- Metric consistency.
- Compression acceptance threshold.

Provide PowerShell commands compatible with Windows 11.

Support incremental execution:

- Environment check.
- Dataset audit.
- Model A training/recovery.
- Model B training/recovery.
- ONNX FP16 export.
- Model validation.
- Benchmark.
- Final report generation.

Avoid reinstalling or replacing a working CUDA-enabled PyTorch build unnecessarily.

Use the existing environment when compatible.

Make all runs resumable where appropriate.

Save environment versions, GPU model, CUDA details, ONNX Runtime version, source commit, checkpoint hashes, and dataset split hashes.

---

## 13. Execution Priority

Follow this order:

**Phase 1 — Inspect and Audit**
- Review existing implementations.
- Verify checkpoint, dataset, and CUDA environment.
- Investigate possible dataset leakage.
- Determine which existing model artifacts can be reused.

**Phase 2 — Implement**
- Add ONNX FP16 support.
- Implement parity and recurrent-state validation.
- Implement unified benchmarking.
- Add necessary tests.

**Phase 3 — Execute**
- Reuse verified checkpoints or train/recover both students.
- Export both models to ONNX FP16.
- Execute validation and runtime benchmarks.
- Record actual results.

**Phase 4 — Finalize**
- Freeze the two final models.
- Run final test evaluation only if the holdout is verified.
- Generate comparison files, figures, and report.
- Verify that both models meet all acceptance criteria.

Do not skip important checks merely to complete the task faster.

---

## 14. Definition of Done

The task is complete ONLY when:

- Both final models have genuinely undergone structural compression.
- Both models use Knowledge Distillation recovery.
- Both final executable artifacts are ONNX FP16.
- Both models have a verified Validation IoU drop no greater than 10 percentage points.
- Both models have fewer structural parameters than the original baseline.
- Both models pass graph, numerical, and stateful inference tests.
- Both models have hardware-consistent benchmarks.
- Dataset leakage has been investigated and documented.
- Source/target data and Transfer Learning are explained.
- Summary report and source code are complete.
- Results are reproducible.

If any condition cannot be fulfilled in the available environment, identify the exact blocker and leave the status as incomplete. Never claim a model was trained, exported, validated, or benchmarked unless that action actually succeeded.

## 15. Final Response Format

When finished, respond in Thai with:

1. Files created or modified.
2. What was reused from the original repository.
3. Model A implementation and experimental results.
4. Model B implementation and experimental results.
5. Benchmark comparison against the original baseline.
6. Pass/fail status for each Final Project requirement.
7. Dataset leakage audit conclusion.
8. Commands to reproduce the complete pipeline.
9. Remaining blockers, if any.

**Start by inspecting the repository, then implement the required changes and execute the workflow. Do not merely provide a conceptual proposal.**