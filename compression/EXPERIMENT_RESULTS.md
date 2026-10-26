# สรุปผลการทดลอง Human Filter

**ปรับปรุงล่าสุด:** 27 ตุลาคม 2026
รวบรวมผลจากรอบ FP16 256×256, รอบเพิ่ม Gaussian blur/noise และรอบ ONNX FP16 512×512 ไว้ที่เดียว

## สรุป

เปรียบเทียบ RVM-MobileNetV3-Large baseline กับ Model A (SVD50 + Knowledge Distillation) และ Model B (SLIM50 + SVD50 + KD). รอบล่าสุด fine-tune ใหม่ด้วย Gaussian blur/noise เฉพาะภาพ train. PyTorch FP16 ทั้งคู่ผ่าน validation IoU gate และลดพารามิเตอร์เชิงโครงสร้างจริง แต่ ONNX FP16 ทั้งคู่ไม่ผ่าน stateful parity gate (max absolute error ≤ 0.08) จึงยังไม่รับรองสำหรับ deployment. ยังไม่มีผลทดสอบบน Jetson Nano และไม่รายงาน test metrics.

## Dataset และเกณฑ์ประเมิน

ใช้ Supervisely person images/masks 2,667 คู่ ขนาด 256×256; split train/validation/test = 2,133/266/268 จาก Midterm notebook. Split manifest SHA-256: 39ab7de07ac5a9d5341163b8fd95c256071deeb89130a9303c41599a51f60749. ภาพ resize bilinear; mask resize nearest-neighbor และ binarize.

Split compression รุ่นเก่าใช้ NumPy permutation ต่างจาก Midterm และมี Midterm test IDs 212 ภาพอยู่ใน train split เก่า จึงไม่นำผลนั้นมาเป็น final evidence. Teacher checkpoint ไม่มี image IDs ฝังอยู่; provenance อ้างอิง source notebook. Test status เป็น **NOT VERIFIED**. Baseline validation IoU 0.9282666; gate คือ 0.8282666.

## รอบล่าสุด: Gaussian blur/noise (26 ต.ค. 2026)

Augmentation online หลัง resize เฉพาะ train: blur p=0.25, radius 0.3–1.2; noise p=0.35, std 0.01–0.05 บนช่วง [0,1]. Mask คงเดิม; validation/test ไม่ augment.

| รุ่น | Best epoch | FP32 Val IoU / Dice | PyTorch FP16 Val IoU / Dice | Params | ลด params | FP16 size | GPU p50 / p95* | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | — | 0.92827 / 0.96075 | — | 3,747,757 | — | — | — | — |
| **A — SVD50 + KD** | 18/23 | 0.94214 / 0.96921 | **0.94212 / 0.96920** | 3,397,417 | **9.35%** | 6,967,371 bytes | 7.46 / 10.65 ms | 125.3 |
| **B — SLIM50 + SVD50 + KD** | 47/50 | 0.93218 / 0.96350 | **0.93220 / 0.96351** | 3,293,525 | **12.12%** | 6,757,779 bytes | 7.11 / 10.26 ms | 131.0 |

ทั้ง A/B ผ่าน validation gate. A ได้ IoU สูงสุด; B เล็กที่สุด. *Timing เป็น PyTorch FP16 บน RTX 5070 Ti Laptop GPU, 50 iterations; ไม่ใช่ ONNX หรือ Jetson benchmark.

### ONNX FP16 รอบล่าสุด

สร้าง stateful graph สำเร็จ, public I/O เป็น FP32 และพบ FP16 initializers (A 170, B 166). แต่ export validation ระบุว่า parity เกิน 0.08 ทั้งคู่ (valid=false). ห้ามใช้ PyTorch FP16 validation metrics แทน ONNX metrics.

| รุ่น | ONNX FP16 ขนาดจริง | Parity ≤ 0.08 | สถานะ |
|---|---:|---|---|
| A — SVD50 + KD | 6,870,333 bytes | ไม่ผ่าน | ยังไม่รับรอง |
| B — SLIM50 + SVD50 + KD | 6,660,779 bytes | ไม่ผ่าน | ยังไม่รับรอง |

Metadata ระบุว่าเกิน gate แต่ไม่มีตัวเลข max error จริง จึงไม่ใส่ค่าประมาณ.

## รอบ FP16 เดิม 256×256 (ก่อน augmentation)

เป็น CPU ONNX smoke/parity; environment ไม่มี CUDA provider จึงไม่มี ONNX GPU benchmark. Parity วัด 5 เฟรมพร้อมส่ง recurrent states; รอบนี้ไม่ได้กำหนด threshold.

| รุ่น | PyTorch Val IoU / Dice | CPU ONNX Val IoU / Dice | Params | ONNX FP16 size | CPU parity max / mean abs |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.92827 / 0.96075 | 0.92824 / 0.96073 | 3,747,757 | 7,563,834 bytes | 0.11783 / 0.00311 |
| A — SVD50 + KD | 0.94253 / 0.96937 | 0.94258 / 0.96939 | 3,397,417 | 6,870,333 bytes | 0.39761 / 0.00329 |
| B — SLIM50 + SVD50 + KD | 0.93268 / 0.96379 | 0.93266 / 0.96377 | 3,293,525 | 6,660,779 bytes | 0.08280 / 0.00160 |

## รอบ ONNX FP16 512×512

Export checkpoint เดิม ไม่ได้ฝึกใหม่. ONNX Runtime CUDAExecutionProvider บน RTX 5070 Ti; validation 266 ภาพ, batch 1, warm-up 20, วัด 100 calls, ส่ง recurrent states ต่อเนื่อง. Latency วัดเฉพาะ InferenceSession.run ไม่รวม capture/preprocess/render.

| รุ่น | Val IoU / Dice | ONNX FP16 size | Mean ms | p50 / p95 ms | FPS | 5-frame parity max / mean abs |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.95102 / 0.97427 | 7,563,908 bytes | 13.88 | 13.41 / 17.89 | 72.07 | 0.15957 / 0.00236 |
| A — SVD50 + KD | **0.95779 / 0.97795** | 6,870,413 bytes | 17.29 | 16.61 / 24.00 | 57.84 | 0.66516 / 0.00591 |
| B — SLIM50 + SVD50 + KD | 0.95299 / 0.97537 | **6,660,857 bytes** | 13.91 | **13.00 / 18.89** | 71.87 | 0.11301 / 0.00204 |

รอบ 512 ไม่ได้กำหนด parity threshold; หากเทียบกับ gate 0.08 ที่ใช้ภายหลัง ทั้งสามกราฟเกิน gate. ตารางนี้ยึด machine-readable comparison_512.json เพราะเอกสารเก่ามี timing อีกชุดที่ไม่ตรงกัน. Error สูงสุดอยู่ที่ logits.

## สินค้าทดลองทั้งหมด: backbone และ fine-tuning

งาน Midterm ทดลอง RVM กับ MobileNetV3-Large, MobileNetV3-Small, MobileNetV2 และ MobileNetV1. ตารางแรกเป็นค่าจาก backbone_comparison.csv ซึ่งรายงาน test metrics, พารามิเตอร์, FLOPs และความเร็วใน benchmark เดียวกัน:

| Backbone | Params (M) | FP32 size (MB) | GFLOPs | IoU | Dice | Test BCE | GPU FPS | CPU FPS | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RVM-MobileNetV3-Large | 3.749 | 14.995 | 3.992 | 0.93322 | 0.96546 | 0.06038 | 53.17 | 18.56 | 20 |
| RVM-MobileNetV3-Small | 1.590 | 6.360 | 3.274 | 0.91194 | 0.95394 | 0.08360 | 52.38 | 20.50 | 12 |
| RVM-MobileNetV2 | 2.418 | 9.672 | 4.209 | 0.92794 | 0.96262 | 0.06567 | 71.64 | 14.57 | 8 |
| RVM-MobileNetV1 | 4.206 | 16.824 | 6.247 | 0.93345 | 0.96558 | 0.07328 | 136.76 | 11.80 | 17 |

ผลเทียบ pretrained กับ fine-tuned ที่บันทึกแยกตามโมเดล:

| Backbone | Pretrained IoU / Dice | Fine-tuned IoU / Dice | Fine-tuned BCE |
|---|---:|---:|---:|
| MobileNetV3-Large | 0.92875 / 0.96306 | 0.93540 / 0.96662 | 0.05765 |
| MobileNetV3-Small | 0.33221 / 0.49873 | 0.91275 / 0.95438 | 0.08279 |
| MobileNetV2 | ≈0.00000 / ≈0.00000 | 0.92687 / 0.96204 | 0.06843 |
| MobileNetV1 | 0.00000 / 0.00000 | 0.93180 / 0.96470 | 0.07897 |

ค่าจาก backbone comparison กับไฟล์ baseline_vs_finetuned ไม่ตรงกันเล็กน้อยสำหรับบาง backbone (เช่น V1/V2/Small/Large). จึงเก็บค่าตามแหล่งของแต่ละตาราง ไม่เฉลี่ยหรือเลือกค่าที่ดูดีกว่า. ตัวเลขชุด Midterm เป็นผลที่รายงานในอดีต; split/checkpoint provenance ไม่ได้ cryptographically tied กับ checkpoint และไม่ควรเทียบตรงกับ split audit รุ่นใหม่.

README เดิมยังมี benchmark snapshot อีกชุดจาก RTX 5060 Laptop GPU / Intel Core, streaming 256×256. เก็บไว้แยกเพราะไม่ตรงกับ CSV benchmark ด้านบน:

| Backbone ตาม README snapshot | Params (M) | GFLOPs | Test BCE | Test IoU / Dice | Precision / Recall | GPU / CPU FPS |
|---|---:|---:|---:|---:|---:|---:|
| V3-Large | 3.74 | 5.40 | 0.0577 | 0.9354 / 0.9666 | 0.9728 / 0.9606 | ~110 / ~18 |
| V1 | 4.21 | 6.25 | 0.0791 | 0.9316 / 0.9646 | 0.9645 / 0.9647 | 185.6 / 20.6 |
| V2 | 2.42 | 4.21 | 0.0686 | 0.9267 / 0.9619 | 0.9643 / 0.9596 | 134.7 / 22.8 |
| V3-Small | 1.59 | 3.27 | 0.0830 | 0.9125 / 0.9543 | 0.9548 / 0.9537 | 126.8 / 33.9 |

ตาราง snapshot เป็น historical report ไม่ใช่ผล benchmark รอบ compression/Jetson. GFLOPs และ FPS แตกต่างจาก backbone_comparison.csv; คงแยก source ไว้และไม่ผสมเป็นค่าเดียว.

## สินค้าทดลองทั้งหมด: compression search รุ่นแรก

มีการฝึก CONTROL และ candidate เจ็ดแบบด้วย teacher checkpoint เดียวกัน. ทุก run ด้านล่างใช้ compression split SHA-256 5e2ae755cdca23268855bb9958312cbec863c6c669ddd5909c89f56d960c7747, ไม่ใช่ split Midterm ที่ใช้สำหรับ run final. Validation gate ของรอบนั้นคือ IoU ≥0.81995. ค่า params และ checkpoint size เป็นค่าจาก artifact จริง.

| Run / วิธี | Val IoU / Dice | Params | Checkpoint bytes | p50 / p95 ms | FPS | หมายเหตุ |
|---|---:|---:|---:|---:|---:|---|
| CONTROL — ไม่บีบอัด, ไม่ใช้ KD | 0.93309 / 0.96251 | 3,747,757 | 15,210,192 | 8.72 / 10.41 | 116.92 | reference ของ search split |
| SVD50 + KD | 0.93228 / 0.96198 | 3,397,417 | 13,813,899 | 9.53 / 11.35 | 106.51 | ลด params 9.35% |
| SLIM50 + KD | 0.90855 / 0.94769 | 3,551,287 | 14,423,184 | 9.34 / 10.47 | 113.80 | ลด params 5.24% |
| SLIM75 + KD | 0.91287 / 0.95117 | 3,638,956 | 14,774,864 | 9.38 / 10.84 | 110.56 | ลด params 2.90% |
| SLIM50 + SVD50 + KD | 0.91157 / 0.95053 | 3,293,525 | 13,395,767 | 9.06 / 10.85 | 111.97 | ลด params 12.12% |
| UP40 — unstructured pruning 40% + KD | 0.93164 / 0.96178 | 3,747,757 | 15,210,192 | 8.36 / 10.67 | 116.57 | sparsity 40%; params/ขนาด dense ไม่ลด |
| UP70 — unstructured pruning 70% + KD | 0.89993 / 0.94071 | 3,747,757 | 15,210,192 | 9.30 / 14.48 | 102.77 | sparsity 70%; params/ขนาด dense ไม่ลด |

CONTROL กับ candidate ผ่าน validation gate ของ search ตาม metadata. ค่าเหล่านี้เป็น exploratory search บน split เก่า ไม่ใช่ผลที่ใช้เลือก final A/B รอบ Midterm split. ค่า IoU ก่อน recovery ของบาง run ต่ำมาก แต่รายงานนี้แสดง best validation checkpoint หลัง recovery.

## สินค้าทดลอง precision และ quantization

การ export เพิ่มเติมใน compression split รุ่นแรก:

| Model / format | Val IoU / Dice | Artifact bytes | Parity / latency | ผล |
|---|---:|---:|---|---|
| C0 baseline ONNX FP32 | 0.91997 / — | 15,009,497 | max abs parity 0.0000267; CPU p50 28.07 ms | export ใช้งานได้ |
| C0 baseline PyTorch FP16 | 0.92001 / 0.95380 | 7,663,033 | GPU p50 10.36 ms; 97.07 FPS | ผ่าน validation gate รอบเก่า |
| C0 baseline ONNX INT8 QDQ | 0.88565 / 0.93137 | 4,074,656 | CPU p50/p95 39.11/59.98 ms; 24.40 FPS | ผ่าน gate รอบเก่า |
| SLIM50+SVD50 ONNX FP32 | 0.91158 / — | 13,197,154 | max abs parity 0.0000482; CPU p50 19.66 ms | export ใช้งานได้ |
| SLIM50+SVD50 PyTorch FP16 | 0.91157 / 0.95053 | 6,757,779 | GPU p50 9.49 ms; 103.19 FPS | ผ่าน validation gate รอบเก่า |
| SLIM50+SVD50 ONNX INT8 QDQ | 0.87750 / 0.92635 | 3,641,036 | CPU p50/p95 25.59/51.63 ms; 35.02 FPS | ถูกเลือกใน frozen selection รุ่นเก่า |

Frozen selection เลือก SLIM50+SVD50 ONNX INT8 ตาม validation IoU/ขนาดใน compression split. มีไฟล์ final_test.json รายงาน test IoU 0.88415 และ baseline 0.92135 (268 ภาพ, ลด 3.72 pp, แตะ test ครั้งเดียว). อย่างไรก็ตาม Midterm test IDs 212 ภาพทับกับ train split compression เดิม; จึงถือ test score นี้เป็น **contaminated / NOT VERIFIED**, ไม่ใช่ผลทดสอบ final ที่ยืนยันได้.

## ONNX export variants สำหรับ edge

มี export metadata ของ baseline C0, SVD50 และ SLIM50+SVD50 ที่ input 256×256 และ 512×512. Graphs ใช้ opset 16, public image/state tensors เป็น FP32 และมี FP16 initializers: C0 155, SVD50 170, SLIM50+SVD50 166. ขนาด artifacts 256×256 คือ 7,563,834 / 6,870,333 / 6,660,779 bytes; ที่ 512×512 คือ 7,563,908 / 6,870,413 / 6,660,857 bytes ตามลำดับ. การ export metadata ไม่ใช่หลักฐานการรันบน Jetson Nano; benchmark ที่มีอยู่เป็นเครื่องพัฒนา RTX 5070 Ti.

## สถานะรวม

| รายการ | สถานะ |
|---|---|
| Backbone trials: MobileNetV3-Large/Small, V2, V1 | มีผลที่บันทึกไว้; ตัวเลขจาก historical reports อาจมาจากคนละ evaluation run |
| Compression search: CONTROL, SVD50, SLIM50, SLIM75, SLIM50+SVD50, UP40, UP70 | มีผล validation บน split เก่า; ไม่นำมาใช้เป็น final comparison |
| Precision trials: PyTorch FP16, ONNX FP32, ONNX INT8 QDQ, ONNX FP16 | บันทึกผลและข้อจำกัดไว้ด้านบน; ONNX FP16 Augmented parity ยังไม่ผ่าน |
| Train-only Gaussian blur/noise | ผ่าน |
| PyTorch validation IoU gate | ผ่าน A/B |
| Structural parameter reduction | ผ่าน; A −9.35%, B −12.12% |
| ONNX FP16 graph | สร้างได้; มี FP16 initializers |
| ONNX stateful parity ≤ 0.08 (รอบล่าสุด) | **ไม่ผ่าน A/B** |
| Test metrics / leakage-free test | **NOT VERIFIED** |
| Jetson Nano hardware test | **ยังไม่มี** |

ผลนี้วัด binary person segmentation จากภาพนิ่ง ไม่ได้ยืนยัน temporal consistency หรือ alpha-matting ของวิดีโอ.

## หลักฐานและการทำซ้ำ

- [ผล 256×256 ก่อน augmentation](outputs/final_fp16/comparison/results.json)
- [Dataset leakage audit](outputs/final_fp16/audit/dataset_split_audit.json)
- Training history/config: outputs/final_fp16_augmented_20261026/runs/{SVD50,SLIM50_SVD50}/
- Export metadata: outputs/final_fp16_augmented_20261026/exports/{SVD50,SLIM50_SVD50}/onnx_fp16.json
- [ผล 512×512](outputs/final_512/comparison_512.json)
- [คำสั่ง reproduce](outputs/final_fp16/REPRODUCE.md)
