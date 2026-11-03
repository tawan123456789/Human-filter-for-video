# Human Filter for Video — Real-time Human Matting for Edge Devices

โปรเจกต์นี้ทดลองใช้ **Robust Video Matting (RVM)** แยกบุคคลออกจากฉากหลังแบบต่อเนื่องสำหรับภาพนิ่ง วิดีโอ และเว็บแคม โดยเปรียบเทียบ backbone หลายขนาด ปรับแต่งโมเดลด้วยชุดข้อมูล Supervisely ทดลองบีบอัดและแปลงโมเดลเป็น ONNX แล้วนำไปทดลองบนเครื่อง Edge เช่น Jetson Nano

ตัวโปรแกรม RVM ประมวลผลทีละเฟรมและส่ง recurrent states 4 ชุดต่อจากเฟรมก่อนหน้า จึงรักษาความต่อเนื่องของวิดีโอได้ เส้นทางหลักของข้อมูลคือ:

```text
ภาพ/วิดีโอ/เว็บแคม
        │
        ├── ฝึกและประเมิน ── notebooks/ + rvm_backbones.py + third_party/RobustVideoMatting
        │                         └── checkpoints และผลลัพธ์
        ├── ทดลองใช้งาน ───── app.py / webcam.py / compare_webcam.py
        ├── บีบอัดโมเดล ───── compression/ ──> PyTorch / ONNX ──> compression_app.py
        └── ทดลองบนอุปกรณ์ ─ export_onnx.py ──> jetson_nano/models/*.onnx
                                  └── Jetson apps หรือ webcam/network streaming
```

## เริ่มต้นใช้งาน

ต้องมี Python และ dependency ตามสภาพแวดล้อมที่ใช้ (CPU, CUDA หรือ Jetson) รายการหลักอยู่ใน [`requirements.txt`](requirements.txt) ส่วนเครื่องมือ quantization อยู่ใน [`compression/requirements-quant.txt`](compression/requirements-quant.txt) และ environment สำหรับ benchmark GPU อยู่ใน [`top2_benchmark/requirements-gpu.txt`](top2_benchmark/requirements-gpu.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch ใน `requirements.txt` ระบุ build CUDA เฉพาะ จึงควรเลือก wheel ให้ตรงกับระบบและ GPU หากใช้ CPU หรือ Jetson ก่อนติดตั้ง dependency ทั้งหมด โค้ดอ้างอิง implementation ต้นทางไว้ที่ `third_party/RobustVideoMatting/`; ชุดข้อมูลและ checkpoint สำหรับฝึก/ส่งออกโมเดลเป็นไฟล์ภายนอกและไม่ได้รวมไว้ใน Git

หาก checkout ใหม่แล้วยังไม่มี source RVM ให้เตรียมไว้ตาม path ที่โค้ดคาดหวัง:

```powershell
git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
```

ตัวอย่างคำสั่งจาก root ของ repository:

```powershell
streamlit run app.py                 # แดชบอร์ดเปรียบเทียบโมเดล
python webcam.py                     # segmentation จาก webcam ด้วย PyTorch
python compare_webcam.py             # เปรียบเทียบหลาย backbone แบบสด
streamlit run compression_app.py     # ตรวจผล compression/exports
streamlit run jetson_nano/realtime_app.py  # แอป ONNX สำหรับเว็บแคม
```

## โครงสร้างไฟล์และหน้าที่

### ไฟล์ระดับ root

| ไฟล์ | หน้าที่ | เชื่อมกับ |
|---|---|---|
| `app.py` | Streamlit dashboard สำหรับเทียบ RVM กับภาพ วิดีโอ และ webcam รวมถึงดูผล/metrics | โหลด checkpoint และใช้ `rvm_backbones.py`, `outputs/`, `data/` |
| `webcam.py` | รันโมเดลกับ webcam หนึ่งตัวและแสดงผล segmentation | ใช้ checkpoint ใน `outputs/checkpoints/` และโมเดลใน `rvm_backbones.py` |
| `compare_webcam.py` | เปรียบเทียบ backbone หลายรุ่นแบบเรียลไทม์ในหน้าต่าง OpenCV | `rvm_backbones.py`, checkpoint และภาพอ้างอิง |
| `rvm_backbones.py` | สร้าง RVM ที่เปลี่ยน encoder เป็น MobileNetV3-Small, MobileNetV2 หรือ MobileNetV1 โดยคง decoder/recurrent path ไว้ | ใช้โดยแอป webcam, notebook, export และ compression |
| `export_onnx.py` | สร้าง ONNX แบบมี recurrent inputs/outputs จาก checkpoint ของ backbone ที่รองรับ เพื่อ deploy | อ่าน `outputs/checkpoints/`, เขียนโมเดลไป `jetson_nano/models/` |
| `compression_app.py` | Streamlit UI สำหรับทดลองเปรียบเทียบ candidate, checkpoint และ ONNX ที่สร้างจาก pipeline compression | อ่าน `compression/outputs/` และชุดข้อมูล |
| `run_v1.py` | รัน notebook fine-tune ของ MobileNetV1 ผ่าน `nbclient` พร้อมบันทึกผลลง notebook | `notebooks/rvm_mobilenetv1_finetune.ipynb` |
| `requirements.txt` | dependency หลักของโปรเจกต์ รวม PyTorch, OpenCV, Streamlit, Jupyter และ ONNX | ใช้ตั้งค่าสภาพแวดล้อม Python หลัก |
| `.gitignore` | กันข้อมูลดิบ, checkpoint, ONNX และผลลัพธ์ขนาดใหญ่จากการ commit; อนุญาตให้เก็บกราฟ/CSV ที่เลือกไว้ | ใช้ควบคุมสิ่งที่ Git ติดตาม |

### `notebooks/` — ฝึกและเปรียบเทียบโมเดล

| ไฟล์ | หน้าที่ | ผล/การเชื่อมต่อ |
|---|---|---|
| `rvm_finetune.ipynb` | notebook fine-tune RVM-MobileNetV3-Large รุ่นตั้งต้น | checkpoint, training history และกราฟใน `outputs/` |
| `rvm_mobilenetv3_small_finetune.ipynb` | fine-tune รุ่น MobileNetV3-Small | ใช้ custom backbone และบันทึก metrics/checkpoint |
| `rvm_mobilenetv2_finetune.ipynb` | fine-tune รุ่น MobileNetV2 | ใช้ custom backbone และบันทึก metrics/checkpoint |
| `rvm_mobilenetv1_finetune.ipynb` | fine-tune รุ่น MobileNetV1 | ใช้ custom backbone และบันทึก metrics/checkpoint |
| `rvm_backbone_comparison.ipynb` | สรุปและเปรียบเทียบผลของแต่ละ backbone | อ่านผลฝึก/ประเมินและสร้างตารางหรือกราฟใน `outputs/` |
| `jetson_nano_webcam_test.ipynb` | บันทึกการทดลองเว็บแคมและการทดสอบบน Jetson Nano | ภาพสรุปอยู่ใน `outputs/figures/` |

### `compression/` — ทดลองบีบอัด, ประเมิน และ export

เอกสารวิธีทำซ้ำและข้อจำกัดอยู่ใน [`compression/README.md`](compression/README.md); ผลการทดลองและข้อควรระวังเรื่อง split/data leakage อยู่ใน [`compression/EXPERIMENT_RESULTS.md`](compression/EXPERIMENT_RESULTS.md). สคริปต์เหล่านี้ทำงานร่วมกันเป็นลำดับ: ตรวจ environment → วัด baseline/ค้นหา candidate → ฝึกฟื้นคุณภาพ → export/quantize → เลือกรุ่นจาก validation และตรวจ test ขั้นสุดท้าย

| ไฟล์ | หน้าที่ |
|---|---|
| `__init__.py` | ทำให้โฟลเดอร์เป็น Python package เพื่อเรียกด้วย `python -m compression.<module>` |
| `core.py` | ฟังก์ชันกลางสำหรับโหลด RVM/ข้อมูล, split, metrics, benchmark, hash และเขียนผลอย่างปลอดภัย |
| `operators.py` | ตัวดำเนินการ compression: ลดขนาด decoder, factorization แบบ SVD และ pruning ด้วย masks |
| `train.py` | ฝึกหรือ fine-tune candidate ที่บีบอัด พร้อมบันทึก checkpoint/history/config |
| `search.py` | วนทดลอง configuration หลายรายการตาม profile แล้วรวมผลค้นหา |
| `quantization.py` | สร้าง/ตรวจ FP16 และ ONNX (รวม INT8 เมื่อรองรับ), ตรวจ parity, metrics และ latency |
| `evaluate.py` | ประเมิน candidate ที่เลือกบน validation และบันทึกผล |
| `finalize.py` | freeze ตัวเลือกจาก validation และทำ final test audit ตามขั้นตอนที่กำหนด |
| `finalize_fp16.py` | สร้าง/สรุปชุดผล export FP16 ขั้นสุดท้ายสำหรับ workflow นั้น |
| `fp16_cpu_check.py` | ตรวจการรันโมเดล FP16 บน CPU และประเมินความตรงกัน/คุณภาพ |
| `final_audit.py` | ตรวจความสอดคล้องของ split และผล final audit |
| `doctor.py` | ตรวจความพร้อมของ CUDA, checkpoint, dataset และ RVM source ก่อนใช้เวลาฝึก |
| `evaluate_baseline_backbones.py` | วัด baseline ของ backbone ต่าง ๆ ด้วยชุดประเมินเดียวกัน |
| `benchmark_backbone_gpu.py` | วัด latency/FPS ของ backbone บน GPU |
| `export_jetson_nano.py` | export artifact สำหรับ workflow Jetson Nano |
| `export_512.py` | สร้างและวัดโมเดล ONNX ขนาด input 512 สำหรับ benchmark/app รุ่น Top 2 |
| `configs/quick.json` | ชุด candidate ขนาดเล็กสำหรับเริ่ม compression search |
| `configs/full.json` | ชุด candidate สำหรับค้นหาแบบเต็ม |
| `configs/final_fp16.json` | ค่าตั้งต้นของ workflow final FP16 |
| `requirements-quant.txt` | dependency เพิ่มเติมสำหรับ ONNX/quantization; ติดตั้งเฉพาะเมื่อทำขั้นตอนนี้ |
| `outputs/.gitignore` | ป้องกันผลทดลอง/checkpoint/export ขนาดใหญ่จากการเพิ่มเข้า Git |

ผลที่สคริปต์สร้างจะอยู่ใต้ `compression/outputs/` (เช่น `runs/`, `exports/`, `reports/`, split manifest และ log). โดยทั่วไปไฟล์เหล่านี้สร้างในเครื่องจากข้อมูลและ checkpoint ไม่ได้ commit ไว้

### `top2_benchmark/` — แอปเทียบ artifact 3 รุ่น

| ไฟล์ | หน้าที่ |
|---|---|
| `README.md` | ขั้นตอนสร้าง ONNX 512 และเปิดแอป |
| `spec.md` | ข้อกำหนดและบริบทของ benchmark |
| `app.py` | Streamlit สำหรับเทียบ ONNX FP16 ด้วยภาพ/validation และดู latency |
| `realtime_app.py` | แอป OpenCV สำหรับสลับโมเดล 3 รุ่นบน webcam แบบเรียลไทม์ |
| `requirements-gpu.txt` | dependency ONNX Runtime GPU สำหรับ environment ที่รองรับ |

โมเดลที่แอปคาดหวังสร้างโดย `compression/export_512.py` ไปไว้ใน `compression/outputs/final_512/`; แอปไม่ฝึกโมเดลเอง

### `jetson_nano/` — แอปและ bundle สำหรับ edge deployment

คู่มือหลักคือ [`jetson_nano/README_TH.md`](jetson_nano/README_TH.md), [`REALTIME_APP_TH.md`](jetson_nano/REALTIME_APP_TH.md), [`WEBCAM_TO_JETSON_TH.md`](jetson_nano/WEBCAM_TO_JETSON_TH.md) และ [`BUNDLE_TH.md`](jetson_nano/BUNDLE_TH.md).

| ไฟล์/โฟลเดอร์ | หน้าที่ |
|---|---|
| `realtime_app.py` | Streamlit webcam app ที่ค้นหา ONNX, เลือก execution provider และแสดง mask/effect |
| `webcam_segment.py` | รัน segmentation จากกล้องบน Jetson และเลือกบันทึกผลได้ |
| `webcam_mask_preview.py` | แสดงภาพกล้องพร้อม mask เพื่อ preview/debug |
| `webcam_sender.py` | อ่าน webcam, รัน ONNX บนเครื่องส่ง แล้วส่ง frame และ mask ทาง TCP |
| `stream_receiver.py` | รับข้อมูลจาก sender, แสดง/เติมบริเวณ mask และบันทึกวิดีโอ |
| `webcam_to_jetson.py` | workflow ส่งภาพและ mask ไปยัง Jetson receiver ตามคู่มือการเชื่อมต่อ |
| `webcam_to_json.py` | รุ่นส่งข้อมูลพร้อม metadata/JSON header ผ่าน TCP |
| `build_standalone_bundle.py` | จัดเตรียมไฟล์โปรแกรมและ dependency ที่ต้องใช้เป็น standalone bundle |
| `build_nano_bundle.sh`, `build_nano_bundle.ps1` | สร้าง bundle สำหรับ Nano จาก Linux shell หรือ PowerShell |
| `build_nano_preview.ps1` | สร้าง bundle preview สำหรับทดสอบ |
| `build_tensorrt_engine.sh` | สร้าง TensorRT engine บนอุปกรณ์เป้าหมาย |
| `check_jetson_env.py` | ตรวจรุ่นระบบ, Python และ dependency บน Jetson |
| `jetson_boost.sh` | ปรับโหมดกำลัง/ความถี่ของ Jetson สำหรับ benchmark (ต้องมีสิทธิ์ระบบ) |
| `Dockerfile.jetson_nano`, `Dockerfile.jetson_preview` | image สำหรับ runtime/preview บน Jetson |
| `Dockerfile.nano256_gpu`, `Dockerfile.nano512_gpu` | image แยกตาม workflow GPU และ resolution |
| `model_test_bundle/` | ชุดทดสอบโมเดลแบบ standalone พร้อม app, sender, requirements, installer และคู่มือ/สถานะโมเดล |
| `nano256_gpu_bundle/`, `nano512_gpu_bundle/` | installer และ launcher สำหรับ bundle GPU 256/512 พร้อมคู่มือ |
| `models/.gitkeep` | เก็บโฟลเดอร์ปลายทางของ ONNX ใน Git; ตัวโมเดลไม่ถูกติดตาม |
| `wheels/onnxruntime_gpu-...whl` | wheel ONNX Runtime GPU สำหรับ Python 3.8/aarch64 ตามสภาพแวดล้อมที่ระบุ |

เส้นทาง streaming โดยย่อ: sender อ่านกล้องและรัน ONNX → ส่งภาพกับ mask ผ่าน TCP → receiver สร้าง preview หรือ inpaint และบันทึกผล ส่วน `realtime_app.py` ทำงานในเครื่องที่เปิดหน้าเว็บและต่อกล้องอยู่โดยตรง

### `build/` — เครื่องมือสร้าง bundle

| ไฟล์ | หน้าที่ |
|---|---|
| `consolidate_jetson_bundle.py` | รวม/จัดระเบียบไฟล์ที่ staging ไว้สำหรับชุด Jetson |
| `convert_all_fp16.py` | แปลง ONNX หลาย artifact เป็น FP16 สำหรับ bundle |
| `stage_jetson_nano_256.py` | เตรียมไฟล์และโครงสร้างสำหรับ bundle Nano 256 |

### `third_party/RobustVideoMatting/`

เป็น source implementation ของ upstream RVM ที่ `rvm_backbones.py` และ pipeline ฝึก/ประเมินนำมาอ้างอิง โดยส่วน encoder/decoder และ utility อยู่ใน `model/`, `inference.py` กับ `inference_utils.py` ใช้ inference, `train.py`/`train_config.py`/`train_loss.py` ใช้ฝึก, `dataset/` มี loader/augmentation และ `evaluation/` มีสคริปต์ประเมินผล โฟลเดอร์นี้เป็น dependency ภายนอก มี license ของ upstream ที่ [`third_party/RobustVideoMatting/LICENSE`](third_party/RobustVideoMatting/LICENSE); ตรวจเงื่อนไข license ก่อนเผยแพร่หรือแจกจ่ายต่อ

### `data/` และ `outputs/` — ข้อมูลและผลลัพธ์

| พาธ | เนื้อหา/บทบาท |
|---|---|
| `data/` | ชุดข้อมูลต้นทาง เช่น Supervisely images/masks; ถูก ignore โดย Git และต้องจัดเตรียมเอง |
| `outputs/checkpoints/` | checkpoint จากการฝึก; ไม่ควร commit ไฟล์ขนาดใหญ่โดยไม่วางแผนจัดเก็บ |
| `outputs/figures/` | ภาพสรุปผลและกราฟที่แสดงใน README/รายงาน |
| `outputs/results/` | CSV ของผล baseline, training history, backbone comparison และ baseline-vs-finetuned |
| `compression/outputs/` | checkpoint, ONNX, manifest และรายงานของ compression workflow; ถูก ignore |

ภาพใน `outputs/figures/` ครอบคลุม accuracy/speed (`backbone_*`), ภาพ segmentation (`background_removed*`, `before_after_*`, `prediction_sample_*`, `rvm_matting_result*`) และกราฟ training/validation (`curves_*`, `loss_curve*`, `val_iou*`, `val_dice*`) รวมถึง `jetson_nano_webcam_test.png`. CSV ใน `outputs/results/` จับคู่กันตาม backbone ได้แก่ `baseline*_results.csv`, `training_history*.csv`, `baseline_vs_finetuned*.csv` และ `backbone_comparison.csv`.

## เอกสารและการตรวจสอบ

| พาธ | ใช้สำหรับ |
|---|---|
| [`compression/README.md`](compression/README.md) | วิธีทำ compression workflow ตั้งแต่ setup จน finalize |
| [`compression/EXPERIMENT_RESULTS.md`](compression/EXPERIMENT_RESULTS.md) | ผลทดลอง compression, เงื่อนไขการวัด และข้อจำกัดของข้อมูล |
| [`jetson_nano/README_TH.md`](jetson_nano/README_TH.md) | การส่งภาพ/segmentation ระหว่าง Jetson Nano |
| [`jetson_nano/REALTIME_APP_TH.md`](jetson_nano/REALTIME_APP_TH.md) | การเปิด webcam app บนอุปกรณ์ |
| [`jetson_nano/WEBCAM_TO_JETSON_TH.md`](jetson_nano/WEBCAM_TO_JETSON_TH.md) | การเชื่อมเครื่อง webcam กับ receiver บน Jetson |
| [`jetson_nano/BUNDLE_TH.md`](jetson_nano/BUNDLE_TH.md) | สร้างและติดตั้ง deployment bundle |
| `jetson_nano/model_test_bundle/*README*.md`, `MODEL_STATUS_TH.md` | คู่มือ standalone และสถานะโมเดลชุดทดสอบ |
| `jetson_nano/nano256_gpu_bundle/README_TH.md`, `nano512_gpu_bundle/README_TH.md` | คู่มือ bundle GPU แยกตามขนาด input |
| [`top2_benchmark/README.md`](top2_benchmark/README.md) | สร้างโมเดลและรัน benchmark รุ่น Top 2 |
| [`tests/INSTALL_TH.md`](tests/INSTALL_TH.md) | วิธีเพิ่ม/ติดตั้ง compression patch ใน repository |
| `tests/test_compression.py` | unit tests สำหรับ compression operators, split และ ONNX helpers |

## หมายเหตุเรื่องไฟล์ที่ต้องสร้างเอง

Git ไม่ได้เก็บข้อมูลดิบใน `data/`, checkpoint ใน `outputs/checkpoints/` หรือไฟล์โมเดล ONNX ใน `jetson_nano/models/` และ `compression/outputs/` ไว้ใน repository ตามค่า ignore ปกติ จึงต้องเตรียมไฟล์เหล่านี้ก่อนรัน workflow ที่เกี่ยวข้อง ตรวจ path และ license ของข้อมูล/โมเดลก่อนเผยแพร่

ผล benchmark ขึ้นกับอุปกรณ์, provider, resolution, preprocessing และการ carry recurrent states; เทียบตัวเลขได้เมื่อเงื่อนไขเดียวกัน โปรดใช้ [`compression/EXPERIMENT_RESULTS.md`](compression/EXPERIMENT_RESULTS.md) เป็นแหล่งอ้างอิงผล compression และข้อจำกัดของ split
