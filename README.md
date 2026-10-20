# Human Filter for Video — Real-Time Background Matting for Edge Computing

[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2Bcu132-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Device](https://img.shields.io/badge/Device-RTX%205060%20%7C%20Edge%20CPU-success.svg)]()

โครงการวิจัยและพัฒนาโมเดลตัดพื้นหลังคนแบบเรียลไทม์ (**Human Matting & Video Filter**) โดยนำสถาปัตยกรรม **Robust Video Matting (RVM)** มาดัดแปลงและทดลองเปลี่ยน Encoder Backbone เป็นตระกูล **MobileNet (V3-Large, V3-Small, V2, V1)** เพื่อเปรียบเทียบประสิทธิภาพ ความแม่นยำ (IoU, Dice Score) และความเร็วในการประมวลผล (FPS, Latency) สำหรับการนำไปใช้งานบนอุปกรณ์ **Edge Computing / IoT / Live Streaming**

---

## 📌 บทสรุปผลการทดลอง (Executive Summary)

การทดลองนี้ทำการประเมินและเปรียบเทียบโมเดลทั้งหมด **4 สถาปัตยกรรม** บนชุดข้อมูล **Supervisely Filtered Person Dataset** (ความละเอียด 256x256, Test Set 268 ภาพ, Seed 42):

1. **RVM-MobileNetV3 (Large)** — โมเดลทางการ (Official Pretrained) สำหรับคุณภาพระดับสตูดิโอ
2. **RVM-MobileNetV3-Small** — โมเดลขนาดเล็กพิเศษ น้ำหนักเบาที่สุด เหมาะสำหรับอุปกรณ์พกพา / Edge CPU
3. **RVM-MobileNetV2** — โมเดลสถาปัตยกรรม Inverted Residuals ที่ให้จุดสมดุลระหว่างขนาดและความแม่นยำ
4. **RVM-MobileNetV1** — โมเดลสถาปัตยกรรม Standard Depthwise Convolution ที่ทำความเร็วบน GPU ได้สูงสุด

---

## 📊 ตารางเปรียบเทียบตัวชี้วัด (Metrics Comparison Table)

### 1. เปรียบเทียบความแม่นยำ ความซับซ้อน และความเร็วในการประมวลผล

| สถาปัตยกรรมโมเดล (Model) | ขนาดพารามิเตอร์ (Params) | ขนาดไฟล์ FP32 (MB) | GFLOPs @256² | Test BCE Loss | Test IoU 🏆 | Test Dice (F1) | Precision | Recall | GPU Speed (FPS) 🚀 | CPU Speed (FPS) ⚡ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RVM-MobileNetV3 (Large)** *(Official)* | 3.74 M | 14.5 MB | 5.40 | **0.0577** | **0.9354 (93.5%)** | **0.9666 (96.7%)** | **0.9728** | 0.9606 | ~110 FPS | ~18 FPS |
| **RVM-MobileNetV1** *(Fine-tuned)* | 4.21 M | 16.8 MB | 6.25 | 0.0791 | **0.9316 (93.2%)** | **0.9646 (96.5%)** | 0.9645 | **0.9647** | **185.6 FPS** | 20.6 FPS |
| **RVM-MobileNetV2** *(Fine-tuned)* | 2.42 M | 9.67 MB | 4.21 | 0.0686 | **0.9267 (92.7%)** | **0.9619 (96.2%)** | 0.9643 | 0.9596 | 134.7 FPS | 22.8 FPS |
| **RVM-MobileNetV3-Small** *(Fine-tuned)* | **1.59 M** | **6.36 MB** | **3.27** | 0.0830 | **0.9125 (91.3%)** | **0.9543 (95.4%)** | 0.9548 | 0.9537 | 126.8 FPS | **33.9 FPS** |

> *หมายเหตุ: การวัด FPS ทำการทดสอบบน NVIDIA GeForce RTX 5060 Laptop GPU และ Intel Core CPU ที่อินพุตขนาด 256x256 แบบ Streaming Loop ต่อเนื่องพร้อม Recurrent State*

---

### 2. เปรียบเทียบก่อนและหลังการ Fine-tune (Pretrained Baseline vs Fine-tuned)

| โมเดล | การตั้งค่าเริ่มต้น (Initialization) | Baseline IoU (ก่อนเทรน) | Fine-tuned IoU (หลังเทรน) | การปรับปรุงสัมพัทธ์ (Relative Gain) |
|---|---|:---:|:---:|:---:|
| **RVM-MobileNetV3 (Large)** | Official RVM Pretrained (Matting) | 0.9288 (92.88%) | **0.9354 (93.54%)** | +0.66% (+0.71% relative) |
| **RVM-MobileNetV1** | ImageNet Pretrained (timm) | 0.0000 (0.00%) | **0.9318 (93.18%)** | +93.18% (เรียนรู้โครงสร้างใหม่สมบูรณ์) |
| **RVM-MobileNetV2** | ImageNet Pretrained (torchvision) | 0.0000 (0.00%) | **0.9269 (92.69%)** | +92.69% (เรียนรู้โครงสร้างใหม่สมบูรณ์) |
| **RVM-MobileNetV3-Small** | ImageNet Pretrained (torchvision) | 0.3322 (33.22%) | **0.9128 (91.28%)** | +58.05% (+174.7% relative) |

---

## 🔍 การวิเคราะห์และข้อค้นพบเชิงลึก (Key Findings & Insights)

### 1. ความแม่นยำสูงสูสี แม้ลดขนาดพารามิเตอร์ลงเกินเท่าตัว
* **RVM-MobileNetV3 (Large)** ให้ความแม่นยำสูงสุดที่ **IoU 93.54%** เนื่องจากมี Receptive Field กว้างและใช้ Pretrained Matting Weights ทางการ
* **RVM-MobileNetV1** ทำผลงานได้อย่างน่าประทับใจโดยได้ **IoU 93.16%** และ Recall สูงสุดที่ **96.47%** เก็บขอบเส้นผมและขอบตัวคนได้หนาและสม่ำเสมอที่สุด
* **RVM-MobileNetV2** ทำ IoU ได้ **92.67%** โดยใช้ขนาดพารามิเตอร์เพียง **2.42M** (ประหยัดกว่า V1 ถึง 42%)
* **RVM-MobileNetV3-Small** ได้ IoU **91.25%** และ Dice **95.43%** ซึ่งแตกต่างจากตัว Large เพียง ~2.3% แต่ลดขนาดลงเหลือเพียง **1.59M พารามิเตอร์ (6.36 MB)**

### 2. การวิเคราะห์ความเร็ว (Speed & Throughput Analysis)
* **แชมป์ความเร็วบน GPU: MobileNetV1 (185.6 FPS)**
  * เนื่องจากสถาปัตยกรรมของ V1 ใช้การเชื่อมต่อแบบเรียบง่าย (Straightforward Depthwise Separable Convolutions) โดยไม่มี Residual Skip Connection ซับซ้อน ทำให้เกิด Memory Bandwidth Overhead ต่ำบนสถาปัตยกรรม GPU สมัยใหม่
* **แชมป์ความเร็วบน CPU: MobileNetV3-Small (33.9 FPS)**
  * ด้วยขนาดเพียง **3.27 GFLOPs** ทำให้ V3-Small สามารถประมวลผลบน CPU แบบ Single Thread ได้เกิน 30 FPS สบายๆ จึงเป็นโมเดลที่แนะนำมากที่สุดสำหรับอุปกรณ์ Edge Device ที่ไม่มีการ์ดจอแยก (เช่น Raspberry Pi 5, Jetson Nano, หรือ Mini PC)

---

## 📈 แผนภาพผลลัพธ์การทดลอง (Visual Evaluation)

### 1. กราฟความแม่นยำเปรียบเทียบทุกโมเดล (IoU & Dice Score)
![Accuracy Comparison](outputs/figures/backbone_comparison_iou_dice.png)

### 2. แผนภาพ Trade-off: Accuracy vs Inference Speed
![Accuracy vs Speed Tradeoff](outputs/figures/backbone_accuracy_vs_speed.png)
*(ขนาดวงกลมแทนจำนวนพารามิเตอร์ของโมเดล ยิ่งอยู่มุมบนขวายิ่งเร็วและแม่นยำ)*

### 3. การเปรียบเทียบเชิงคุณภาพกับ Ground Truth (Qualitative Comparison)
![Qualitative Comparison](outputs/figures/backbone_qualitative_all.png)

---

## 🛠️ โครงสร้างโปรเจกต์และการใช้งาน

```text
Human-filter-for-video/
├── data/
│   └── supervisely/              # Supervisely Person Dataset (2,667 images & masks)
├── notebooks/
│   ├── rvm_finetune.ipynb                   # โน้ตบุ๊กต้นฉบับ RVM MobileNetV3 (Large)
│   ├── rvm_mobilenetv3_small_finetune.ipynb # โน้ตบุ๊ก RVM MobileNetV3-Small (66 เซลล์)
│   ├── rvm_mobilenetv2_finetune.ipynb       # โน้ตบุ๊ก RVM MobileNetV2 (66 เซลล์)
│   ├── rvm_mobilenetv1_finetune.ipynb       # โน้ตบุ๊ก RVM MobileNetV1 (66 เซลล์)
│   └── rvm_backbone_comparison.ipynb        # โน้ตบุ๊กประมวลผลและเปรียบเทียบทุกโมเดล
├── outputs/
│   ├── checkpoints/              # Checkpoint โมเดล (.pth) และ TorchScript (.torchscript)
│   ├── figures/                  # กราฟและรูปผลลัพธ์ทั้งหมด
│   └── results/                  # ตารางสถิติและประวัติการเทรน (.csv)
├── app.py                        # Streamlit Web Dashboard เปรียบเทียบแบบ Interactive
├── compare_webcam.py             # OpenCV Real-Time Webcam (เทียบทุกโมเดลในจอเดียว + Ground Truth)
├── rvm_backbones.py              # โมดูลสถาปัตยกรรม RVM Encoders รองรับ V3-Small, V2, V1
└── requirements.txt              # รายการ Dependencies
```

---

## 💻 วิธีการรันแอปพลิเคชันเปรียบเทียบ (Interactive Comparison)

### 1. หน้าต่าง OpenCV Real-Time Webcam (60 FPS พร้อม Ground Truth ในจอเดียว)
เปรียบเทียบทุกโมเดลแบบสดๆ ผ่านกล้องเว็บแคม พร้อมพาเนล **Ground Truth Reference** และตารางคะแนน Live IoU / FPS:
```powershell
python compare_webcam.py
```
* **ปุ่มลัด (Hotkeys)**:
  * `m` : สลับเอฟเฟกต์พื้นหลัง (Green Screen, หน้ากากขาว-ดำ, เบลอพื้นหลัง, Cutout สีดำ, Cyberpunk Neon)
  * `g` : สลับการจัดวางหน้าจอ (ตาราง 2x3 Grid หรือ แถว 1x5 Strip)
  * `q` : ออกจากโปรแกรม

### 2. เว็บแดชบอร์ด Streamlit (Web Application)
เปิด Web UI สำหรับทดสอบรูปภาพ, วิดีโอ, ชุดข้อมูล Supervisely และ Live Webcam:
```powershell
streamlit run app.py
```

### 3. การรันโน้ตบุ๊กทดลอง (Jupyter Notebooks)
สามารถเปิดไฟล์โน้ตบุ๊กในโฟลเดอร์ `notebooks/` ด้วย VSCode แล้วกด **Run All** เพื่อรันขั้นตอนตั้งแต่การเตรียมข้อมูล, การเทรน, การประเมินผล, การวาดกราฟ ไปจนถึงการ Export เป็น TorchScript:
* [rvm_mobilenetv3_small_finetune.ipynb](notebooks/rvm_mobilenetv3_small_finetune.ipynb)
* [rvm_mobilenetv2_finetune.ipynb](notebooks/rvm_mobilenetv2_finetune.ipynb)
* [rvm_mobilenetv1_finetune.ipynb](notebooks/rvm_mobilenetv1_finetune.ipynb)
* [rvm_backbone_comparison.ipynb](notebooks/rvm_backbone_comparison.ipynb)

---

## 📦 สรุปคำแนะนำสำหรับการนำไปติดตั้งใช้งานจริง (Deployment Recommendation)

| สถานการณ์การใช้งาน | โมเดลที่แนะนำ | เหตุผล |
|---|:---:|---|
| **อุปกรณ์ Edge / IoT / CPU เท่านั้น** *(เช่น Raspberry Pi, Thin Client)* | **RVM-MobileNetV3-Small** | ขนาดไฟล์เล็กที่สุด (6.36 MB), กินสเปกต่ำเพียง 3.27 GFLOPs, ทำความเร็ว CPU ได้สูงสุดถึง 33.9 FPS |
| **อุปกรณ์ Embedded GPU / Mobile Device** *(เช่น Jetson Orin, Android/iOS)* | **RVM-MobileNetV2** | ขนาดไฟล์พอเหมาะ (9.67 MB), สถาปัตยกรรม Inverted Residuals รองรับฮาร์ดแวร์ NPU/DSP ดีเยี่ยม, IoU สูงถึง 92.67% |
| **เซิร์ฟเวอร์สตรีมมิ่ง / Live Broadcast GPU** *(ต้องการ FPS สูงสุด)* | **RVM-MobileNetV1** | ทำความเร็วบน GPU ได้สูงสุดถึง 185.6 FPS, ความแม่นยำสูง (IoU 93.16%), ตัดขอบคนได้คมและสม่ำเสมอ |
| **งาน Production ตัดต่อวิดีโอคุณภาพสูง** | **RVM-MobileNetV3 (Large)** | ความแม่นยำและ Fine Detail ขอบเส้นผมดีที่สุด (IoU 93.54%), Test Loss ต่ำที่สุด (0.0577) |
