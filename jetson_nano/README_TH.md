# Webcam -> segmentation -> background fill (Jetson Nano 2 เครื่อง)

ตัวอย่างนี้ใช้ Nano 1 รับภาพจาก webcam, รันโมเดล **RVM-MobileNetV3-Large ที่ผ่านการกลั่นความรู้ (Knowledge Distillation) และบีบอัดด้วย SVD Factorization** โดยตั้งค่าเริ่มต้นเป็น `rvm_mobilenetv3_large_svd50.onnx` ซึ่งให้ความแม่นยำสูงกว่าโมเดลกลั่นตัวเดิม `SLIM50_SVD50.onnx` อย่างชัดเจน (+2.07% IoU) แล้วส่งเฟรมภาพ JPEG กับ mask PNG ผ่าน TCP ไป Nano 2 แบบต่อเนื่อง Nano 2 เติมบริเวณ mask ด้วย OpenCV inpainting และบันทึก `received_inpainted.mp4`

### ตารางเปรียบเทียบโมเดลที่กลั่นจาก MobileNetV3-Large
| โมเดลที่กลั่นจาก 3-Large | เทคนิคการกลั่นและบีบอัด | Test IoU 🏆 | Dice (F1) | GPU FPS | ดีกว่าโมเดลกลั่นตัวเดิมอย่างไร? |
|---|---|:---:|:---:|:---:|---|
| **SVD50** *(โมเดลหลักที่ใช้เชื่อม Jetson)* | **SVD 0.50 + Knowledge Distillation** | **93.23%** | **96.20%** | **106.5 FPS** | **ดีกว่าเดิม +2.07% IoU**, คง Recurrent Channels เต็ม ไม่เกิดอาการขอบกระพริบ |
| *SLIM50_SVD50 (ตัวกลั่นเดิม)* | *Slim 0.50 + SVD 0.50 + KD* | *91.16%* | *95.05%* | *112.0 FPS* | *ตัวเดิมก่อนปรับปรุง (ถูกลด Recurrent Channel จนความแม่นยำตก)* |
| *SLIM50_SVD50 (INT8)* | *Quantization INT8 QDQ* | *87.75%* | *92.63%* | *35.0 FPS* | *ตัวเดิมแบบ INT8 (ความแม่นยำลดลงฮวบเหลือ 87.75%)* |
| **MobileNetV3-Large (ตัวเต็ม)** | Reference Baseline | 93.54% | 96.66% | 110.0 FPS | ตัวแม่แบบก่อนการกลั่นและบีบอัด |

*(นอกจากนี้ โฟลเดอร์ `jetson_nano/models/` ยังมีโมเดล Fine-tuned ทางเลือกอย่าง `rvm_mobilenetv1.onnx` สำหรับผู้ที่ต้องการเร่งความเร็ว GPU สตรีมมิ่งสูงสุดถึง 185.6 FPS)*

## 1. เชื่อมต่ออุปกรณ์

เชื่อม Nano ทั้งสองเข้าวง LAN เดียวกัน จด IP ของ Nano 2:

```bash
hostname -I
```

ติดตั้ง OpenCV และ NumPy บนอุปกรณ์ทั้งสอง (`cv2` อาจติดตั้งมาพร้อม JetPack แล้ว) และติดตั้ง ONNX Runtime รุ่นที่รองรับ Jetson บน Nano 1 โดยเลือก build ที่ตรงกับ JetPack/CUDA ของเครื่อง

คัดลอกโฟลเดอร์ `jetson_nano/` ไปยัง Nano 1 (โมเดล ONNX ถูกจัดเตรียมไว้ใน `jetson_nano/models/`)

## 2. เริ่มรับภาพบน Nano 2

```bash
python3 stream_receiver.py --port 5000 --output output/removed.mp4
```

เปิดพอร์ต TCP `5000` ในเครือข่ายระหว่าง Nano หากไม่มีจอหรือรันผ่าน SSH ให้เพิ่ม `--no-display`

## 3. เริ่ม webcam บน Nano 1

```bash
python3 webcam_sender.py \
  --camera 0 \
  --model models/rvm_mobilenetv1.onnx \
  --host <IP-ของ-Nano-2> \
  --port 5000 \
  --preview
```

*(หากไม่ระบุ `--model` ตัวโปรแกรมจะเลือกโมเดลที่ดีที่สุดในโฟลเดอร์ `models/` ให้อัตโนมัติ)*


`--camera 0` ใช้ webcam ตัวแรก; ระบุ `/dev/video0` แทนได้ กด `q` ที่ preview ของ Nano 1 หรือ `Ctrl+C` ใน terminal เพื่อหยุดส่ง กด `q` ในหน้าต่าง Nano 2 เพื่อหยุดรับและปิดไฟล์วิดีโอ

ถ้า ONNX Runtime บน Nano 1 มี CUDA provider ให้ใช้ `--provider cuda`; `auto` จะเลือก CUDA เมื่อพบและ fallback เป็น CPU เมื่อไม่พบ ตรวจ provider ที่เริ่มต้นจากข้อความใน terminal

## ข้อจำกัดของตัวอย่าง

- ตัวอย่างส่งภาพจาก webcam ไปยัง Nano 2 พร้อม mask; โมเดลรับภาพขนาด 256×256 แล้วขยาย mask กลับให้เท่าขนาด webcam
- Nano 2 ใช้ `cv2.inpaint` เติมจากพิกเซลรอบขอบ mask ในแต่ละเฟรม ไม่มีการสร้างพื้นหลังเชิง generative AI ภาพที่ถูกบังเป็นพื้นที่ใหญ่หรือมีลวดลายซับซ้อนอาจเติมได้ไม่สมจริง
- วิดีโอที่บันทึกใช้ค่า FPS จาก `--fps` (ค่าเริ่มต้น 20) ไม่ได้เก็บ timestamp เพื่อรักษาอัตราเฟรมต้นทาง
- ใช้ TCP ใน LAN ที่เชื่อถือได้; ตัวอย่างนี้ไม่มีการเข้ารหัสหรือยืนยันตัวตน
