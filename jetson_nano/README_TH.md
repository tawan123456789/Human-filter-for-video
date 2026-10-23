# Webcam -> segmentation -> background fill (Jetson Nano 2 เครื่อง)

ตัวอย่างนี้ใช้ Nano 1 รับภาพจาก webcam, รัน `SLIM50_SVD50.onnx`, แล้วส่งเฟรมภาพ JPEG กับ mask PNG ผ่าน TCP ไป Nano 2 แบบต่อเนื่อง Nano 2 เติมบริเวณ mask ด้วย OpenCV inpainting และบันทึก `received_inpainted.mp4` ไฟล์ ONNX เป็นโมเดล stateful สำหรับวิดีโอ โดยส่ง recurrent states ต่อเนื่องทีละเฟรม

## 1. เชื่อมต่ออุปกรณ์

เชื่อม Nano ทั้งสองเข้าวง LAN เดียวกัน จด IP ของ Nano 2:

```bash
hostname -I
```

ติดตั้ง OpenCV และ NumPy บนอุปกรณ์ทั้งสอง (`cv2` อาจติดตั้งมาพร้อม JetPack แล้ว) และติดตั้ง ONNX Runtime รุ่นที่รองรับ Jetson บน Nano 1 โดยเลือก build ที่ตรงกับ JetPack/CUDA ของเครื่อง

คัดลอกโฟลเดอร์ `jetson_nano/` และไฟล์โมเดล `SLIM50_SVD50.onnx` ไปยังอุปกรณ์ทั้งสอง หรือเก็บโมเดลไว้เฉพาะ Nano 1

## 2. เริ่มรับภาพบน Nano 2

```bash
python3 stream_receiver.py --port 5000 --output output/removed.mp4
```

เปิดพอร์ต TCP `5000` ในเครือข่ายระหว่าง Nano หากไม่มีจอหรือรันผ่าน SSH ให้เพิ่ม `--no-display`

## 3. เริ่ม webcam บน Nano 1

```bash
python3 webcam_sender.py \
  --camera 0 \
  --model SLIM50_SVD50.onnx \
  --host <IP-ของ-Nano-2> \
  --port 5000 \
  --preview
```

`--camera 0` ใช้ webcam ตัวแรก; ระบุ `/dev/video0` แทนได้ กด `q` ที่ preview ของ Nano 1 หรือ `Ctrl+C` ใน terminal เพื่อหยุดส่ง กด `q` ในหน้าต่าง Nano 2 เพื่อหยุดรับและปิดไฟล์วิดีโอ

ถ้า ONNX Runtime บน Nano 1 มี CUDA provider ให้ใช้ `--provider cuda`; `auto` จะเลือก CUDA เมื่อพบและ fallback เป็น CPU เมื่อไม่พบ ตรวจ provider ที่เริ่มต้นจากข้อความใน terminal

## ข้อจำกัดของตัวอย่าง

- ตัวอย่างส่งภาพจาก webcam ไปยัง Nano 2 พร้อม mask; โมเดลรับภาพขนาด 256×256 แล้วขยาย mask กลับให้เท่าขนาด webcam
- Nano 2 ใช้ `cv2.inpaint` เติมจากพิกเซลรอบขอบ mask ในแต่ละเฟรม ไม่มีการสร้างพื้นหลังเชิง generative AI ภาพที่ถูกบังเป็นพื้นที่ใหญ่หรือมีลวดลายซับซ้อนอาจเติมได้ไม่สมจริง
- วิดีโอที่บันทึกใช้ค่า FPS จาก `--fps` (ค่าเริ่มต้น 20) ไม่ได้เก็บ timestamp เพื่อรักษาอัตราเฟรมต้นทาง
- ใช้ TCP ใน LAN ที่เชื่อถือได้; ตัวอย่างนี้ไม่มีการเข้ารหัสหรือยืนยันตัวตน
