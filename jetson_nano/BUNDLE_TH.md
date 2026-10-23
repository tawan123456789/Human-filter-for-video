# แพ็ก `webcam_sender` พร้อมโมเดล

ตัวส่งทำงานต่อเนื่องตามลำดับ webcam → `SLIM50_SVD50.onnx` → JPEG ภาพต้นฉบับและ PNG mask → TCP ไป Nano 2 โดยคง recurrent state ของโมเดลข้ามเฟรม และตั้ง webcam buffer ให้เก็บเฟรมล่าสุดเพื่อลดภาพค้าง โมเดลถูกรวมไว้ในโฟลเดอร์โปรแกรม จึงไม่ต้องส่งโมเดลซ้ำทางเครือข่าย

## สร้างแพ็กเกจสำหรับ Nano 1

บน Windows ที่ติดตั้งและเปิด Docker Desktop แล้ว สามารถสร้าง Linux ARM64 package โดยใช้ Docker ได้:

```powershell
.\jetson_nano\build_nano_bundle.ps1
```

หรือ build บน Nano 1 ด้วย Linux ARM64 และ Python 3.8 ขึ้นไปที่มี Jetson-compatible `onnxruntime`, OpenCV และ NumPy:

```bash
cd AI-edge/jetson_nano
bash build_nano_bundle.sh
```

ผลลัพธ์จาก PowerShell อยู่ที่ `dist/nano1_webcam_sender_opset16_gpu_bundle.tar` ให้คัดลอกไฟล์นี้ไปยัง Nano 1 แล้วแตกไฟล์ โฟลเดอร์ `nano1_webcam_sender/` ที่ได้รวม ONNX ไว้แล้ว Dockerfile ใช้ Ubuntu 18.04 ARM64 เพื่อให้เข้ากับ JetPack 4

> JetPack 4.6.x บน Nano มักติดตั้ง Python 3.6 มาให้เป็นค่าเริ่มต้น ซึ่งใช้ PyInstaller รุ่นปัจจุบันไม่ได้ และต้องมี ONNX Runtime build ที่เข้ากับ JetPack/CUDA ด้วย การรวม ONNX ในแอปไม่ได้รวมไดรเวอร์ JetPack หรือ CUDA ของระบบ

## รัน

เริ่ม receiver บน Nano 2 ก่อน แล้วบน Nano 1:

```bash
tar -xf nano1_webcam_sender_opset16_gpu_bundle.tar
cd nano1_webcam_sender
./nano1_webcam_sender --host <IP ของ Nano 2> --port 5000 --camera 0
```

โปรแกรมค้นหา `model/SLIM50_SVD50.onnx` ในแพ็กเกจให้อัตโนมัติ หากต้องการดู preview ให้เพิ่ม `--preview`; กด `q` เพื่อหยุดเมื่อเปิด preview หรือกด `Ctrl+C` ใน terminal

ทดสอบจาก source โดยไม่แพ็กได้เช่นกัน:

```bash
python3 webcam_sender.py --host <IP ของ Nano 2> --port 5000 --camera 0
```

เมื่อรันจาก source ตัวโปรแกรมจะหาโมเดลใน `compression/outputs/jetson_nano/` ให้อัตโนมัติ หรือกำหนด `--model /path/to/SLIM50_SVD50.onnx` เองได้

## แอปดูภาพ webcam และ mask บน Nano 1

แอปนี้ประมวลผลบน Nano 1 และแสดง webcam ทางซ้ายกับ person mask ทางขวา ไม่ส่งข้อมูลไป Nano 2 กด `q` หรือ Esc ที่หน้าต่าง preview เพื่อปิด

Build บน Windows ด้วย Docker Desktop:

```powershell
.\jetson_nano\build_nano_preview.ps1
```

ไฟล์ `dist/nano1_mask_preview_opset16_gpu_bundle.tar` รวมโมเดล ONNX opset 16 และ ONNX Runtime GPU 1.11.0 จาก wheel ใน `jetson_nano/wheels/` ไว้แล้ว คัดลอกไป Nano 1 แล้วรัน:

```bash
tar -xf nano1_mask_preview_opset16_gpu_bundle.tar
cd nano1_mask_preview
./nano1_mask_preview --camera 0
```

ตัวโปรแกรมเลือก CUDA เป็นค่าเริ่มต้น หากต้องการบังคับ CPU ให้เพิ่ม `--provider cpu`; เลือกกล้องอื่นได้ด้วย `--camera 1` หรือระบุ device path เช่น `--camera /dev/video0`; ปรับภาพได้ด้วย `--width 640 --height 480` โดยค่าเริ่มต้น

GPU provider ต้องใช้ JetPack/CUDA/cuDNN/TensorRT libraries ที่เข้ากับ wheel บน Nano 1 ด้วย; หาก provider โหลดไม่ได้ โปรแกรมจะแจ้ง error ให้ตรวจ JetPack และ dependency บนบอร์ด
