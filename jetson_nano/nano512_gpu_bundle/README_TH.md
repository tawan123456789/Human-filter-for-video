# Jetson Nano 512 GPU bundle

แพ็กเกจนี้สร้างสำหรับ Jetson Nano รุ่นเดิมที่ใช้ JetPack 4.x, CUDA 10.2 และ cuDNN 8 โดยมี ONNX Runtime GPU 1.11 และโมเดล ONNX FP16 ขนาด 512×512 (opset 16) ทั้งสามรุ่น

> CUDA, cuDNN, TensorRT และไดรเวอร์ GPU ต้องมาจาก JetPack ที่ติดตั้งอยู่บน Nano ไฟล์นี้มีตัวโปรแกรม ARM64 และ ONNX Runtime GPU แต่ไม่รวมไลบรารีระบบของ JetPack จึงใช้กับ JetPack รุ่นอื่นโดยไม่ตรวจสอบความเข้ากันได้ไม่ได้

## แตกไฟล์

```bash
tar -xf jetson_nano_512_gpu_bundle.tar
cd jetson_nano_512_gpu_bundle
```

## เปิดแอปทดสอบ webcam และสลับทั้ง 3 โมเดล

ตัวรัน ARM64 ที่รวม Python, OpenCV และ ONNX Runtime ไว้แล้ว:

```bash
chmod +x nano512_realtime/app/nano512_realtime
./nano512_realtime/app/nano512_realtime --provider CUDAExecutionProvider
```

กด `1`, `2`, `3` หรือปุ่มลูกศรเพื่อเปลี่ยนโมเดล, `m` เพื่อสลับ overlay/mask และ `q` เพื่อออก

หรือใช้คำสั่ง Python module ตามเดิม (ต้องมี Python 3.8 สำหรับ ARM64):

```bash
bash install.sh
.venv/bin/python -m top2_benchmark.realtime_app --provider CUDAExecutionProvider
```

## ส่งภาพ webcam และ mask ไป Nano อีกเครื่อง

บน Nano ปลายทาง เปิดตัวรับข้อมูล (ต้องมี Python 3):

```bash
python3 jetson_nano/stream_receiver.py --bind 0.0.0.0 --port 5000 --output received_inpainted.mp4
```

บน Nano ที่ต่อกล้อง ใช้ executable ที่รวมมาแล้ว:

```bash
chmod +x nano512_webcam_to_jetson/app/webcam_to_json
./nano512_webcam_to_jetson/app/webcam_to_json --host <NANOปลายทาง_IP> --port 5000 --variant slim50_svd50 --provider cuda --preview
```

เลือก `--variant baseline`, `svd50` หรือ `slim50_svd50` ได้ หากต้องการรัน source แทน executable ให้ติดตั้งด้วย `bash install.sh` แล้วใช้:

```bash
.venv/bin/python -m jetson_nano.webcam_to_json --host <NANOปลายทาง_IP> --port 5000 --variant slim50_svd50 --provider cuda --preview
```

กด `q` ในหน้าต่าง preview หรือ `Ctrl+C` เพื่อหยุด การทดสอบ export และ inference ทำบน workstation; ยังไม่ได้วัดความเร็วหรือทดสอบกล้องบนอุปกรณ์ Jetson Nano จริง
