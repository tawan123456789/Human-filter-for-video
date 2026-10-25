# Jetson Nano 256 GPU bundle

แพ็กเกจเดียวนี้รวม executable ARM64, Python runtime, OpenCV, ONNX Runtime GPU 1.11, source code และ ONNX FP16 ทั้ง 3 โมเดลที่ input/output 256×256 และ opset 16 ไว้ครบแล้ว ไม่มี hotfix หรือไฟล์ runtime แยก

ต้องใช้ Jetson Nano รุ่นเดิมที่ติดตั้ง JetPack 4.x, CUDA 10.2 และ cuDNN 8 ไว้แล้ว ไลบรารี CUDA/cuDNN/TensorRT ของระบบและ driver มาจาก JetPack บน Nano จึงไม่ได้อยู่ใน tar

## แตกไฟล์และรันแอปเทียบ webcam

```bash
tar -xf jetson_nano_256_gpu_bundle.tar
cd jetson_nano_256_gpu_bundle
bash realtime_256.sh
```

หรือเรียก executable โดยตรง: `./nano256_realtime/app/nano256_realtime --provider CUDAExecutionProvider`

กด `1`, `2`, `3` หรือปุ่มลูกศรเพื่อสลับโมเดล, `m` เพื่อสลับ overlay/mask และ `q` เพื่อออก

## ส่ง webcam + mask ไป Nano อีกเครื่อง

เปิด receiver บน Nano ปลายทาง:

```bash
python3 jetson_nano/stream_receiver.py --bind 0.0.0.0 --port 5000 --output received_inpainted.mp4
```

บน Nano ที่ต่อ webcam:

```bash
./nano256_webcam_to_jetson/app/webcam_to_json --host <NANOปลายทาง_IP> --port 5000 --variant slim50_svd50 --provider cuda --preview
```

เลือก `--variant baseline`, `svd50` หรือ `slim50_svd50` ได้

## รันแบบ Python module (ทางเลือก)

ต้องมี Python 3.8 สำหรับ ARM64 บน Nano ก่อน:

```bash
bash install.sh
.venv/bin/python -m top2_benchmark.realtime_app --provider CUDAExecutionProvider
```

ตัว executable ข้างต้นรวม Python runtime และ `libpython3.8.so.1.0` ไว้ใต้ `_internal` ใน bundle แล้ว ไม่ต้องติดตั้ง Python 3.8 แยกเพื่อรัน executable
