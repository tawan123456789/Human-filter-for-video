# Jetson Nano standalone bundle (ไม่ต้องติดตั้ง Python packages)

ใน tar จะมี executable Linux ARM64 แบบ PyInstaller สำหรับ realtime webcam และ webcam-to-Jetson, พร้อม Python runtime, OpenCV, NumPy, ONNX Runtime GPU 1.11.0 และ Model A/B ONNX FP16 ไว้ในโฟลเดอร์แอป ไม่ต้องติดตั้ง wheel หรือ `pip install` บน Nano

ต้องมี JetPack ที่เข้ากับ ONNX Runtime GPU 1.11.0 อยู่บน Nano เพราะ CUDA, cuDNN, TensorRT และไดรเวอร์ NVIDIA เป็นไลบรารีระดับระบบของ JetPack ซึ่งไม่สามารถฝังทดแทนใน executable ได้

## รัน realtime app บน Nano เครื่องเดียว

```bash
tar -xf jetson_nano_modelA_modelB_standalone_bundle.tar
cd jetson_nano_modelA_modelB_standalone_bundle
cd realtime_app
./realtime_app --provider CUDAExecutionProvider
```

หน้าต่าง OpenCV แสดง webcam และผล segmentation แบบสด: `1` เลือก baseline, `2` เลือก Model A, `3` เลือก Model B; ใช้ปุ่มลูกศรซ้าย/ขวาเพื่อวนสลับ, `m` สลับ mask/overlay และ `q` เพื่อออก

## ส่ง webcam + mask ไป Nano อีกเครื่อง

บน Nano ปลายทาง ให้เริ่ม receiver ด้วย Python/OpenCV ที่มากับ JetPack (ไม่ต้องมี ONNX Runtime):

```bash
python3 jetson_nano/stream_receiver.py --bind 0.0.0.0 --port 5000 --output received_inpainted.mp4
```

บน Nano ที่ต่อกล้อง:

```bash
cd webcam_to_jetson
./webcam_to_jetson --host <IP_NANOปลายทาง> --port 5000 --variant svd50 --provider cuda --preview
```

เลือก Model B ด้วย `--variant slim50_svd50`; กด `q` ใน preview หรือ `Ctrl+C` เพื่อหยุด

## ข้อจำกัดของ ONNX FP16 ชุดนี้

ไฟล์ ONNX เป็น opset 16 และผ่าน graph checker แต่ parity กับ PyTorch FP16 บนเครื่องพัฒนายังเกินเกณฑ์ max error 0.08 จึงแพ็กมาเพื่อทดสอบบน Jetson ตามคำขอ และยังไม่รับรองเป็น production model ผลที่ได้บน Nano อาจขึ้นกับ CUDA EP และ TensorRT/CUDA libraries ของ JetPack
