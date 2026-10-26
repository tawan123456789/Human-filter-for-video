# Jetson Nano Model A / Model B FP16 test bundle

Bundle นี้รวม ONNX FP16 ของโมเดลที่ fine-tune ใหม่, แอป webcam สำหรับเลือกทดสอบ Model A/B, สคริปต์ `webcam_to_jetson.py` สำหรับส่งภาพและ mask ไป Nano อีกเครื่อง, receiver และ ONNX Runtime GPU wheel ตามที่ระบุ

> **สถานะโมเดล:** ONNX FP16 export ผ่าน ONNX graph check และคง recurrent inputs/outputs แต่การเทียบ stateful 5 เฟรมบนเครื่องพัฒนายังเกิน parity gate max error 0.08 จึงยังไม่ถือว่าเป็น export ที่ผ่านเกณฑ์ deploy การทดสอบบน Jetson ใน bundle นี้มีไว้ตรวจการโหลดและพฤติกรรมจริงของโมเดลใหม่ โปรดเปรียบเทียบ mask/คุณภาพกับ PyTorch FP16 reference ก่อนนำไปใช้งานจริง

## เนื้อหา

- `models/model_a_svd50_fp16.onnx` — Model A: SVD50 + KD
- `models/model_b_slim50_svd50_fp16.onnx` — Model B: SLIM50 + SVD50 + KD
- `wheels/onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl` — wheel ที่รวมมาสำหรับ Python 3.8 / Linux aarch64
- `realtime_app.py` — แอป Streamlit สำหรับเลือกโมเดลและดู webcam สดบน Nano
- `webcam_to_jetson.py` — inference จาก webcam แล้วส่งภาพ JPEG + mask PNG ผ่าน TCP
- `stream_receiver.py` — รับ TCP จาก `webcam_to_jetson.py` บน Nano ปลายทาง
- `install_jetson.sh` — ติดตั้ง dependencies และ wheel
- `MODEL_STATUS_TH.md` — validation metrics และสถานะ ONNX parity ของโมเดล

โมเดลทั้งสองรับภาพ 256×256; recurrent states ถูกส่งต่อระหว่างเฟรม

## อุปกรณ์และซอฟต์แวร์

แพ็กเกจ ORT ในชุดนี้เป็น CPython 3.8 สำหรับ ARM64 เท่านั้น ใช้กับ Python 3.8 บน Jetson Nano ที่มี JetPack/CUDA/cuDNN/TensorRT dependencies เข้ากันได้ ต้องมี LAN ระหว่าง Nano หากใช้ streaming ไปเครื่องที่สอง ต้องมีกล้องที่ต่อกับเครื่องที่รัน realtime app หรือ sender

แตกไฟล์แล้วติดตั้ง:

```bash
tar -xf jetson_nano_modelA_modelB_fp16_bundle.tar
cd jetson_nano_modelA_modelB_fp16_bundle
bash install_jetson.sh
```

ตรวจ CUDA provider:

```bash
python3.8 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

ควรเห็น `CUDAExecutionProvider` ก่อนเลือก CUDA ในแอป ตัว wheel ไม่ได้รวม JetPack, CUDA, cuDNN หรือ TensorRT system libraries

## ทดสอบ webcam บน Nano เครื่องเดียว

```bash
python3.8 -m streamlit run realtime_app.py --server.address 0.0.0.0
```

เปิด URL ที่ Streamlit แสดงจาก browser ในเครือข่ายเดียวกัน เลือก Model A หรือ B, webcam, provider และ effect แล้วเปิด `Start realtime webcam` ปิดสวิตช์เพื่อหยุดกล้อง

## ส่งภาพและ mask ไป Nano เครื่องที่สอง

เริ่ม receiver ที่ Nano ปลายทางก่อน:

```bash
python3.8 stream_receiver.py --bind 0.0.0.0 --port 5000 --output received_inpainted.mp4
```

จาก Nano ที่มีกล้องและโมเดล:

```bash
python3.8 webcam_to_jetson.py --host <IP_NANOปลายทาง> --port 5000 --variant model_a_svd50 --provider cuda --preview
```

หรือเลือก Model B:

```bash
python3.8 webcam_to_jetson.py --host <IP_NANOปลายทาง> --port 5000 --variant model_b_slim50_svd50 --provider cuda
```

กด `q` ใน preview หรือ `Ctrl+C` เพื่อหยุด sender; receiver หยุดด้วย `q` เมื่อมี display หรือ `Ctrl+C` เมื่อรันแบบ `--no-display`

## ผล validation จากการ fine-tune

ใช้ validation split เดิม 266 ภาพ; test split ไม่ได้ใช้ในการเลือก checkpoint. PyTorch FP16 validation IoU คือ Model A `0.9421`, Model B `0.9322`. ตัวเลขนี้เป็น reference เท่านั้น ไม่ใช่ validation metric ของ ONNX บน Jetson. ดู `MODEL_STATUS_TH.md` สำหรับ parity caveat
