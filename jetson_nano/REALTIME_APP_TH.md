# แอป webcam แบบ realtime

แอปนี้รองรับโมเดลตระกูล RVM ทั้งหมด โดยเฉพาะโมเดล Fine-tuned ที่ให้ประสิทธิภาพและความแม่นยำสูงกว่าโมเดลที่ผ่านการกลั่น (Distilled Model):
- **RVM-MobileNetV1** (GPU FPS 185.6, IoU 93.18%) — แนะนำสำหรับ Jetson Nano CUDA
- **RVM-MobileNetV2** (134.7 FPS, 9.67 MB, IoU 92.69%) — โมเดล Embedded Edge สมดุล
- **RVM-MobileNetV3-Large** (IoU 93.54%) — คุณภาพสูงสุด
- **RVM-MobileNetV3-Small** (CPU FPS 33.9, 6.39 MB, IoU 91.28%) — สำหรับ CPU-only
- **SLIM50_SVD50** (IoU 91.16%) — โมเดลกลั่นตั้งต้น

สามารถเลือกโมเดล, webcam index 0–5, provider (Auto/CUDA/CPU) และ effect (Mask overlay, Binary mask, Green background, Blur background, Cyberpunk Neon, Black Cutout) ได้สดๆ


## ติดตั้งและเริ่มใช้

ต้องมี Python, Streamlit, OpenCV, NumPy และ ONNX Runtime รุ่นที่ใช้ได้กับระบบของเครื่อง

    python -m pip install streamlit opencv-python numpy onnxruntime
    streamlit run jetson_nano/realtime_app.py

ถ้าเปิดแอปจากเครื่องเดียวกัน ให้ใช้ URL ที่ Streamlit แสดง (ปกติ http://localhost:8501) ถ้าเปิดผ่าน browser ในเครือข่ายเดียวกัน ต้องรัน Streamlit ให้รับการเชื่อมต่อจาก LAN และเปิดพอร์ต 8501

ในหน้าเว็บ เลือก webcam index, provider และ effect จากแถบด้านข้าง จากนั้นเปิดสวิตช์เริ่ม webcam การประมวลผลจะทำบนเครื่องที่รันแอป; ต่อ webcam เข้ากับเครื่องนั้นโดยตรง กดปิดสวิตช์เพื่อหยุดกล้อง

ค่า latency/FPS เป็นเวลาของ inference โมเดล ไม่รวมการรับภาพ webcam, วาด UI หรือ network display แอปนี้ยังไม่ได้ส่งเฟรมไป Nano 2

สำหรับ Jetson Nano ให้ใช้ ONNX Runtime ที่ build ให้ตรงกับ JetPack/CUDA ของเครื่อง; pip command ทั่วไปด้านบนเหมาะกับเครื่องพัฒนาและอาจใช้ไม่ได้บน Nano
