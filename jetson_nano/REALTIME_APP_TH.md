# แอป webcam แบบ realtime

แอปนี้ใช้ SLIM50_SVD50.onnx โดยตรง อ่านภาพจาก webcam ที่ต่อกับเครื่องซึ่งรัน Streamlit และแสดงภาพต้นฉบับคู่กับผล segmentation สด ใช้ webcam index 0–5 และเลือก Mask overlay, Binary mask, Green background หรือ Blur background ได้

## ติดตั้งและเริ่มใช้

ต้องมี Python, Streamlit, OpenCV, NumPy และ ONNX Runtime รุ่นที่ใช้ได้กับระบบของเครื่อง

    python -m pip install streamlit opencv-python numpy onnxruntime
    streamlit run jetson_nano/realtime_app.py

ถ้าเปิดแอปจากเครื่องเดียวกัน ให้ใช้ URL ที่ Streamlit แสดง (ปกติ http://localhost:8501) ถ้าเปิดผ่าน browser ในเครือข่ายเดียวกัน ต้องรัน Streamlit ให้รับการเชื่อมต่อจาก LAN และเปิดพอร์ต 8501

ในหน้าเว็บ เลือก webcam index, provider และ effect จากแถบด้านข้าง จากนั้นเปิดสวิตช์เริ่ม webcam การประมวลผลจะทำบนเครื่องที่รันแอป; ต่อ webcam เข้ากับเครื่องนั้นโดยตรง กดปิดสวิตช์เพื่อหยุดกล้อง

ค่า latency/FPS เป็นเวลาของ inference โมเดล ไม่รวมการรับภาพ webcam, วาด UI หรือ network display แอปนี้ยังไม่ได้ส่งเฟรมไป Nano 2

สำหรับ Jetson Nano ให้ใช้ ONNX Runtime ที่ build ให้ตรงกับ JetPack/CUDA ของเครื่อง; pip command ทั่วไปด้านบนเหมาะกับเครื่องพัฒนาและอาจใช้ไม่ได้บน Nano
