# ส่งผล segmentation จากเว็บแคมไป Jetson Nano

`webcam_to_jetson.py` ทำงานบนเครื่องต้นทางที่ต่อเว็บแคมและมี ONNX Runtime ติดตั้งอยู่: อ่านภาพ → รัน ONNX FP16 512×512 → สร้าง binary mask → ส่งภาพ JPEG กับ mask PNG ผ่าน TCP ไปยัง Jetson Nano ส่วน Nano ใช้ `stream_receiver.py` รับ packet, ทำ inpaint และแสดง/บันทึกผล

## 1. เปิด receiver บน Jetson Nano

ใน terminal บน Nano:

```bash
cd ~/Human-filter-for-video
python3 jetson_nano/stream_receiver.py --bind 0.0.0.0 --port 5000 --output received_inpainted.mp4
```

หา IP ของ Nano ด้วย `hostname -I` และให้เครื่องต้นทางกับ Nano เชื่อมต่อ LAN เดียวกัน หากไม่ต้องการแสดงหน้าต่างบน Nano ให้เพิ่ม `--no-display`.

## 2. เริ่มส่งจากเครื่องที่มีกล้องและโมเดล

ใน PowerShell ที่ root ของ repository เปลี่ยน `<NANO_IP>` เป็น IP จากขั้นก่อน:

```powershell
.\.venv\Scripts\python.exe .\jetson_nano\webcam_to_jetson.py --host <NANO_IP> --port 5000 --variant slim50_svd50 --provider cuda --preview
```

เลือก `--variant baseline`, `--variant svd50` หรือ `--variant slim50_svd50`; default คือ `slim50_svd50`. กด `q` ในหน้าต่าง preview หรือ Ctrl+C เพื่อหยุดส่ง หากเว็บแคมไม่ใช่กล้องหมายเลข 0 ให้ระบุ `--camera 1`. ใช้ `--model <path>` เพื่อระบุ ONNX ไฟล์เอง.

ไฟล์ 512×512 ที่มีให้เลือกอยู่ใน `compression/outputs/final_512/`. โดยปกติไม่ต้องคัดลอก ONNX ไป Nano เพราะ inference ทำบนเครื่องต้นทาง; Nano รับภาพและ mask ที่ประมวลผลแล้วทางเครือข่าย

Wire format เข้ากันได้กับ `stream_receiver.py`: 4-byte network-order header length, JSON header, JPEG frame และ PNG binary mask. Receiver ที่มีอยู่จะทำ inpaint จาก mask; ถ้าต้องการแสดง overlay/mask โดยไม่ลบพื้นหลัง ให้ใช้ decoder/renderer ฝั่ง Nano แทน receiver นี้.
