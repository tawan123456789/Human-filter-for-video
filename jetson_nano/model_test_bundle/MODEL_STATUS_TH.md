# สถานะโมเดลใน bundle

โมเดลทั้งคู่ fine-tune ใหม่ด้วย train-only Gaussian blur และ Gaussian noise; validation/test ไม่ถูก augment

| Model | PyTorch FP16 validation IoU / Dice | ONNX FP16 ไฟล์ | ONNX parity |
|---|---:|---|---|
| A — SVD50 + KD | 0.9421 / 0.9692 | `models/model_a_svd50_fp16.onnx` | เกินเกณฑ์ max error 0.08 บนเครื่องพัฒนา |
| B — SLIM50 + SVD50 + KD | 0.9322 / 0.9635 | `models/model_b_slim50_svd50_fp16.onnx` | เกินเกณฑ์ max error 0.08 บนเครื่องพัฒนา |

ไฟล์ ONNX ถูกสร้างเป็น opset 16 และผ่าน ONNX graph checker แต่ ONNX Runtime parity validation ไม่ผ่าน จึงรวมมาเพื่อทดสอบบน Nano ตามคำขอเท่านั้น ยังไม่ควรถือว่าเทียบเท่ากับ PyTorch FP16 checkpoints หรือพร้อม deploy production ตัวเลข PyTorch เป็น validation split 266 ภาพบนเครื่องพัฒนา ไม่ใช่ผล Jetson

ผลจากการรันบน Nano (provider, latency/FPS และคุณภาพ mask) ต้องบันทึกแยกหลังทดสอบบนอุปกรณ์จริง
