press# ติดตั้ง Local RVM Compression ลง Repository เดิม

> นี่เป็น Patch สำหรับเพิ่มไฟล์ ไม่ได้เปลี่ยนโค้ดหรือ Notebook Midterm เดิม และยังไม่ได้ push ขึ้น GitHub

1. เปิด PowerShell ไปที่โฟลเดอร์ Repository เดิม `Human-filter-for-video`.
2. แตกไฟล์ ZIP นี้ลง **root ของ Repository เดิม** โดยโฟลเดอร์ `compression` และ `tests` จะปรากฏขึ้น
3. ตรวจว่าไฟล์ `outputs/checkpoints/rvm_mobilenetv3_supervisely_best.pth` และ `data/supervisely/images`, `data/supervisely/masks` มีอยู่ (หรือระบุ path ผ่าน options)
4. ติดตั้ง dependencies ตาม `compression/README.md` และ clone RVM Source ลง `third_party/RobustVideoMatting`

```powershell
# รันที่ root ของ Repository เดิม
git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
python -m pip install numpy pillow pytest
python -m compression.doctor --rvm-source third_party/RobustVideoMatting --device cuda
python -m compression.search --profile compression/configs/quick.json --rvm-source third_party/RobustVideoMatting --device cuda --epochs 8
```

Optional Quantization (หลังเทรนและตรวจ Baseline แล้ว):

```powershell
python -m pip install -r compression/requirements-quant.txt
python -m compression.quantization --run-id C0 --format both --rvm-source third_party/RobustVideoMatting
python -m compression.quantization --run-id SLIM50 --format both --rvm-source third_party/RobustVideoMatting
```

Final Test เรียกเมื่อเลือก Candidate ผ่าน Validation แล้วเท่านั้น:

```powershell
python -m compression.finalize --confirm-final --rvm-source third_party/RobustVideoMatting
```

**ข้อควรระวัง:** Test IoU ตาม README เดิม 93.54% ไม่ควรนำมาหัก 10 จุดแล้วใช้คัดเลือกโมเดลบน Test Set แผนนี้วัด Baseline Validation จาก Split เดียวกันก่อน แล้วใช้ `Val IoU - 10 percentage points` เป็นเพดานในการค้นหา หลังเลือกโมเดลจึงตรวจ Test ครั้งเดียว

หากพาธของข้อมูลหรือ Checkpoint ต่างจากนี้ ให้เติม `--checkpoint "..." --data-dir "..."` ลงในทุกคำสั่ง และหากต้องการเทียบผลเดิมโดยตรง ให้ใช้ Split Manifest ของการทดลอง Midterm ผ่าน `--split-manifest "..."`

ห้ามเพิ่ม `third_party/RobustVideoMatting`, `compression/outputs`, Dataset หรือ Checkpoints ขนาดใหญ่เข้า Git โดยไม่ตรวจ License/ขนาดไฟล์ก่อน
