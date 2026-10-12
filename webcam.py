import cv2
import torch
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = (
    "outputs/checkpoints/"
    "rvm_mobilenetv3_supervisely_best.pth"
)

CAMERA_ID = 0

# ตอน train เราใช้ 256x256
MODEL_SIZE = 256

# Threshold เดียวกับตอน evaluate
MASK_THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("RVM MobileNetV3 - Fine-tuned Human Segmentation")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD RVM ARCHITECTURE
# ============================================================

print("\nLoading RVM-MobileNetV3 architecture...")

model = torch.hub.load(
    "PeterL1n/RobustVideoMatting",
    "mobilenetv3",
    pretrained=False,
    trust_repo=True
)


# ============================================================
# LOAD OUR FINE-TUNED CHECKPOINT
# ============================================================

print("Loading checkpoint:")
print(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

# checkpoint ที่เราสร้างใน notebook
# มี model_state_dict อยู่ข้างใน
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# แสดงข้อมูล checkpoint
if "epoch" in checkpoint:
    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

if "val_metrics" in checkpoint:
    print(
        "Validation metrics:"
    )

    for key, value in (
        checkpoint["val_metrics"].items()
    ):
        print(
            f"  {key}: {value:.4f}"
        )

print("\nFine-tuned model loaded successfully.")


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    # fallback ถ้า CAP_DSHOW ใช้ไม่ได้
    cap = cv2.VideoCapture(
        CAMERA_ID
    )


if not cap.isOpened():

    raise RuntimeError(
        "Cannot open webcam."
    )


# ตั้ง resolution ของ webcam
# ไม่เกี่ยวกับ model input โดยตรง
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\nWebcam started.")
print("Press Q to quit.")


# ============================================================
# RECURRENT STATES
# ============================================================

# RVM มี temporal memory
r1 = None
r2 = None
r3 = None
r4 = None


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break


    # ----------------------------------------
    # Original webcam size
    # ----------------------------------------

    original_height, original_width = (
        frame.shape[:2]
    )


    # ----------------------------------------
    # BGR -> RGB
    # ----------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------
    # Resize เหมือน training
    #
    # Webcam:
    # H x W
    #
    # ->
    #
    # 256 x 256
    # ----------------------------------------

    rgb_resized = cv2.resize(
        rgb,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR
    )


    # ----------------------------------------
    # numpy -> torch
    #
    # HWC
    # ->
    # CHW
    # ->
    # BCHW
    #
    # uint8 0-255
    # ->
    # float 0-1
    # ----------------------------------------

    src = torch.from_numpy(
        rgb_resized
    )

    src = src.permute(
        2, 0, 1
    )

    src = src.unsqueeze(0)

    src = src.float() / 255.0

    src = src.to(device)


    # ----------------------------------------
    # FINE-TUNED SEGMENTATION
    # ----------------------------------------

    with torch.inference_mode():

        pred_seg, r1, r2, r3, r4 = model(
            src,
            r1,
            r2,
            r3,
            r4,
            1.0,
            True
        )


    # ----------------------------------------
    # Logits -> Probability
    # ----------------------------------------

    pred_prob = torch.sigmoid(
        pred_seg
    )


    # ----------------------------------------
    # Probability -> Binary Mask
    # ----------------------------------------

    pred_mask = (
        pred_prob >= MASK_THRESHOLD
    ).float()


    # ----------------------------------------
    # Tensor -> numpy
    #
    # [1,1,H,W]
    # ->
    # [H,W]
    # ----------------------------------------

    mask_np = (
        pred_mask[
            0, 0
        ]
        .detach()
        .cpu()
        .numpy()
    )


    # ----------------------------------------
    # 0 / 1
    # ->
    # 0 / 255
    # ----------------------------------------

    mask_np = (
        mask_np * 255
    ).astype(
        np.uint8
    )


    # ----------------------------------------
    # Resize mask กลับเป็นขนาด webcam
    #
    # Binary mask ต้องใช้ NEAREST
    # ----------------------------------------

    mask_display = cv2.resize(
        mask_np,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # ----------------------------------------
    # SHOW ONLY MASK
    # ----------------------------------------

    cv2.imshow(
        "Fine-tuned RVM - Person Mask",
        mask_display
    )


    # ----------------------------------------
    # Quit
    # ----------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Webcam closed.")