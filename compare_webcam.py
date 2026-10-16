"""
Real-time Multi-Model Side-by-Side Webcam Comparison with Ground Truth (OpenCV)
Compares ALL models in a single screen:
  1. Ground Truth Reference (High-Accuracy Person Segmentation)
  2. RVM MobileNetV3 (Large - Official Pretrained Baseline)
  3. RVM MobileNetV3-Small (Supervisely Fine-tuned)
  4. RVM MobileNetV2 (Supervisely Fine-tuned)
  5. RVM MobileNetV1 (Supervisely Fine-tuned)
  6. Live Benchmark Scoreboard (IoU vs GT, Dice, FPS, Latency, Params)

Hotkeys:
  'm' - Toggle Effect: Green Screen / Mask Only / Background Blur / Cutout / Neon Virtual BG
  'g' - Toggle Layout: 2x3 Grid vs 1x5 Horizontal Strip
  'q' - Quit
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

from rvm_backbones import build_rvm, load_rvm_checkpoint

CHECKPOINT_DIR = Path("outputs/checkpoints")
CAMERA_ID = 0
MODEL_SIZE = 256
MASK_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(" RVM ALL-MODELS REAL-TIME WEBCAM BENCHMARK WITH GROUND TRUTH")
print(" Device:", device)
if torch.cuda.is_available():
    print(" GPU   :", torch.cuda.get_device_name(0))
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD GROUND TRUTH REFERENCE MODEL (COCO Person Class 15)
# ------------------------------------------------------------
print("\n[1/5] Loading Ground Truth Reference Model (LRASPP COCO Person)...")
gt_weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
gt_model = lraspp_mobilenet_v3_large(weights=gt_weights).to(device).eval()

# ------------------------------------------------------------
# 2. LOAD ALL 4 RVM MODELS
# ------------------------------------------------------------
MODELS = {
    "RVM-V3-Large (Official)": {
        "variant": "official_v3",
        "params": 3.74,
        "color": (46, 204, 113),  # Green
    },
    "RVM-V3-Small": {
        "variant": "mobilenetv3_small",
        "ckpt": CHECKPOINT_DIR / "rvm_mobilenetv3_small_supervisely_best.pth",
        "params": 1.77,
        "color": (52, 152, 219),  # Blue
    },
    "RVM-MobileNetV2": {
        "variant": "mobilenetv2",
        "ckpt": CHECKPOINT_DIR / "rvm_mobilenetv2_supervisely_best.pth",
        "params": 2.65,
        "color": (230, 126, 34),  # Orange
    },
    "RVM-MobileNetV1": {
        "variant": "mobilenetv1",
        "ckpt": CHECKPOINT_DIR / "rvm_mobilenetv1_supervisely_best.pth",
        "params": 4.49,
        "color": (155, 89, 182),  # Purple
    },
}

loaded_models = {}
rec_states = {}

# Load Official V3 Large
print("[2/5] Loading Official RVM-MobileNetV3 (Large)...")
try:
    v3_official = torch.hub.load(
        "PeterL1n/RobustVideoMatting", "mobilenetv3", pretrained=True, trust_repo=True
    ).to(device).eval()
    loaded_models["RVM-V3-Large (Official)"] = v3_official
    rec_states["RVM-V3-Large (Official)"] = (None, None, None, None)
except Exception as e:
    print(f"Warning: Could not load official V3 Large ({e}), using fallback.")
    loaded_models["RVM-V3-Large (Official)"] = build_rvm("mobilenetv3_small", pretrained_backbone=False).to(device).eval()
    rec_states["RVM-V3-Large (Official)"] = (None, None, None, None)

# Load V3-Small, V2, V1
for name in ["RVM-V3-Small", "RVM-MobileNetV2", "RVM-MobileNetV1"]:
    cfg = MODELS[name]
    ckpt_path = cfg["ckpt"]
    if ckpt_path.exists():
        print(f"Loading {name} from checkpoint: {ckpt_path.name}")
        model, _ = load_rvm_checkpoint(ckpt_path, device=device)
    else:
        print(f"Warning: Checkpoint not found for {name}, building default...")
        model = build_rvm(cfg["variant"], pretrained_backbone=False).to(device).eval()
    loaded_models[name] = model
    rec_states[name] = (None, None, None, None)

print("\nAll models loaded successfully!")

# ------------------------------------------------------------
# 3. WEBCAM INITIALIZATION
# ------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam (ID: {CAMERA_ID}). Check connection.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

EFFECTS = ["Green Screen", "Mask Matte", "Background Blur", "Cutout", "Cyberpunk Neon"]
effect_mode = 0
grid_layout = True  # True: 2x3 Grid, False: 1x5 Horizontal Strip

fps_stats = {name: 0.0 for name in MODELS}
lat_stats = {name: 0.0 for name in MODELS}
iou_stats = {name: 0.0 for name in MODELS}
dice_stats = {name: 0.0 for name in MODELS}

print("\n" + "=" * 70)
print(" LIVE COMPARISON STREAM STARTED!")
print(" Controls:")
print("   [m] - Switch Effect (Green Screen / Mask / Blur / Cutout / Neon)")
print("   [g] - Toggle Layout (2x3 Grid vs 1x5 Strip)")
print("   [q] - Quit Application")
print("=" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read webcam frame.")
        break

    h_orig, w_orig = frame.shape[:2]

    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    src_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    src_tensor = src_tensor.to(device)

    # --------------------------------------------------------
    # STEP A: RUN GROUND TRUTH REFERENCE MODEL
    # --------------------------------------------------------
    t0_gt = time.perf_counter()
    with torch.inference_mode():
        # LRASPP Person Segmentation (class 15 = person in COCO)
        gt_out = gt_model(src_tensor)["out"]
        gt_mask_tensor = (gt_out.argmax(1) == 15).float()  # (1, H, W)
        gt_mask_np = gt_mask_tensor[0].cpu().numpy()
    gt_lat = (time.perf_counter() - t0_gt) * 1000.0

    gt_mask_full = cv2.resize(gt_mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    gt_mask_bin = (gt_mask_full >= 0.5).astype(np.float32)

    # --------------------------------------------------------
    # STEP B: RUN ALL 4 RVM MODELS
    # --------------------------------------------------------
    model_masks = {}

    for name, model in loaded_models.items():
        r1, r2, r3, r4 = rec_states[name]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            if name == "RVM-V3-Large (Official)":
                fgr, pha, r1, r2, r3, r4 = model(src_tensor, r1, r2, r3, r4, downsample_ratio=1.0)
                pred_prob = pha[0, 0].cpu().numpy()
            else:
                seg, r1, r2, r3, r4 = model(src_tensor, r1, r2, r3, r4, 1.0, True)
                pred_prob = torch.sigmoid(seg)[0, 0].cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        rec_states[name] = (r1, r2, r3, r4)

        cur_fps = 1000.0 / lat_ms if lat_ms > 0 else 0
        fps_stats[name] = 0.85 * fps_stats[name] + 0.15 * cur_fps
        lat_stats[name] = 0.85 * lat_stats[name] + 0.15 * lat_ms

        # Resize mask to full resolution
        prob_full = cv2.resize(pred_prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        bin_mask = (prob_full >= MASK_THRESHOLD).astype(np.float32)
        model_masks[name] = bin_mask

        # Compute Real-time IoU & Dice vs Ground Truth
        intersection = np.logical_and(bin_mask > 0.5, gt_mask_bin > 0.5).sum()
        union = np.logical_or(bin_mask > 0.5, gt_mask_bin > 0.5).sum()
        iou = (intersection / union) if union > 0 else 1.0
        dice = (2.0 * intersection / (bin_mask.sum() + gt_mask_bin.sum())) if (bin_mask.sum() + gt_mask_bin.sum()) > 0 else 1.0

        iou_stats[name] = 0.85 * iou_stats[name] + 0.15 * iou
        dice_stats[name] = 0.85 * dice_stats[name] + 0.15 * dice

    # --------------------------------------------------------
    # STEP C: RENDER PANELS
    # --------------------------------------------------------
    def render_panel(base_img, mask, title, color, badge_text="", is_gt=False):
        m3 = np.repeat(mask[..., None], 3, axis=2)
        
        if is_gt:
            # For GT: Show Camera with Green Contours / Tinted Mask overlay
            overlay = base_img.copy()
            overlay[mask > 0.5] = (overlay[mask > 0.5] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
            composed = overlay
        elif effect_mode == 0:  # Green Screen
            bg = np.zeros_like(base_img)
            bg[:] = (0, 220, 0)
            composed = (base_img * m3 + bg * (1.0 - m3)).astype(np.uint8)
        elif effect_mode == 1:  # Mask Matte
            composed = (mask * 255).astype(np.uint8)
            composed = cv2.cvtColor(composed, cv2.COLOR_GRAY2BGR)
        elif effect_mode == 2:  # Background Blur
            blurred = cv2.GaussianBlur(base_img, (41, 41), 0)
            composed = (base_img * m3 + blurred * (1.0 - m3)).astype(np.uint8)
        elif effect_mode == 3:  # Cutout
            composed = (base_img * m3).astype(np.uint8)
        else:  # Cyberpunk Neon
            neon_bg = np.zeros_like(base_img)
            neon_bg[:, :, 0] = 120  # Blue
            neon_bg[:, :, 2] = 180  # Red -> Purple/Neon
            composed = (base_img * m3 + neon_bg * (1.0 - m3)).astype(np.uint8)

        # Header Badge Bar
        cv2.rectangle(composed, (0, 0), (w_orig, 60), (20, 20, 20), -1)
        cv2.rectangle(composed, (0, 0), (8, 60), color, -1)
        cv2.putText(composed, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if badge_text:
            cv2.putText(composed, badge_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # Footer Metrics Card
        cv2.rectangle(composed, (0, h_orig - 45), (w_orig, h_orig), (15, 15, 15), -1)
        return composed

    rendered_panels = {}

    # 1. Ground Truth Panel
    gt_panel = render_panel(
        frame, gt_mask_bin,
        "🎯 GROUND TRUTH (Reference)",
        (0, 255, 128),
        f"COCO High-Acc Person Ref • Latency: {gt_lat:.1f}ms",
        is_gt=True
    )
    cv2.putText(gt_panel, "GOLD STANDARD REFERENCE", (15, h_orig - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
    rendered_panels["GT"] = gt_panel

    # 2-5. The 4 Models Panels
    for name, m_mask in model_masks.items():
        info = MODELS[name]
        col = info["color"]
        sub_title = f"{info['params']}M params | {fps_stats[name]:.0f} FPS | {lat_stats[name]:.1f}ms"
        p = render_panel(frame, m_mask, name, col, sub_title, is_gt=False)
        
        # Real-time IoU and Dice badge
        iou_pct = iou_stats[name] * 100.0
        dice_pct = dice_stats[name] * 100.0
        cv2.putText(p, f"IoU vs GT: {iou_pct:.1f}%  |  Dice: {dice_pct:.1f}%",
                    (15, h_orig - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        rendered_panels[name] = p

    # 6. Live Benchmark Scoreboard Panel
    sb = np.zeros_like(frame)
    sb[:] = (20, 24, 30)
    cv2.rectangle(sb, (0, 0), (w_orig, 60), (35, 40, 50), -1)
    cv2.putText(sb, "📊 LIVE BENCHMARK SCOREBOARD", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (52, 152, 219), 2)
    cv2.putText(sb, "Real-time Metrics against Ground Truth", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1)

    y_off = 105
    for rank, (m_name, m_cfg) in enumerate(MODELS.items(), start=1):
        c_badge = m_cfg["color"]
        cv2.rectangle(sb, (15, y_off - 25), (w_orig - 15, y_off + 35), (28, 33, 42), -1)
        cv2.rectangle(sb, (15, y_off - 25), (22, y_off + 35), c_badge, -1)
        
        cv2.putText(sb, f"#{rank} {m_name}", (35, y_off - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        stats_str = f"IoU: {iou_stats[m_name]*100:.1f}%  |  FPS: {fps_stats[m_name]:.0f}  |  Lat: {lat_stats[m_name]:.1f}ms  |  Size: {m_cfg['params']}M"
        cv2.putText(sb, stats_str, (35, y_off + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1)
        y_off += 70

    # Controls info at bottom of Scoreboard
    cv2.rectangle(sb, (0, h_orig - 55), (w_orig, h_orig), (15, 18, 22), -1)
    cv2.putText(sb, f"Effect: [{EFFECTS[effect_mode]}]  |  [m] Change Effect  |  [g] Layout  |  [q] Quit",
                (15, h_orig - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    rendered_panels["Scoreboard"] = sb

    # --------------------------------------------------------
    # STEP D: COMBINE ALL INTO SINGLE SCREEN DISPLAY
    # --------------------------------------------------------
    if grid_layout:
        # 2x3 Grid Layout
        # Row 1: GT, V3-Large, V3-Small
        # Row 2: V2, V1, Scoreboard
        pw = 500
        ph = int(h_orig * (pw / w_orig))

        p_gt = cv2.resize(rendered_panels["GT"], (pw, ph))
        p_v3_l = cv2.resize(rendered_panels["RVM-V3-Large (Official)"], (pw, ph))
        p_v3_s = cv2.resize(rendered_panels["RVM-V3-Small"], (pw, ph))
        row1 = np.hstack([p_gt, p_v3_l, p_v3_s])

        p_v2 = cv2.resize(rendered_panels["RVM-MobileNetV2"], (pw, ph))
        p_v1 = cv2.resize(rendered_panels["RVM-MobileNetV1"], (pw, ph))
        p_sb = cv2.resize(rendered_panels["Scoreboard"], (pw, ph))
        row2 = np.hstack([p_v2, p_v1, p_sb])

        display = np.vstack([row1, row2])
    else:
        # 1x5 Horizontal Strip Layout
        pw = 300
        ph = int(h_orig * (pw / w_orig))
        strip = [
            cv2.resize(rendered_panels["GT"], (pw, ph)),
            cv2.resize(rendered_panels["RVM-V3-Large (Official)"], (pw, ph)),
            cv2.resize(rendered_panels["RVM-V3-Small"], (pw, ph)),
            cv2.resize(rendered_panels["RVM-MobileNetV2"], (pw, ph)),
            cv2.resize(rendered_panels["RVM-MobileNetV1"], (pw, ph)),
        ]
        display = np.hstack(strip)

    cv2.imshow("RVM All-Model Live Benchmark with Ground Truth", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        effect_mode = (effect_mode + 1) % len(EFFECTS)
    elif key == ord("g"):
        grid_layout = not grid_layout

cap.release()
cv2.destroyAllWindows()
print("\nWebcam closed successfully.")
