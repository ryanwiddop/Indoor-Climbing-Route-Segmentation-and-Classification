import os
import sys
import logging
import cv2
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.phase_1 import load_model, tiled_predict

DEFAULT_CKPT = os.path.join(REPO_ROOT, "model", "checkpoints", "phase_1.pt")
LOG_FILE_PATH = os.path.join(HERE, "logs/color_bin.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


_LAB_CENTROIDS_STD = [
    ("red",   ( 42,  40,  38)),
    ("orange",   ( 62,  62,  90)),
    ("yellow",   ( 82,  11,  68)),
    ("green",   ( 52, -38,  20)),
    ("blue",   ( 45,   3, -43)),
    ("purple",   ( 50,  26, -40)),
    ("pink",   ( 62,  59,  -3)),
    ("white",  ( 88,   2,   4)),
    ("black",  ( 10,   0,   0)),
]

_LAB_CENTROIDS = [
    (name, (int(L * 255 / 100), int(a + 128), int(b + 128)))
    for name, (L, a, b) in _LAB_CENTROIDS_STD
]

_LAB_CHROMATIC_CENTROIDS = [
    (name, ab) for name, ab in _LAB_CENTROIDS if name not in ("white", "black")
]


def _nearest_lab_chromatic(a, b):
    def _dist(c):
        _, ca, cb = c
        return (a - ca) ** 2 + (b - cb) ** 2
    return min(_LAB_CHROMATIC_CENTROIDS, key=lambda nc: _dist(nc[1]))[0]


def classify_lab(L, a, b):
    if len(L) < 5:
        return "unknown", None

    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    chroma_p75 = float(np.percentile(chroma, 75))
    chroma_p95 = float(np.percentile(chroma, 95))
    is_chromatic = (chroma_p75 >= 12.0) or (chroma_p95 >= 25.0)

    if not is_chromatic:
        med_L_all = float(np.median(L))
        color = "white" if med_L_all > 130 else "black"
        return color, {
            "med_L": med_L_all, "med_a": float(np.median(a)), "med_b": float(np.median(b)),
            "p75": chroma_p75, "p95": chroma_p95, "achromatic": True,
        }

    chromatic = chroma >= chroma_p75
    med_L = float(np.median(L[chromatic]))
    med_a = float(np.median(a[chromatic]))
    med_b = float(np.median(b[chromatic]))
    color = _nearest_lab_chromatic(med_a, med_b)
    return color, {
        "med_L": med_L, "med_a": med_a, "med_b": med_b,
        "p75": chroma_p75, "p95": chroma_p95, "achromatic": False,
    }


def _name_color_lab(bgr_image, mask, idx):
    lab_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    mask_bool = mask > 0
    L = lab_img[:, :, 0][mask_bool].astype(np.float32)
    a = lab_img[:, :, 1][mask_bool].astype(np.float32)
    b = lab_img[:, :, 2][mask_bool].astype(np.float32)

    color, stats = classify_lab(L, a, b)
    if stats is None:
        return color
    if stats["achromatic"]:
        print(f"[lab]      #{idx:3d}  achromatic (p75={stats["p75"]:.1f}, p95={stats["p95"]:.1f}, L={stats["med_L"]:.0f})  -> {color}")
    else:
        print(f"[lab]      #{idx:3d}  med_Lab=({stats["med_L"]:.0f},{stats["med_a"]:.0f},{stats["med_b"]:.0f})  p75={stats["p75"]:.1f} p95={stats["p95"]:.1f}  -> {color}")
    return color


def color_bin_lab(image, model, device="cuda", tile_size=800, overlap=0.25, score_threshold=0.3, nms_iou=0.5):
    if image is None:
        raise ValueError("Input image is None. Check the image path and file readability.")

    H, W = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    pred = tiled_predict(
        model,
        image_tensor,
        device,
        tile_size=tile_size,
        overlap=overlap,
        score_threshold=score_threshold,
        nms_iou=nms_iou
    )

    masks_to_colors = defaultdict(list)
    for idx, (x_off, y_off, local_mask) in enumerate(pred["masks"]):
        full_mask = np.zeros((H, W), dtype=np.uint8)
        mh, mw = local_mask.shape
        x2 = min(W, x_off + mw)
        y2 = min(H, y_off + mh)
        full_mask[y_off:y2, x_off:x2] = local_mask[:y2 - y_off, :x2 - x_off].astype(np.uint8) * 255
        name = _name_color_lab(image, full_mask, idx)
        masks_to_colors[name].append((idx, full_mask))

    return masks_to_colors
        
    
def visualize_bins(image, masks_to_colors):
    color_map = {
        "red":    (0,   0,   255),
        "orange": (0,   165, 255),
        "yellow": (0,   255, 255),
        "green":  (34,  139, 34),
        "blue":   (255, 0,   0),
        "purple": (128, 0,   128),
        "pink":   (147, 20,  255),
        "white":  (255, 255, 255),
        "black":  (30,  30,  30),
    }
    
    output_image = image.copy()
    for color_name, entries in masks_to_colors.items():
        color = color_map.get(color_name, (255, 255, 255))
        label_color = (0, 0, 0) if color_name in ("white", "yellow") else (255, 255, 255)
        for idx, mask in entries:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, color, thickness=cv2.FILLED)
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(output_image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    return output_image
        
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(DEFAULT_CKPT, device)
    image_path = "/home/public/rwiddop/images/00.jpg"
    image = cv2.imread(image_path)

    output_dir = os.path.join(HERE, "output")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    masks_lab = color_bin_lab(image, model, device=device)
    out_lab = visualize_bins(image, masks_lab)
    cv2.imwrite(os.path.join(output_dir, base + "_lab.png"), out_lab)
    print(f"Saved -> {os.path.join(output_dir, base + "_lab.png")}")