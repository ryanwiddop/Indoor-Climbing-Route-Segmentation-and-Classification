import os
import sys
import json
import logging
import time
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ..datasets.tiled_wall_dataset import TiledWallDataset, collate_fn
except ImportError:
    from datasets.tiled_wall_dataset import TiledWallDataset, collate_fn

TILED_IMG_PATH = "/home/public/rwiddop/tiled/tiles/"
TILED_ANN_PATH = "/home/public/rwiddop/tiled/tiles.csv"
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs/phase_1.log")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints/phase_1.pt")
FIGURES_PATH = os.path.join(os.path.dirname(__file__), "figures/phase_1/train")


def ensure_output_dirs():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)


ensure_output_dirs()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


def log_script_start():
    separator = "=" * 80
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "\n%s\nPHASE 1 SCRIPT STARTED | %s | pid=%s\n%s",
        separator,
        started_at,
        os.getpid(),
        separator,
    )
    

def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}
    

def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=240,
        max_size=800,
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def box_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    boxA_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    boxB_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    
    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def load_model(checkpoint_path, device, max_detections=500):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hold_type_to_idx = ckpt["hold_type_to_idx"]
    num_classes = ckpt["num_classes"]

    model = build_model(num_classes)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.roi_heads.detections_per_img = max_detections

    model.to(device).eval()
    return model, hold_type_to_idx, num_classes


def _iter_tiles(H, W, tile, overlap):
    stride = max(1, int(round(tile * (1.0 - overlap))))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    ys = sorted(set(ys))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    xs = sorted(set(xs))
    for y0 in ys:
        y1 = min(y0 + tile, H)
        for x0 in xs:
            x1 = min(x0 + tile, W)
            yield y0, y1, x0, x1


def _nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        bi = boxes[i]
        rest = boxes[order]
        x1 = np.maximum(bi[0], rest[:, 0])
        y1 = np.maximum(bi[1], rest[:, 1])
        x2 = np.minimum(bi[2], rest[:, 2])
        y2 = np.minimum(bi[3], rest[:, 3])
        iw = np.clip(x2 - x1, 0, None)
        ih = np.clip(y2 - y1, 0, None)
        inter = iw * ih
        a_i = max(0, bi[2] - bi[0]) * max(0, bi[3] - bi[1])
        a_r = np.clip(rest[:, 2] - rest[:, 0], 0, None) * \
              np.clip(rest[:, 3] - rest[:, 1], 0, None)
        union = a_i + a_r - inter
        ious = np.where(union > 0, inter / union, 0.0)
        order = [order[k] for k in range(len(order)) if ious[k] < iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _crop_mask_to_bbox(mask, box, img_h, img_w):
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(img_w, int(np.ceil(box[2])))
    y2 = min(img_h, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, np.zeros((1, 1), dtype=bool)
    return x1, y1, mask[y1:y2, x1:x2].copy()


def tiled_predict(model, image_tensor_cpu, device, tile_size=800, overlap=0.25, score_threshold=0.05, nms_iou=0.5):
    C, H, W = image_tensor_cpu.shape
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []

    for (y0, y1, x0, x1) in _iter_tiles(H, W, tile_size, overlap):
        tile = image_tensor_cpu[:, y0:y1, x0:x1].to(device)
        with torch.no_grad():
            out = model([tile])[0]

        pb = out["boxes"].detach().cpu().numpy()
        ps = out["scores"].detach().cpu().numpy()
        pl = out["labels"].detach().cpu().numpy()
        pm = (out["masks"].detach().cpu().numpy()[:, 0] > 0.5)
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        keep = ps >= score_threshold
        if not keep.any():
            continue
        pb = pb[keep]; ps = ps[keep]; pl = pl[keep]; pm = pm[keep]

        tile_h = y1 - y0
        tile_w = x1 - x0
        for b, s, l, m in zip(pb, ps, pl, pm):
            lx, ly, local = _crop_mask_to_bbox(m, b, tile_h, tile_w)
            wall_box = np.array([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], dtype=np.float32)
            all_boxes.append(wall_box)
            all_scores.append(float(s))
            all_labels.append(int(l))
            all_masks.append((lx + x0, ly + y0, local))

    if not all_boxes:
        return {
            "boxes":  np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,),   dtype=np.float32),
            "labels": np.zeros((0,),   dtype=np.int64),
            "masks":  []
        }

    all_boxes  = np.stack(all_boxes, axis=0)
    all_scores = np.asarray(all_scores, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int64)

    keep_idx = _nms_numpy(all_boxes, all_scores, nms_iou)

    return {
        "boxes":  all_boxes[keep_idx],
        "scores": all_scores[keep_idx],
        "labels": all_labels[keep_idx],
        "masks":  [all_masks[i] for i in keep_idx]
    }


def visualize_predictions(image_pil, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, idx_to_hold, score_threshold=0.3, output_name="prediction.png"):
    result = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        grade = idx_to_hold.get(int(label), "BG")
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
        draw.text((x1, max(0, y1 - 12)), f"GT: {grade}", fill="green")

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        grade = idx_to_hold.get(int(label), "BG")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, min(y2 + 2, result.height - 12)), f"Pred: {grade} ({score:.2f})", fill="red")

    plt.figure(figsize=(14, 10))
    plt.imshow(result)
    plt.title("Green = GT, Red = Predicted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    mp.set_start_method("spawn", force=True)
    
    hold_dataset = TiledWallDataset(TILED_IMG_PATH, TILED_ANN_PATH)

    dataset_size = len(hold_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_base = TiledWallDataset(TILED_IMG_PATH, TILED_ANN_PATH, augment=True)
    val_base   = TiledWallDataset(TILED_IMG_PATH, TILED_ANN_PATH, augment=False)
    train_dataset = torch.utils.data.Subset(train_base, train_indices)
    val_dataset   = torch.utils.data.Subset(val_base,   val_indices)
    logger.info(f"Dataset size: {dataset_size}, Train size: {train_size}, Val size: {val_size}")

    
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    NUM_EPOCHS = 45

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    num_classes = len(hold_dataset.hold_type_to_idx) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"GPU compute capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"Grades: {hold_dataset.hold_type_to_idx}")
    
    model = build_model(num_classes).to(device)
    # model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 40], gamma=0.1
    )

    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    
    best_train_loss = float("inf")
    best_loss_epoch = -1
    
    start_time = time.perf_counter()
    last_epoch_time = start_time
    
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        running_count = 0
        
        with logging_redirect_tqdm(loggers=[logger]):
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                unit="batch",
                leave=True,
                dynamic_ncols=True
            )
            for images, targets in pbar:
                t_batch_got = time.perf_counter()
                if not images or not targets:
                    continue
                images  = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += losses.item()
                running_count += 1
                
                pbar.set_postfix({
                    "loss":     f"{losses.item():.4f}",
                    "avg":      f"{train_loss / running_count:.4f}",
                    "lr":       f"{optimizer.param_groups[0]['lr']:.1e}",
                    "cls":      f"{loss_dict.get('loss_classifier', 0):.3f}",
                    "box":      f"{loss_dict.get('loss_box_reg', 0):.3f}",
                    "mask":     f"{loss_dict.get('loss_mask', 0):.3f}",
                })
            
            lr_scheduler.step()
            
            avg_train_loss = train_loss / max(running_count, 1)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_loss_epoch = epoch + 1
                
            loss_history.append(avg_train_loss)
            
            epoch_time = time.perf_counter() - last_epoch_time
            # logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Epoch Time: {epoch_time:.2f} seconds")
            last_epoch_time = time.perf_counter()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best Train Loss: {best_train_loss:.4f} at epoch {best_loss_epoch}")
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": NUM_EPOCHS,
        "train_loss": avg_train_loss,
        "hold_type_to_idx": hold_dataset.hold_type_to_idx,
        "num_classes": num_classes,
        "loss_history": loss_history
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    logger.info(f"Model checkpoint saved to {CHECKPOINT_PATH}")
    
    model.eval()
    iou_threshold = 0.5
    no_detection_label = num_classes

    all_raw = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                all_raw.append((
                    target["boxes"].cpu().numpy(),
                    target["labels"].cpu().numpy(),
                    output["boxes"].cpu().numpy(),
                    output["labels"].cpu().numpy(),
                    output["scores"].cpu().numpy(),
                ))
    logger.info(f"Validation inference completed in {time.perf_counter() - start_time:.2f} seconds")

    def score_threshold(threshold):
        all_true, all_pred = [], []
        for gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores in all_raw:
            matched_gt = set()
            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                if ps < threshold:
                    continue
                best_iou, best_gt_idx = 0.0, -1
                for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt:
                        continue
                    iou = box_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, idx
                if best_iou >= iou_threshold and best_gt_idx != -1 and pl == gt_labels[best_gt_idx]:
                    all_true.append(pl)
                    all_pred.append(pl)
                    matched_gt.add(best_gt_idx)
                else:
                    all_true.append(no_detection_label)
                    all_pred.append(pl)
            for idx, gl in enumerate(gt_labels):
                if idx not in matched_gt:
                    all_true.append(gl)
                    all_pred.append(no_detection_label)
        return all_true, all_pred

    eval_labels = list(range(num_classes)) + [no_detection_label]
    best_f1, best_threshold, best_true, best_pred = 0.0, 0.3, [], []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        t, p = score_threshold(thresh)
        cm = confusion_matrix(t, p, labels=eval_labels)
        tp = np.trace(cm) - cm[eval_labels.index(0), eval_labels.index(0)]
        precision = tp / max(1, sum(cm[:, 1]))
        recall = tp / max(1, sum(cm[1, :]))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        acc = np.trace(cm) / np.sum(cm)
        logger.info(f"  thresh={thresh:.1f}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  acc={acc:.4f}")
        if f1 > best_f1:
            best_f1, best_threshold, best_true, best_pred = f1, thresh, t, p

    logger.info(f"Best threshold: {best_threshold} (F1={best_f1:.3f})")
    all_true, all_pred = best_true, best_pred

    report = classification_report(all_true, all_pred, labels=eval_labels, zero_division=0)
    conf_matrix = confusion_matrix(all_true, all_pred, labels=eval_labels)
    acc = np.trace(conf_matrix) / np.sum(conf_matrix)
    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Validation Accuracy: {acc:.4f}")
    
    # visualize confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(eval_labels))
    tick_names = []
    for lbl in eval_labels:
        if lbl == 0:
            tick_names.append("BG")
        elif lbl == no_detection_label:
            tick_names.append("No Det")
        else:
            tick_names.append(hold_dataset.idx_to_hold_type.get(lbl, str(lbl)))

    plt.xticks(tick_marks, tick_names, rotation=45, ha="right")
    plt.yticks(tick_marks, tick_names)

    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black",
                     fontsize=8)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + "/phase_1_cm.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {FIGURES_PATH}/phase_1_cm.png")
    
    # visualize loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, marker="o")
    plt.title("Training Loss Curve (Avg.)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.xticks(range(1, NUM_EPOCHS + 1, max(1, NUM_EPOCHS // 15)))
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + "/phase_1_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curve to {FIGURES_PATH}/phase_1_loss_curve.png")
    
    # visualize some predictions
    model.eval()
    val_iter = iter(val_loader)
    with torch.no_grad():
        for i in range(min(5, len(val_loader))):
            images, targets = next(val_iter)
            images = [img.to(device) for img in images]
            outputs = model(images)

            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

            gt_boxes = targets[0]["boxes"].cpu().numpy()
            gt_labels = targets[0]["labels"].cpu().numpy()

            pred_boxes = outputs[0]["boxes"].detach().cpu().numpy()
            pred_labels = outputs[0]["labels"].detach().cpu().numpy()
            pred_scores = outputs[0]["scores"].detach().cpu().numpy()

            visualize_predictions(
                image_pil,
                gt_boxes, gt_labels,
                pred_boxes, pred_labels, pred_scores,
                hold_dataset.idx_to_hold_type,
                score_threshold=0.6,
                output_name=f"{FIGURES_PATH}/prediction_{i+1}.png"
            )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    log_script_start()
    main()