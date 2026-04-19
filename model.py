import os
import json
import time
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

RT_ID_COLOR_MAP = {
    -1: "yellow",
    0: "red",
    1: "blue",
    2: "green",
    3: "purple",
    4: "orange",
    5: "cyan",
    6: "magenta",
    7: "brown",
    8: "pink",
    9: "gray",
    10: "black",
    11: "lime",
    12: "teal",
    13: "navy",
    14: "maroon",
    15: "olive",
    16: "coral"
}

GD_TEST_IMG_PATH = "images"
GD_TEST_ANN_PATH = "annotation.csv"


class GradeDataset(Dataset):
    def __init__(self, img_dir, ann_csv, drop_incomplete=True, drop_volume=False):
        self.img_dir = img_dir
        self.ann_csv = ann_csv
        self.drop_incomplete = drop_incomplete
        self.drop_volume = drop_volume

        # gathering all img names and reading annotations
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.annotations = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)
        
        # dict grouping attributes by img
        self.rows_by_img = defaultdict(list)
        for _, row in self.annotations.iterrows():
            self.rows_by_img[row["filename"]].append(row)
            
        # collate all grades in dataset
        grades = []
        for _, row in self.annotations.iterrows():
            attr = self._json(row["region_attributes"])
            g = attr.get("route_grade", "")
            if g != "" and g is not None:
                grades.append(g.strip())
        unique_grades = sorted(set(grades))
        
        # indexing grades
        self.grade_to_idx = {grade: idx + 1 for idx, grade in enumerate(unique_grades)}
        self.idx_to_grade = {idx: grade for grade, idx in self.grade_to_idx.items()}
        
        if len(self.grade_to_idx) == 0:
            raise ValueError("No valid grades found in dataset.")
        
    def __len__(self):
        return len(self.img_files)
    
    def _json(self, s):
        if not isinstance(s, str) or s.strip() in ("", "[]"):
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}
        
    def _polygon_to_mask(self, poly_xy, h, w):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(poly_xy, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)
    
    def _parse_bool(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower in ("true", "1", "yes"):
                return True
            elif val_lower in ("false", "0", "no"):
                return False
        return False
    
    # filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
    # 00.jpg,8294565,"{}",188,0,"{"all_points_x": [2406, 2416, 2421, 2422, 2415, 2406, 2395, 2391, 2389, 2395, 2402], "all_points_y": [1670, 1667, 1659, 1648, 1639, 1637, 1640, 1648, 1658, 1666, 1671], "name": "polygon"}","{"incomplete_route": false, "is_volume": false, "route_grade": "VB", "route_id": 1}"
    def __getitem__(self, index):
        img_name = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_name)
        
        # EXIF orientation
        image = ImageOps.exif_transpose(Image.open(img_path))
        w, h = image.size
        rows = self.rows_by_img.get(img_name, [])
        
        # group polygons and grades by route id
        grouped = defaultdict(list)
        for row in rows:
            shape = self._json(row["region_shape_attributes"])
            attr = self._json(row["region_attributes"])
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys or len(xs) != len(ys):
                continue
            
            route_id_raw = attr.get("route_id", "")
            if route_id_raw in ("", None):
                continue
            
            try:
                route_id = int(route_id_raw)
            except (ValueError, TypeError):
                continue
            
            grade = attr.get("route_grade", "")
            if grade in ("", None):
                continue
            else:
                grade = grade.strip()
            
            incomplete_route = self._parse_bool(attr.get("incomplete_route", False))
            if self.drop_incomplete and incomplete_route:
                continue
            
            is_volume = self._parse_bool(attr.get("is_volume", False))
            if self.drop_volume and is_volume:
                continue
            
            polygon = list(zip(xs, ys))
            grouped[route_id].append({
                "polygon": polygon,
                "grade": grade,
                # "is_volume": is_volume,
            })
        
        masks = []
        boxes = []
        labels = []
        areas = []
        
        # construct instance masks
        for route_id, items in grouped.items():
            route_mask = np.zeros((h, w), dtype=np.uint8)
            grade_votes = []
            
            # check for inconsistent grades within same route id (incorrect preproccessing backstop)
            for item in items:
                poly_mask = self._polygon_to_mask(item["polygon"], h, w)
                route_mask = np.maximum(route_mask, poly_mask)
                grade_votes.append(item["grade"])
            final_grade = Counter(grade_votes).most_common(1)[0][0]
            if final_grade not in self.grade_to_idx:
                continue
  
            # bounding box
            ys, xs = np.where(route_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            if x_max <= x_min or y_max <= y_min:
                continue
            
            masks.append(route_mask)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.grade_to_idx[final_grade])
            areas.append(float((route_mask > 0).sum()))
            
        image_tensor = F.to_tensor(image)
        
        # if no routes return empty else return found
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([index], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            masks_np = np.stack(masks, axis=0).astype(np.uint8, copy=False)
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.from_numpy(masks_np),
                "image_id": torch.tensor([index], dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
            }
            
        return image_tensor, target


def visualize_gd_sample(image, target):
    image = ImageOps.exif_transpose(image)
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    
    for polygon, route_id, grade, is_incomplete, is_volume in zip(target["polygons"], target["route_ids"], target["grades"], target["is_incomplete"], target["is_volume"]):
        color = RT_ID_COLOR_MAP.get(route_id, "yellow")
        draw.polygon(polygon, outline=color, width=8)
        label = f"ID: {route_id}, Grade: {grade}"
        if is_incomplete:
            label += " (Incomplete)"
        if is_volume:
            label += " (Volume)"
        draw.text(polygon[0], label, fill=color)
    
    plt.imshow(result)
    plt.title("Grade Sample")
    plt.show()
    
    
def visualize_predictions(image_pil, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, idx_to_grade, score_threshold=0.3, output_name="prediction.png"):
    result = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        grade = idx_to_grade.get(int(label), "BG")
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
        draw.text((x1, max(0, y1 - 12)), f"GT: {grade}", fill="green")

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        grade = idx_to_grade.get(int(label), "BG")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, min(y2 + 2, result.height - 12)), f"Pred: {grade} ({score:.2f})", fill="red")

    plt.figure(figsize=(14, 10))
    plt.imshow(result)
    plt.title("Green = GT, Red = Predicted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close()


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
    )
    
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
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


def main():
    grade_dataset = GradeDataset(
        img_dir=GD_TEST_IMG_PATH,
        ann_csv=GD_TEST_ANN_PATH,
        drop_incomplete=True,
        drop_volume=True
    )
    dataset_size = len(grade_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(grade_dataset, [train_size, val_size], generator=generator)

    BATCH_SIZE = 2
    NUM_WORKERS = 2

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    num_classes = len(grade_dataset.grade_to_idx) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU compute capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")
    print(f"Grades: {grade_dataset.grade_to_idx}")

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 15
    loss_history = []
    start_train_time = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        avg_loss = train_loss / len(train_dataloader)

        est_total_time = (time.perf_counter() - start_train_time) / (epoch + 1) * num_epochs
        elapsed_time = time.perf_counter() - start_train_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Est. Time: {est_total_time:.2f}s")
        loss_history.append(avg_loss)
    end_train_time = time.perf_counter()
    print(f"Training completed in {(end_train_time - start_train_time):.2f} seconds")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "grade_to_idx": grade_dataset.grade_to_idx,
        "idx_to_grade": grade_dataset.idx_to_grade,
        "epoch": num_epochs,
        "loss_history": loss_history,
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, "route_grader_maskrcnn_checkpoint.pt")
    print("Saved checkpoint to route_grader_maskrcnn_checkpoint.pt")

    model.eval()
    all_true = []
    all_pred = []
    iou_threshold = 0.5
    score_threshold = 0.3
    no_detection_label = num_classes
    eval_start_time = time.perf_counter()
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            gt_boxes = targets[0]["boxes"].cpu().numpy()
            gt_labels = targets[0]["labels"].cpu().numpy()

            pred_scores = outputs[0]["scores"].detach().cpu().numpy()
            pred_boxes = outputs[0]["boxes"].detach().cpu().numpy()
            pred_labels = outputs[0]["labels"].detach().cpu().numpy()

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

            candidates = []
            for gt_idx, gt_box in enumerate(gt_boxes):
                for pred_idx, pred_box in enumerate(pred_boxes):
                    iou = box_iou(gt_box, pred_box)
                    if iou >= iou_threshold:
                        candidates.append((iou, gt_idx, pred_idx))

            candidates.sort(reverse=True, key=lambda x: x[0])
            matched_gt = set()
            matched_pred = set()

            for _, gt_idx, pred_idx in candidates:
                if gt_idx in matched_gt or pred_idx in matched_pred:
                    continue
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                all_true.append(int(gt_labels[gt_idx]))
                all_pred.append(int(pred_labels[pred_idx]))

            for gt_idx in range(len(gt_labels)):
                if gt_idx not in matched_gt:
                    all_true.append(int(gt_labels[gt_idx]))
                    all_pred.append(no_detection_label)

            for pred_idx in range(len(pred_labels)):
                if pred_idx not in matched_pred:
                    all_true.append(no_detection_label)
                    all_pred.append(int(pred_labels[pred_idx]))

    eval_labels = list(range(num_classes)) + [no_detection_label]
    CM = confusion_matrix(all_true, all_pred, labels=eval_labels)
    acc = float(np.mean(np.array(all_true) == np.array(all_pred))) if all_true else 0.0
    report = classification_report(all_true, all_pred, labels=eval_labels, zero_division=0, digits=4)

    print("Confusion Matrix:")
    print(CM)
    print(f"Accuracy: {acc:.4f}")
    print(f"No-detection label index: {no_detection_label}")
    print("Classification report:")
    print(report)

    end_eval_time = time.perf_counter()
    print(f"Evaluation completed in {(end_eval_time - eval_start_time):.2f} seconds")

    # visualize confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(CM, interpolation="nearest", cmap=plt.cm.Blues)
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
            tick_names.append(grade_dataset.idx_to_grade.get(lbl, str(lbl)))

    plt.xticks(tick_marks, tick_names, rotation=45, ha="right")
    plt.yticks(tick_marks, tick_names)

    thresh = CM.max() / 2.0
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            plt.text(j, i, str(CM[i, j]),
                     ha="center", va="center",
                     color="white" if CM[i, j] > thresh else "black",
                     fontsize=8)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # visualize some predictions
    model.eval()
    val_iter = iter(val_dataloader)
    with torch.no_grad():
        for i in range(min(5, len(val_dataloader))):
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
                grade_dataset.idx_to_grade,
                score_threshold=0.3,
                output_name=f"prediction_{i+1}.png"
            )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()