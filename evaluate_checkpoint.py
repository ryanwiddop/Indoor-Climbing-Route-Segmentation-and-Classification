import argparse
import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader

from model import GradeDataset, build_model, collate_fn, box_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Mask R-CNN checkpoint and export visualizations.")
    parser.add_argument("--checkpoint", default="route_grader_maskrcnn_checkpoint.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--images", default="images", help="Directory containing .jpg images")
    parser.add_argument("--annotations", default="annotation.csv", help="Path to annotation CSV")
    parser.add_argument("--batch-size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for train/val split")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for box matching")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Confidence threshold for predictions")
    parser.add_argument("--max-vis", type=int, default=5, help="Max number of prediction visualizations to save")
    parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts")
    return parser.parse_args()


def draw_prediction_overlay(
    image_pil,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    idx_to_grade,
    score_threshold,
    output_path,
):
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
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, eval_labels, idx_to_grade, no_detection_label, output_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
            tick_names.append(idx_to_grade.get(lbl, str(lbl)))

    plt.xticks(tick_marks, tick_names, rotation=45, ha="right")
    plt.yticks(tick_marks, tick_names)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def match_batch_predictions(targets, outputs, iou_threshold, score_threshold, no_detection_label):
    all_true = []
    all_pred = []

    for target, output in zip(targets, outputs):
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()

        pred_scores = output["scores"].detach().cpu().numpy()
        pred_boxes = output["boxes"].detach().cpu().numpy()
        pred_labels = output["labels"].detach().cpu().numpy()

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

    return all_true, all_pred


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("eval_outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    idx_to_grade = {int(k): v for k, v in checkpoint["idx_to_grade"].items()}

    dataset = GradeDataset(
        img_dir=args.images,
        ann_csv=args.annotations,
        drop_incomplete=True,
        drop_volume=True,
    )

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    iou_threshold = args.iou_threshold
    score_threshold = args.score_threshold
    no_detection_label = num_classes

    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            batch_true, batch_pred = match_batch_predictions(
                targets,
                outputs,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                no_detection_label=no_detection_label,
            )
            all_true.extend(batch_true)
            all_pred.extend(batch_pred)

    eval_labels = list(range(num_classes)) + [no_detection_label]
    cm = confusion_matrix(all_true, all_pred, labels=eval_labels)
    acc = float(np.mean(np.array(all_true) == np.array(all_pred))) if all_true else 0.0
    report = classification_report(all_true, all_pred, labels=eval_labels, zero_division=0, digits=4)

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"No-detection label index: {no_detection_label}")
    print("Classification report:")
    print(report)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(cm, eval_labels, idx_to_grade, no_detection_label, cm_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "accuracy": acc,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
                "no_detection_label": no_detection_label,
                "labels": eval_labels,
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            },
            f,
            indent=2,
        )

    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved metrics to: {metrics_path}")

    # Save a few qualitative predictions.
    vis_count = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images_gpu = [img.to(device) for img in images]
            outputs = model(images_gpu)

            for image_tensor, target, output in zip(images, targets, outputs):
                image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                pred_boxes = output["boxes"].detach().cpu().numpy()
                pred_labels = output["labels"].detach().cpu().numpy()
                pred_scores = output["scores"].detach().cpu().numpy()

                vis_path = os.path.join(output_dir, f"prediction_{vis_count + 1}.png")
                draw_prediction_overlay(
                    image_pil,
                    gt_boxes,
                    gt_labels,
                    pred_boxes,
                    pred_labels,
                    pred_scores,
                    idx_to_grade,
                    score_threshold=score_threshold,
                    output_path=vis_path,
                )
                vis_count += 1

                if vis_count >= args.max_vis:
                    break

            if vis_count >= args.max_vis:
                break

    print(f"Saved {vis_count} prediction visualizations to: {output_dir}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
