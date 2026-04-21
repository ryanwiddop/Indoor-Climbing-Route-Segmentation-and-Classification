import ast
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class WholeWallDataset(Dataset):
    def __init__(self, img_dir, ann_csv, hold_type_to_idx=None, return_masks=False, transform=None):
        self.img_dir = img_dir
        self.ann_csv = ann_csv
        self.return_masks = return_masks
        self.transform = transform

        self.annotations = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)

        self.rows_by_img = defaultdict(list)
        for _, row in self.annotations.iterrows():
            self.rows_by_img[row["filename"]].append(row)

        self.img_files = []
        for fname in sorted(self.rows_by_img.keys()):
            if os.path.isfile(os.path.join(img_dir, fname)):
                self.img_files.append(fname)
            else:
                logger.warning(
                    "Image %s has annotations but is not in %s - skipping.",
                    fname, img_dir,
                )

        if hold_type_to_idx is None:
            seen = set()
            for rows in self.rows_by_img.values():
                for row in rows:
                    attr = self._json_or_literal(row["region_attributes"])
                    seen.add(attr.get("hold_type", "hold"))
            seen = sorted(seen)
            self.hold_type_to_idx = {ht: i + 1 for i, ht in enumerate(seen)}
        else:
            self.hold_type_to_idx = dict(hold_type_to_idx)
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _json_or_literal(s):
        if not isinstance(s, str) or s.strip() in ("", "[]", "{}"):
            return {}
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return {}

    @staticmethod
    def _rasterize_polygon(xs, ys, width, height):
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=1)
        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, index):
        fname = self.img_files[index]
        img_path = os.path.join(self.img_dir, fname)

        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = image.size
        image_tensor = self.transform(image) if self.transform else F.to_tensor(image)

        boxes, labels, polygons = [], [], []
        hold_types, route_ids, route_grades, is_volumes = [], [], [], []
        masks_np = []

        for row in self.rows_by_img[fname]:
            shape = self._json_or_literal(row["region_shape_attributes"])
            attr = self._json_or_literal(row["region_attributes"])
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])

            if not xs or not ys or len(xs) != len(ys):
                logger.warning("%s:%s - invalid polygon, skipping.", fname, row.get("region_id"))
                continue

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            if x_max <= x_min or y_max <= y_min:
                logger.warning("%s:%s - degenerate bbox, skipping.", fname, row.get("region_id"))
                continue

            hold_type = attr.get("hold_type", "hold")
            label_idx = self.hold_type_to_idx.get(hold_type)
            if label_idx is None:
                logger.warning(
                    "%s:%s - hold_type %r not in training mapping %s; skipping.",
                    fname, row.get("region_id"), hold_type,
                    sorted(self.hold_type_to_idx.keys()),
                )
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label_idx)
            polygons.append((list(xs), list(ys)))
            hold_types.append(hold_type)
            route_ids.append(attr.get("route_id", ""))
            route_grades.append(attr.get("route_grade", ""))
            is_volumes.append(bool(attr.get("is_volume", False)))

            if self.return_masks:
                masks_np.append(self._rasterize_polygon(xs, ys, W, H))

        if len(boxes) == 0:
            logger.warning("%s - no valid annotations found.", fname)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "area": area_t,
            "image_id": torch.tensor([index]),
            "polygons": polygons,
            "filename": fname,
            "image_size": (W, H),
            "hold_types": hold_types,
            "route_ids": route_ids,
            "route_grades": route_grades,
            "is_volumes": is_volumes,
        }

        if self.return_masks:
            if masks_np:
                target["masks"] = torch.from_numpy(np.stack(masks_np, axis=0))
            else:
                target["masks"] = torch.zeros((0, H, W), dtype=torch.uint8)

        return image_tensor, target


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return tuple(zip(*batch)) if batch else ([], [])
