import ast
import os
import logging
import json
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageDraw

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs/tiled_hold_dataset.log")


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


class TiledWallDataset(Dataset):
    def __init__(self, img_dir, ann_csv, transform=None):
        self.img_dir = img_dir
        self.ann_csv = ann_csv
        self.transform = transform

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.annotations = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)
        
        self.rows_by_img = defaultdict(list)
        for _, row in self.annotations.iterrows():
            self.rows_by_img[row["filename"]].append(row)
            
        self.tiles = []
        for fname in sorted(self.rows_by_img.keys()):
            if not os.path.isfile(os.path.join(img_dir, fname)):
                logger.warning(f"Image file {fname} listed in CSV but not found in directory. Skipping.")
                continue
            rows = self.rows_by_img[fname]
            is_neg = self._parse_bool(rows[0].get("is_negative", False)) is True
            self.tiles.append({
                "filename": fname,
                "rows": rows,
                "is_negative": is_neg
            })
            
        hold_types = set()
        for tile in self.tiles:
            if tile["is_negative"]:
                continue
            for row in tile["rows"]:
                hold_types.add(row["hold_type"])
        hold_types = sorted(hold_types)
        
        self.hold_type_to_idx = {ht: idx + 1 for idx, ht in enumerate(hold_types)}
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}
        
        assert len(self.img_files) == len(self.annotations), (
            f"Mismatch: {len(self.img_files)} image files vs {len(self.annotations)} annotation rows"
        )
        
        logger.info(
            "TiledWallDataset ready: %d tiles (%d positive, %d negative) | "
            "%d hold classes: %s",
            len(self.tiles),
            sum(1 for t in self.tiles if not t["is_negative"]),
            sum(1 for t in self.tiles if t["is_negative"]),
            len(self.hold_type_to_idx),
            list(self.hold_type_to_idx.keys()),
        )
                         
    def __len__(self):
        return len(self.tiles)
    
    def _json(self, s):
        if not isinstance(s, str) or s.strip() in ("", "[]", "{}"):
            return {}
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return {}
            
    def _parse_bool(self, s):
        if isinstance(s, bool):
            return s
        if isinstance(s, str):
            s = s.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            elif s in ("false", "0", "no", ""):
                return False
        return False
        
    def __getitem__(self, index):
        tile_info = self.tiles[index]
        tile_path = os.path.join(self.img_dir, tile_info["filename"])
        
        # EXIF orientation
        image = ImageOps.exif_transpose(Image.open(tile_path)).convert("RGB")
        W, H = image.size
        image_tensor = self.transform(image) if self.transform else F.to_tensor(image)
        
        if tile_info["is_negative"]:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "masks": torch.empty((0, H, W), dtype=torch.uint8),
                "area": torch.empty((0,), dtype=torch.float32),
                "image_id": torch.tensor([index])
            }
            return image_tensor, target
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for row in tile_info["rows"]:
            polygon = self._json(row.get("polygon", ""))
            xs = polygon.get("all_points_x", [])
            ys = polygon.get("all_points_y", [])
            if not xs or not ys or len(xs) != len(ys):
                logger.warning(f"{tile_info['filename']}:{row.get('region_id')} - Unannotated tile or invalid polygon. Skipping.")
                continue
            
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            if x_max <= x_min or y_max <= y_min:
                logger.warning(f"{tile_info['filename']}:{row.get('region_id')} - Degenerate bounding box. Skipping.")
                continue
            
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, W)
            y_max = min(y_max, H)
            
            if x_max <= x_min or y_max <= y_min:
                logger.warning(f"{tile_info['filename']}:{row.get('region_id')} - Bounding box out of bounds after clipping. Skipping.")
                continue
            
            hold_type = row.get("hold_type", "hold").strip()
            label_idx = self.hold_type_to_idx.get(hold_type)
            if label_idx is None:
                logger.warning(
                    f"{tile_info['filename']}:{row.get('region_id')} - hold_type '{hold_type}' not in training mapping {sorted(self.hold_type_to_idx.keys())}; skipping.",
                )
                continue
            
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=1)
            mask_np = np.array(mask, dtype=np.uint8)
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label_idx)
            masks.append(mask_np)
            areas.append((x_max - x_min) * (y_max - y_min))
            
        if not boxes:
            logger.warning(f"{tile_info['filename']} - no valid annotations found.")
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "masks": torch.empty((0, H, W), dtype=torch.uint8),
                "area": torch.empty((0,), dtype=torch.float32),
                "image_id": torch.tensor([index])
            }
            return image_tensor, target

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks, axis=0)),
            "area": torch.tensor(areas, dtype=torch.float32),
            "image_id": torch.tensor([index])
        }
        return image_tensor, target

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return tuple(zip(*batch)) if batch else ([], [])
