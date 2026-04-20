import ast
import os
import logging
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageDraw

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs/cropped_hold_dataset.log")


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


class HoldDataset(Dataset):
    def __init__(self, img_dir, ann_csv, transform=None):
        self.img_dir = img_dir
        self.ann_csv = ann_csv
        self.transform = transform

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.annotations = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)
        
        self.rows_by_img = defaultdict(list)
        for _, row in self.annotations.iterrows():
            self.rows_by_img[row["filename"]].append(row)
            
        hold_types = set()
        for rows in self.rows_by_img.values():
            for row in rows:
                hold_types.add(row["hold_type"])
        hold_types = sorted(hold_types)
        self.hold_type_to_idx = {ht: idx + 1 for idx, ht in enumerate(hold_types)}
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}
        
        assert len(self.img_files) == len(self.annotations), (
            f"Mismatch: {len(self.img_files)} image files vs {len(self.annotations)} annotation rows"
        )
                         
    def __len__(self):
        return len(self.annotations)
    
    def _json(self, s):
        if not isinstance(s, str) or s.strip() in ("", "[]", "{}"):
            return {}
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return {}

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.isfile(img_path):
            logger.warning(f"Image file {img_path} not found. Skipping.")
            return None
        
        # EXIF orientation
        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)
            
        hold_type = row["hold_type"]
        polygon = self._json(row["polygon"])
        xs = polygon.get("all_points_x", [])
        ys = polygon.get("all_points_y", [])
        
        if not xs or not ys or len(xs) != len(ys):
            logger.warning(f"{img_name}:{row['region_id']} - Unannotated image or invalid polygon. Skipping.")
            return None
        
        pil_mask = Image.new("L", (image.shape[2], image.shape[1]), 0)
        draw = ImageDraw.Draw(pil_mask)
        draw.polygon(list(zip(xs, ys)), fill=1)
        mask = torch.tensor(np.array(pil_mask), dtype=torch.uint8)
        
        box = torch.tensor([min(xs), min(ys), max(xs), max(ys)], dtype=torch.float32)   
        label_idx = self.hold_type_to_idx.get(hold_type, 0)
        area = (box[2] - box[0]) * (box[3] - box[1]) 

        target = {
            "boxes": box.unsqueeze(0),
            "labels": torch.tensor([label_idx], dtype=torch.int64),
            "masks": mask.unsqueeze(0),
            "area": area.unsqueeze(0),
            "image_id": torch.tensor([index])
        }
        return image, target


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return tuple(zip(*batch)) if batch else ([], [])
