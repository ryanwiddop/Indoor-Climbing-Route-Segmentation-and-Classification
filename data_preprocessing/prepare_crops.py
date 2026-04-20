from collections import defaultdict
import random
import os, sys, json, csv, shutil
import pandas as pd
from PIL import Image, ImageOps, ImageDraw


img_dir = sys.argv[1]
ann_csv = sys.argv[2]
out_dir = sys.argv[3]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "null", "none", "nan", ""}:
            return False
    return None


def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def polygon_to_bbox(xs, ys):
    return [min(xs), min(ys), max(xs), max(ys)]


def get_padded_crop_box(bbox, img_width, img_height, padding=10):
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img_width, x_max + padding)
    y_max = min(img_height, y_max + padding)
    return (x_min, y_min, x_max, y_max)


def pad_crop(image, crop_box):
    return image.crop(crop_box)


def translate_polygon_to_crop(xs, ys, crop_box):
    crop_x_min, crop_y_min, _, _ = crop_box
    return [x - crop_x_min for x in xs], [y - crop_y_min for y in ys]


def visualize_crop_with_polygon(crop, xs, ys):
    draw = ImageDraw.Draw(crop)
    polygon = list(zip(xs, ys))
    draw.polygon(polygon, outline="yellow", width=3)
    return crop



crops_dir = os.path.join(out_dir, "crops")
crops_csv_path = os.path.join(out_dir, "crops.csv")

if os.path.exists(crops_dir):
    shutil.rmtree(crops_dir)
os.makedirs(crops_dir)

if os.path.exists(crops_csv_path):
    os.remove(crops_csv_path)

annotations = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)

rows_by_img = defaultdict(list)
for _, row in annotations.iterrows():
    rows_by_img[row["filename"]].append(row)
    
skipped_images = 0
crop_widths = []
crop_heights = []

fieldnames = ["filename", "source_image", "grade", "is_volume", "incomplete_route", "region_id", "route_id", "hold_type", "polygon"]
with open(crops_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for img_name, rows in rows_by_img.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"Warning: Image file {img_path} not found. Skipping.")
            skipped_images += 1
            continue

        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        w, h = image.size
        img_stem = os.path.splitext(img_name)[0]
        
        rows = rows_by_img.get(img_name, [])
        for row in rows:
            shape = parse_json(row["region_shape_attributes"])
            attr = parse_json(row["region_attributes"])
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys or len(xs) != len(ys):
                skipped_images += 1
                continue
            
            x_min, y_min, x_max, y_max = polygon_to_bbox(xs, ys)
            if x_max <= x_min or y_max <= y_min:
                skipped_images += 1
                continue
            
            crop_box = get_padded_crop_box((x_min, y_min, x_max, y_max), w, h)
            translated_xs, translated_ys = translate_polygon_to_crop(xs, ys, crop_box)
            
            region_id = row.get("region_id", "unknown")
            is_volume = parse_bool(attr.get("is_volume", False))
            incomplete_route = parse_bool(attr.get("incomplete_route", False))
            route_id = attr.get("route_id", "")
            route_grade = attr.get("route_grade", "")
            region_id = row.get("region_id", "unknown")
            hold_type = attr.get("hold_type", "hold")
            if incomplete_route is True:
                route_grade = "INC"

            crop = pad_crop(image, crop_box)
            crop_widths.append(crop_box[2] - crop_box[0])
            crop_heights.append(crop_box[3] - crop_box[1])
            crop_filename = f"{img_stem}_region{region_id}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            if os.path.exists(crop_path):
                os.remove(crop_path)
            crop.save(crop_path, "JPEG", quality=95)
            
            writer.writerow({
                "filename": crop_filename,
                "source_image": img_name,
                "grade": route_grade,
                "is_volume": is_volume,
                "incomplete_route": incomplete_route,
                "region_id": region_id,
                "route_id": route_id,
                "hold_type": hold_type,
                "polygon": {"all_points_x": translated_xs, "all_points_y": translated_ys}
            })
            
            vis_dir = os.path.join(out_dir, "visualizations")
            if os.path.exists(vis_dir):
                for f in os.listdir(vis_dir):
                    if f.startswith(f"{img_stem}_region{region_id}_vis"):
                        os.remove(os.path.join(vis_dir, f))
            os.makedirs(vis_dir, exist_ok=True)
            if random.random() < 0.1:
                vis_path = os.path.join(vis_dir, f"{img_stem}_region{region_id}_vis.jpg")
                if os.path.exists(vis_path):
                    os.remove(vis_path)
                visualize_crop_with_polygon(crop, translated_xs, translated_ys).save(vis_path, "JPEG", quality=95)
        
        print(f"  {img_name}: {len(rows)} polygons processed")
            
print(f"Crops created:  {writer.writerow.__code__.co_argcount - 1} rows")
print(f"Images skipped: {skipped_images}")
print(f"Crops DIR:      {crops_dir}")
print(f"Crops CSV:      {crops_csv_path}")
print(f"Avg. Widths:    {sum(crop_widths) / len(crop_widths):.2f}")
print(f"Crop Heights:   {sum(crop_heights) / len(crop_heights):.2f}")
