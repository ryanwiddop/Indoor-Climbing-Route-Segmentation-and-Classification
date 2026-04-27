import argparse
import csv
import json
import os
import random
import shutil
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageDraw, ImageOps


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
    
    
def iter_tiles(H, W, tile, overlap):
    stride = max(1, int(round(tile * (1 - overlap))))
    
    def axis_positions(length):
        positions = list(range(0, max(0, length - tile) + 1, stride))
        if positions and positions[-1] + tile < length:
            positions.append(length - tile)
        return sorted(set(positions))
    
    for y0 in axis_positions(H):
        y1 = y0 + tile
        for x0 in axis_positions(W):
            x1 = x0 + tile
            yield (x0, y0, x1, y1)
            

def bb_overlap_fraction(xs, ys, tx0, ty0, tx1, ty1):
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    
    inter_xmin = max(x_min, tx0)
    inter_ymin = max(y_min, ty0)
    inter_xmax = min(x_max, tx1)
    inter_ymax = min(y_max, ty1)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    hold_area = (x_max - x_min) * (y_max - y_min)
    
    if hold_area == 0:
        return 0.0
    return inter_area / hold_area


def translate_polygon(xs, ys, x_offset, y_offset):
    return [x - x_offset for x in xs], [y - y_offset for y in ys]


def draw_tile_annotations(tile_img, holds):
    vis = tile_img.copy()
    draw = ImageDraw.Draw(vis)
    for hold in holds:
        polygon = list(zip(hold["xs"], hold["ys"]))
        if len(polygon) >= 2:
            draw.polygon(polygon, outline="yellow", width=3)
        x0 = min(hold["xs"])
        y0 = min(hold["ys"])
        label = f"{['hold_type']} #({hold['region_id']})"
        draw.text((x0, y0), label, fill="yellow")
    return vis


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("img_dir")
    p.add_argument("ann_csv")
    p.add_argument("out_dir")
    p.add_argument("--tile-size",         type=int,   default=800)
    p.add_argument("--overlap",           type=float, default=0.25)
    p.add_argument("--min-overlap-frac",  type=float, default=0.35,
                   help="Min fraction of hold bbox that must overlap tile (default 0.50)")
    p.add_argument("--vis-per-image",     type=int,   default=3,
                   help="Positive-tile visualisations to save per source image (default 3)")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    
    tiles_dir = os.path.join(args.out_dir, "tiles")
    vis_dir = os.path.join(args.out_dir, "visualisations")
    csv_path = os.path.join(args.out_dir, "tiles.csv")
    
    for d in [tiles_dir, vis_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(args.out_dir) and not os.listdir(args.out_dir):
        os.rmdir(args.out_dir)
            
    annotations = pd.read_csv(args.ann_csv, dtype=str, keep_default_na=False)
    
    rows_by_img = defaultdict(list)
    for _, row in annotations.iterrows():
        rows_by_img[row["filename"]].append(row)
        
    csv_fieldnames = [
        "filename",
        "source_image", 
        "tile_x0", "tile_y0", "tile_x1", "tile_y1", 
        "is_negative",
        "region_id",
        "hold_type",
        "grade",
        "is_volume",
        "incomplete_route",
        "route_id",
        "polygon",
    ]
    
    total_tiles = 0
    total_positive = 0
    total_negative = 0
    total_hold_instances = 0
    skipped_images = 0
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        
        for img_name, rows in sorted(rows_by_img.items()):
            img_path = os.path.join(args.img_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"Warning: Image file {img_path} not found. Skipping.")
                skipped_images += 1
                continue
            
            image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            W, H = image.size
            img_stem = os.path.splitext(img_name)[0]
            
            holds = []
            for row in rows:
                shape = parse_json(row["region_shape_attributes"])
                attr = parse_json(row["region_attributes"])
                xs = shape.get("all_points_x", [])
                ys = shape.get("all_points_y", [])
                if not xs or not ys or len(xs) != len(ys):
                    continue
                if max(xs) <= min(xs) or max(ys) <= min(ys):
                    continue
                
                grade = attr.get("route_grade", "")
                incomplete_route = parse_bool(attr.get("incomplete_route", False))
                if incomplete_route:
                    grade = "INC"
                    
                holds.append({
                    "region_id": row.get("region_id", ""),
                    "hold_type": attr.get("hold_type", ""),
                    "grade": grade,
                    "is_volume": parse_bool(attr.get("is_volume", False)),
                    "incomplete_route": incomplete_route,
                    "route_id": attr.get("route_id", ""),
                    "xs": xs,
                    "ys": ys,
                })
            
            tile_records = []
            for tx0, ty0, tx1, y1 in iter_tiles(H, W, args.tile_size, args.overlap):
                tile_name = f"{img_stem}_t{tx0}_{ty0}.jpg"
                tile_path = os.path.join(tiles_dir, tile_name)
                
                tile_crop = image.crop((tx0, ty0, tx1, y1))
                tile_crop.save(tile_path, "JPEG", quality=95)
                total_tiles += 1
                
                qualifying_holds = []
                for hold in holds:
                    frac = bb_overlap_fraction(hold["xs"], hold["ys"], tx0, ty0, tx1, y1)
                    if frac >= args.min_overlap_frac:
                        lxs, lys = translate_polygon(hold["xs"], hold["ys"], tx0, ty0)
                        qualifying_holds.append({**hold, "local_xs": lxs, "local_ys": lys})
                
                is_negative = len(qualifying_holds) == 0
                if is_negative:
                    total_negative += 1
                    writer.writerow({
                        "filename": tile_name,
                        "source_image": img_name,
                        "tile_x0": tx0,
                        "tile_y0": ty0,
                        "tile_x1": tx1,
                        "tile_y1": y1,
                        "is_negative": True,
                        "region_id": -1,
                        "hold_type": "",
                        "grade": "",
                        "is_volume": False,
                        "incomplete_route": False,
                        "route_id": "",
                        "polygon": {},
                    })
                else:
                    total_positive += 1
                    total_hold_instances += len(qualifying_holds)
                    for hold in qualifying_holds:
                        polygon_dict = {
                            "all_points_x": hold["local_xs"],
                            "all_points_y": hold["local_ys"],
                        }
                        writer.writerow({
                            "filename": tile_name,
                            "source_image": img_name,
                            "tile_x0": tx0,
                            "tile_y0": ty0,
                            "tile_x1": tx1,
                            "tile_y1": y1,
                            "is_negative": False,
                            "region_id": hold["region_id"],
                            "hold_type": hold["hold_type"],
                            "grade": hold["grade"],
                            "is_volume": hold["is_volume"],
                            "incomplete_route": hold["incomplete_route"],
                            "route_id": hold["route_id"],
                            "polygon": json.dumps(polygon_dict),
                        })
                
                tile_records.append((tile_name, tile_crop, qualifying_holds))
                
            pos_records = [(n, c, q) for n, c, q in tile_records if q]
            sample_n = min(args.vis_per_image, len(pos_records))
            for tile_name, tile_crop, qualifying_holds in random.sample(pos_records, sample_n):
                hold_data = [{
                    "xs": hold["local_xs"],
                    "ys": hold["local_ys"],
                    "region_id": hold["region_id"],
                    "hold_type": hold["hold_type"],
                } for hold in qualifying_holds]
                vis = draw_tile_annotations(tile_crop, hold_data)
                vis_name = f"{os.path.splitext(tile_name)[0]}_vis.jpg"
                vis_path = os.path.join(vis_dir, vis_name)
                vis.save(vis_path, "JPEG", quality=95)
                
            print(
                f"   {img_name}: {len(holds)} holds -> "
                f"{total_tiles} tiles, "
                f"{total_positive} pos / {total_negative} neg, "
            )
            
    neg_pos_ratio = total_negative / max(1, total_positive)
    avg_holds_per_pos_tile = total_hold_instances / max(1, total_positive)
    
    print()
    print("=" * 60)
    print(f"Tile size           : {args.tile_size}px  overlap={args.overlap}")
    print(f"Min hold overlap    : {args.min_overlap_frac:.0%}")
    print(f"Total tiles         : {total_tiles}")
    print(f"  Positive          : {total_positive}")
    print(f"  Negative          : {total_negative}")
    print(f"  Neg:Pos ratio     : {neg_pos_ratio:.2f}")
    print(f"Hold instances      : {total_hold_instances}  ({avg_holds_per_pos_tile:.1f} avg per pos tile)")
    print(f"Images skipped      : {skipped_images}")
    print(f"Tiles dir           : {tiles_dir}")
    print(f"CSV                 : {csv_path}")
    print(f"Visualizations      : {vis_dir}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 