import os, csv, json, sys

path = sys.argv[1]
out = sys.argv[2]


def parse_bool_like(value):
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


def normalize_attr_value(value, field_name=None):
    if field_name in {"incomplete_route", "is_volume"}:
        if isinstance(value, dict):
            if len(value) == 0:
                return False
            if len(value) == 1:
                key, inner_value = next(iter(value.items()))
                parsed_inner = parse_bool_like(inner_value)
                if parsed_inner is not None:
                    return parsed_inner
                parsed_key = parse_bool_like(key)
                if parsed_key is not None:
                    return parsed_key
            return None

        parsed_bool = parse_bool_like(value)
        if parsed_bool is not None:
            return parsed_bool

    if isinstance(value, dict):
        if len(value) == 0:
            return None
        if len(value) == 1:
            key, inner_value = next(iter(value.items()))
            key = str(key).strip().lower()
            if key in ("true", "false") and isinstance(inner_value, bool):
                return inner_value
            if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
                return inner_value
        return json.dumps(value, sort_keys=True)

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"undefined", "null", "none", "nan"}:
            return None
        return value

    return value

hold_counts = 0
route_counts = 0
grade_distribution = {}
output_rows = []


def update_grade_distribution_for_routes(routes, grade_distribution):
    for route_id, route_attrs in routes.items():
        if route_id is None or len(route_attrs) == 0:
            continue

        chosen_attr, _ = max(
            route_attrs.items(),
            key=lambda item: (len(item[1]), str(item[0]))
        )
        chosen_grade = chosen_attr[0]
        grade_distribution[chosen_grade] = grade_distribution.get(chosen_grade, 0) + 1


def report_route_conflicts(image_name, routes):
    for route_id, route_attrs in routes.items():
        if route_id is None:
            continue
        if len(route_attrs) == 0:
            print(f"{image_name} - Route {route_id} has no attributes")
        elif len(route_attrs) > 1:
            print(f"{image_name} - Route {route_id} has conflicting attributes:")
            print("  Grade\t| Inc.\t| Volume\t| Regions")
            for attr, region_ids in sorted(route_attrs.items(), key=lambda item: str(item[0])):
                print(f"    {attr[0]}\t| {attr[1]}\t| {attr[2]}\t| {sorted(region_ids)}")

            majority_count = max(len(region_ids) for region_ids in route_attrs.values())
            outliers = [
                (attr, sorted(region_ids))
                for attr, region_ids in route_attrs.items()
                if len(region_ids) < majority_count
            ]
            if outliers:
                print("  Outlier region_ids:")
                for attr, region_ids in sorted(outliers, key=lambda item: str(item[0])):
                    print(f"    {region_ids} -> {attr}")

# csv format: filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
# Ex: 00.jpg,8294565,"{}",187,0,"{""name"":""polygon"",""all_points_x"":[2406,2416,2421,2422,2415,2406,2395,2391,2389,2395,2402],""all_points_y"":[1670,1667,1659,1648,1639,1637,1640,1648,1658,1666,1671]}","{""route_id"":""1"",""route_grade"":""VB"",""incomplete_route"":""true"",""is_volume"":""false""}"

with open(path, "r") as f:
    reader = csv.DictReader(f)

    current_image = None
    image_route_ids = set()
    routes = {}

    for row in reader:
        hold_counts += 1
        
        img_name = row["filename"]

        if current_image is None:
            current_image = img_name
        elif img_name != current_image:
            report_route_conflicts(current_image, routes)
            update_grade_distribution_for_routes(routes, grade_distribution)
            route_counts += len(image_route_ids)
            image_route_ids = set()
            routes = {}
            current_image = img_name

        region_id = row["region_id"]
        count = int(row["region_count"])
        size = int(row["file_size"])
        shape = json.loads(row["region_shape_attributes"])
        attr = json.loads(row["region_attributes"])
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        route_id = int(normalize_attr_value(attr.get("route_id", None), "route_id")) if normalize_attr_value(attr.get("route_id", None), "route_id") is not None else None
        route_grade = normalize_attr_value(attr.get("route_grade", None), "route_grade")
        # print(f"Processing {img_name}:{region_id} - route_id={route_id}, route_grade={route_grade}")
        incomplete_route = normalize_attr_value(attr.get("incomplete_route", False), "incomplete_route")
        is_volume = normalize_attr_value(attr.get("is_volume", False), "is_volume")
        hold_type = normalize_attr_value(attr.get("hold_type", "hold"), "hold_type")
        
        if route_grade == "INC" or route_grade == "A":
            normalized_route_grade = None
        else:
            normalized_route_grade = route_grade
        if incomplete_route is True and normalized_route_grade is None:
            normalized_route_grade = "INC"

        normalized_attr = {"route_id": route_id, "route_grade": normalized_route_grade, "incomplete_route": incomplete_route, "is_volume": is_volume, "hold_type": hold_type}
        
        if not xs or not ys or len(xs) != len(ys):
            print(f"{img_name}:{region_id} - Unannotated image")
            continue
        
        if route_id is not None:
            image_route_ids.add(route_id)

            if routes.get(route_id) is None:
                routes[route_id] = {}

            route_attr_tuple = (normalized_route_grade, incomplete_route, is_volume, hold_type)
            if route_attr_tuple not in routes[route_id]:
                routes[route_id][route_attr_tuple] = []
            routes[route_id][route_attr_tuple].append(region_id)
        
        if route_id is None and normalized_route_grade is None and incomplete_route is None and is_volume is None and hold_type is None:
            print(f"{img_name}:{region_id} - Missing route attributes")
        if route_id is None and is_volume is False:
            print(f"{img_name}:{region_id} - Missing route_id")
        if normalized_route_grade is None and is_volume is False and incomplete_route is False:
            print(f"{img_name}:{region_id} - Missing route_grade")
        if route_id is not None and normalized_route_grade is None and incomplete_route is False and is_volume is False:
            print(f"{img_name}:{region_id} - Missing route_grade")
        if incomplete_route is True and normalized_route_grade != "INC":
            print(f"{img_name}:{region_id} - Incomplete route should not have a grade")
            print(f"  Attributes: route_id={route_id}, route_grade={normalized_route_grade}, incomplete_route={incomplete_route}, is_volume={is_volume}")
        if hold_type is None:
            print(f"{img_name}:{region_id} - Missing hold_type")
        if hold_type not in ["hold", "volume", "pinch", "jug", "sloper", "crimp", "edge", "pocket"]:
            print(f"{img_name}:{region_id} - Unrecognized hold_type: {hold_type}")

        # csv format: filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
        # Ex: 00.jpg,8294565,"{}",187,0,"{""name"":""polygon"",""all_points_x"":[2406,2416,2421,2422,2415,2406,2395,2391,2389,2395,2402],""all_points_y"":[1670,1667,1659,1648,1639,1637,1640,1648,1658,1666,1671]}","{""route_id"":""1"",""route_grade"":""VB"",""incomplete_route"":""true"",""is_volume"":""false""}"
        output_rows.append({
            "filename": img_name,
            "file_size": size,
            "file_attributes": "{}",
            "region_count": count,
            "region_id": region_id,
            "region_shape_attributes": json.dumps(shape, sort_keys=True),
            "region_attributes": json.dumps(normalized_attr, sort_keys=True),
        })
        
    report_route_conflicts(current_image, routes)
    update_grade_distribution_for_routes(routes, grade_distribution)

    if current_image is not None:
        route_counts += len(image_route_ids)
        
with open(out, "w", newline="") as f:
    fieldnames = [
        "filename",
        "file_size",
        "file_attributes",
        "region_count",
        "region_id",
        "region_shape_attributes",
        "region_attributes",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)
    print(f"Preprocessed annotations written to {out}")

print(f"Total holds: {hold_counts}")
print(f"Total routes: {route_counts}")
print("Grade distribution:")
for grade, count in sorted(grade_distribution.items(), key=lambda item: (item[0] is None, item[0])):
    print(f"  {grade}: {count}")