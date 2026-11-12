"""Descriptive particle annotation audit only: no subtype, clustering or training."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import textwrap
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived/particle_analysis_v1"
FIXED = ROOT / "data/derived/fixed_split_v1"
AUDIT = ROOT / "data/derived/gt_629_audit_v1"
SEED = 42
PERCENTILES = (25, 50, 75, 90, 95, 99)
GEOMETRY = ("bbox_width", "bbox_height", "bbox_area", "aspect_ratio",
            "normalized_width", "normalized_height", "normalized_area",
            "center_x", "center_y", "relative_center_x", "relative_center_y")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return None
    require(np.isfinite(values).all(), "Non-finite descriptive input")
    return {
        "n": int(len(values)), "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "std_population": float(values.std(ddof=0)),
        "percentiles": {str(q): float(np.percentile(values, q, method="linear"))
                        for q in PERCENTILES},
    }


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap *= max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / (aw * ah + bw * bh - overlap) if overlap > 0 else 0.0


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fraction_record(count, total=5444):
    return {"count": int(count), "fraction": count / total}


def load():
    require(not OUT.exists(), "Output exists; never overwrite the v1 audit")
    frozen = json.loads((FIXED / "split_summary.json").read_text(encoding="utf-8"))
    manifest = FIXED / "split_manifest.csv"
    require(sha(manifest) == frozen["split_manifest_sha256"], "Fixed manifest changed")
    protected = {p: sha(p) for base in (FIXED, AUDIT)
                 for p in base.rglob("*") if p.is_file()}
    source = ROOT / frozen["source"]["coco_archive"]
    protected[source] = sha(source)
    expected = frozen["source"]["input_sha256"][source.relative_to(ROOT).as_posix()]
    require(protected[source] == expected, "COCO source differs from audited source")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        mapping = list(csv.DictReader(stream))
    require(len(mapping) == len({r["image_id"] for r in mapping}) == 629, "Invalid mapping")
    by_id = {int(r["image_id"]): r for r in mapping}
    with zipfile.ZipFile(source) as archive:
        member = "train/_annotations.coco.json"
        coco_bytes = archive.read(member)
    coco = json.loads(coco_bytes)
    require(len(coco["images"]) == 629 and len(coco["annotations"]) == 5953, "Source size changed")
    images = {im["id"]: im for im in coco["images"]}
    require(set(images) == set(by_id), "Missing mapping identities")
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    per_image = defaultdict(list)
    ids = set()
    for ann in coco["annotations"]:
        require(ann["id"] not in ids, "Repeated annotation ID")
        ids.add(ann["id"])
        require(ann["image_id"] in images and categories[ann["category_id"]] in {"p", "o", "s", "sc"},
                "Unknown annotation reference")
        per_image[ann["image_id"]].append(ann)
    for image_id, image in images.items():
        row = by_id[image_id]
        require(row["filename"] == image["file_name"] and row["original_filename"] == image["extra"]["name"],
                "COCO/mapping identity differs")
        counts = Counter(categories[a["category_id"]] for a in per_image[image_id])
        require(all(counts[c] == int(row[c + "_count"]) for c in ("p", "o", "s", "sc")),
                "COCO/mapping class counts differ")
    return source, coco, images, by_id, categories, per_image, protected, {
        "coco_archive": source.relative_to(ROOT).as_posix(),
        "coco_json_member": member, "coco_archive_sha256": protected[source],
        "coco_json_sha256": hashlib.sha256(coco_bytes).hexdigest(),
        "fixed_manifest": manifest.relative_to(ROOT).as_posix(),
        "fixed_manifest_sha256": protected[manifest],
        "audit": AUDIT.relative_to(ROOT).as_posix(),
    }


def measure(coco, images, by_id, categories, per_image):
    particles = []
    overlap_pairs = Counter()
    overlap_ids = defaultdict(set)
    overlap_image_ids = defaultdict(set)
    overlap_ious = defaultdict(list)
    cooccurrence = Counter()
    inclusive = Counter()
    for image_id in sorted(images):
        image = images[image_id]
        mapping = by_id[image_id]
        width, height = image["width"], image["height"]
        p_anns = [a for a in per_image[image_id] if categories[a["category_id"]] == "p"]
        other = [a for a in per_image[image_id] if categories[a["category_id"]] != "p"]
        if p_anns:
            others = {categories[a["category_id"]] for a in other}
            key = "p_only" if not others else (
                "p_multiple_defects" if len(others) > 1 else "p_plus_" + next(iter(others)))
            cooccurrence[key] += 1
            for cls in others:
                inclusive["p_plus_" + cls] += 1
        for ann in sorted(p_anns, key=lambda a: a["id"]):
            x, y, w, h = map(float, ann["bbox"])
            require(all(math.isfinite(v) for v in (x, y, w, h)), "Non-finite bbox")
            require(w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height,
                    "Invalid source bbox; no automatic fix permitted")
            cx, cy = x + w / 2, y + h / 2
            rx, ry = cx / width, cy / height
            name = mapping["original_filename"]
            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
            crop_name = f"image_{image_id}__ann_{ann['id']}__{safe}__crop.png"
            left, top = math.floor(x), math.floor(y)
            right, bottom = math.ceil(x + w), math.ceil(y + h)
            row = {
                "annotation_id": ann["id"], "image_id": image_id, "filename": image["file_name"],
                "split": mapping["split"], "wafer": mapping["wafer"] or "UNKNOWN",
                "exposure": mapping["exposure"] or "UNKNOWN", "position": mapping["position"] or "UNKNOWN",
                "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h,
                "bbox_area": w * h, "normalized_width": w / width, "normalized_height": h / height,
                "normalized_area": w * h / (width * height), "aspect_ratio": w / h,
                "center_x": cx, "center_y": cy, "relative_center_x": rx, "relative_center_y": ry,
                "original_filename": name, "image_width_px": width, "image_height_px": height,
                "horizontal_region": ("left", "center", "right")[min(int(rx * 3), 2)],
                "vertical_region": ("top", "center", "bottom")[min(int(ry * 3), 2)],
                "edge_near_center": min(rx, 1 - rx, ry, 1 - ry) <= 0.1,
                "center_region": 1 / 3 <= rx < 2 / 3 and 1 / 3 <= ry < 2 / 3,
                "crop_path": "crops/" + crop_name, "crop_left_px": left, "crop_top_px": top,
                "crop_right_px": right, "crop_bottom_px": bottom,
                "overlap_sc_count": 0, "overlap_o_count": 0, "overlap_s_count": 0,
                "max_other_bbox_iou": 0.0,
            }
            for other_ann in other:
                score = iou(ann["bbox"], other_ann["bbox"])
                if score > 0:
                    cls = categories[other_ann["category_id"]]
                    row["overlap_" + cls + "_count"] += 1
                    row["max_other_bbox_iou"] = max(row["max_other_bbox_iou"], score)
                    overlap_pairs[cls] += 1
                    overlap_ids[cls].add(ann["id"])
                    overlap_image_ids[cls].add(image_id)
                    overlap_ious[cls].append(score)
            particles.append(row)
    require(len(particles) == 5444 and len({r["image_id"] for r in particles}) == 620,
            "Particle source counts differ")
    grouped = defaultdict(list)
    for row in particles:
        grouped[row["image_id"]].append(row)
    image_rows = []
    for image_id, image in sorted(images.items()):
        mapping = by_id[image_id]
        rows = grouped[image_id]
        area = [r["bbox_area"] for r in rows]
        ratios = [r["aspect_ratio"] for r in rows]
        image_rows.append({
            "image_id": image_id, "filename": image["file_name"],
            "original_filename": mapping["original_filename"],
            "split": mapping["split"], "wafer": mapping["wafer"] or "UNKNOWN",
            "exposure": mapping["exposure"] or "UNKNOWN", "position": mapping["position"] or "UNKNOWN",
            "particle_count": len(rows),
            "mean_particle_area": statistics.mean(area) if area else None,
            "median_particle_area": statistics.median(area) if area else None,
            "max_particle_area": max(area) if area else None,
            "mean_aspect_ratio": statistics.mean(ratios) if ratios else None,
            "max_aspect_ratio": max(ratios) if ratios else None,
            "o_count": int(mapping["o_count"]), "s_count": int(mapping["s_count"]),
            "sc_count": int(mapping["sc_count"]),
        })
    overlaps = {c: {
        "positive_iou_pair_count": overlap_pairs[c], "unique_particle_count": len(overlap_ids[c]),
        "image_count": len(overlap_image_ids[c]), "iou_statistics": describe(overlap_ious[c]),
    } for c in ("sc", "o", "s")}
    return particles, image_rows, grouped, {
        "exclusive_image_counts": {k: cooccurrence[k] for k in
                                   ("p_only", "p_plus_sc", "p_plus_o", "p_plus_s", "p_multiple_defects")},
        "inclusive_image_counts": {k: inclusive[k] for k in ("p_plus_sc", "p_plus_o", "p_plus_s")},
        "overlap_definition": "Axis-aligned bbox IoU > 0; edge-only touch is not overlap. Count annotation pairs, not physical-object identity.",
        "overlap_by_other_class": overlaps, "total_overlap_pairs": sum(overlap_pairs.values()),
        "unique_particles_with_any_other_overlap": sum(r["max_other_bbox_iou"] > 0 for r in particles),
        "images_with_any_other_overlap": len({r["image_id"] for r in particles if r["max_other_bbox_iou"] > 0}),
    }


def metadata_groups(particles, image_rows):
    result = {}
    for field in ("split", "wafer", "position", "exposure"):
        entries = {}
        for value in sorted({r[field] for r in image_rows}):
            ims = [r for r in image_rows if r[field] == value]
            anns = [r for r in particles if r[field] == value]
            entries[value] = {
                "images": len(ims), "particle_images": sum(r["particle_count"] > 0 for r in ims),
                "particle_annotations": len(anns),
                "count_per_image": describe(r["particle_count"] for r in ims),
                "bbox_area_px2": describe(r["bbox_area"] for r in anns),
                "aspect_ratio": describe(r["aspect_ratio"] for r in anns),
            }
        result[field] = entries
    return result


def samples(particles, image_rows):
    n = 12
    tie = lambda r: (r["image_id"], r["annotation_id"])
    rng = random.Random(SEED)
    selections = {
        "smallest": sorted(particles, key=lambda r: (r["bbox_area"], *tie(r)))[:n],
        "largest": sorted(particles, key=lambda r: (-r["bbox_area"], *tie(r)))[:n],
        "low_aspect_ratio": sorted(particles, key=lambda r: (r["aspect_ratio"], *tie(r)))[:n],
        "high_aspect_ratio": sorted(particles, key=lambda r: (-r["aspect_ratio"], *tie(r)))[:n],
        "near_square_bbox": sorted(particles, key=lambda r: (abs(math.log(r["aspect_ratio"])), *tie(r)))[:n],
        "random_seed42": rng.sample(particles, n),
    }
    high_count = sorted(image_rows, key=lambda r: (-r["particle_count"], r["image_id"]))[:8]
    return selections, high_count


def font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_sheet(title, records, full_images=None):
    columns = 4
    tile_w, tile_h = (400, 310) if full_images is not None else (320, 280)
    header, gap = 48, 12
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * tile_w, header + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black", font=font(20))
    for index, row in enumerate(records):
        x, y = (index % columns) * tile_w, header + (index // columns) * tile_h
        if full_images is None:
            source = Image.open(OUT / row["crop_path"])
            label = f"image {row['image_id']} / ann {row['annotation_id']}\n{row['bbox_width']:.2f} x {row['bbox_height']:.2f} px; AR {row['aspect_ratio']:.3f}\narea {row['bbox_area']:.2f} px2"
            label += "\n" + "\n".join(textwrap.wrap(row["original_filename"], 38)[:2])
        else:
            source = full_images[row["image_id"]].copy()
            source_draw = ImageDraw.Draw(source)
            for ann in row["_particles"]:
                source_draw.rectangle((ann["bbox_x"], ann["bbox_y"],
                                       ann["bbox_x"] + ann["bbox_width"],
                                       ann["bbox_y"] + ann["bbox_height"]), outline="cyan", width=3)
            label = f"image {row['image_id']}; p count {row['particle_count']}\n"
            label += "\n".join(textwrap.wrap(row["original_filename"], 43)[:2])
        available = (tile_w - 2 * gap, tile_h - 100)
        scale = min(available[0] / source.width, available[1] / source.height)
        size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        # Display-only scale. Native crop files are NEVER resized.
        preview = source.resize(size, Image.Resampling.NEAREST)
        canvas.paste(preview, (x + (tile_w - size[0]) // 2, y + 92))
        draw.text((x + gap, y + 4), label, fill="black", font=font(14))
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#cccccc")
        source.close()
    canvas.save(OUT / "contact_sheets" / (title.split(" | ")[0] + ".png"))


def crops_and_sheets(source, images, by_id, grouped, selections, high_count):
    (OUT / "crops").mkdir()
    (OUT / "contact_sheets").mkdir()
    high_ids = {r["image_id"] for r in high_count}
    full_images = {}
    verified = 0
    with zipfile.ZipFile(source) as archive:
        members = defaultdict(list)
        for item in archive.infolist():
            if not item.is_dir():
                members[PurePosixPath(item.filename).name].append(item.filename)
        for image_id in sorted(images):
            filename = images[image_id]["file_name"]
            require(len(members[filename]) == 1, "Missing/ambiguous source image")
            if not grouped[image_id]:
                continue
            raw = archive.read(members[filename][0])
            require(hashlib.sha256(raw).hexdigest() == by_id[image_id]["sha256"], "Source image hash changed")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                require(image.size == (images[image_id]["width"], images[image_id]["height"]), "Actual size differs")
                if image_id in high_ids:
                    full_images[image_id] = image.copy()
                for row in grouped[image_id]:
                    box = tuple(row[k] for k in ("crop_left_px", "crop_top_px", "crop_right_px", "crop_bottom_px"))
                    crop = image.crop(box)
                    path = OUT / row["crop_path"]
                    require(not path.exists(), "Crop filename collision")
                    crop.save(path, format="PNG")
                    with Image.open(path) as saved:
                        require(saved.size == crop.size and saved.mode == crop.mode and saved.tobytes() == crop.tobytes(),
                                "Crop changed decoded source pixels")
                    verified += 1
        for name, records in selections.items():
            render_sheet(name + " | display-scaled; native crops unchanged", records)
        for row in high_count:
            row["_particles"] = grouped[row["image_id"]]
        render_sheet("high_particle_count_images | existing p bbox overlays", high_count, full_images)
        for row in high_count:
            row.pop("_particles")
    for image in full_images.values():
        image.close()
    require(verified == 5444 and len(list((OUT / "crops").glob("*.png"))) == 5444, "Crop count mismatch")
    return verified


def plots(particles, image_rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, field, label in zip(axes.flat,
        ("bbox_width", "bbox_height", "bbox_area", "normalized_area"),
        ("BBox width (px)", "BBox height (px)", "BBox area (px2)", "BBox area / image area")):
        values = np.array([r[field] for r in particles])
        bins = np.geomspace(values.min(), values.max(), 41)
        ax.hist(values, bins=bins, color="#286c8e", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set(xlabel=label + " [log axis]", ylabel="Particle annotations")
    fig.suptitle("Particle bbox sizes (not physical size; not polygon area)")
    fig.savefig(OUT / "bbox_size_distribution.png", dpi=150)
    plt.close(fig)
    values = np.array([r["aspect_ratio"] for r in particles])
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.hist(values, bins=np.geomspace(values.min(), values.max(), 51), color="#286c8e")
    ax.set_xscale("log")
    ax.axvline(1, color="#c46b20", linestyle="--", label="width / height = 1")
    ax.set(xlabel="BBox aspect ratio width/height [log axis]", ylabel="Particle annotations",
           title="BBox aspect ratio (not a morphology label)")
    ax.legend()
    fig.savefig(OUT / "aspect_ratio_distribution.png", dpi=150)
    plt.close(fig)
    counts = [r["particle_count"] for r in image_rows]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.hist(counts, bins=np.arange(-0.5, max(counts) + 1.5), color="#286c8e")
    ax.set(xlabel="Particle annotations per image", ylabel="Images",
           title="All 629 images (including 9 with no particle annotation)")
    fig.savefig(OUT / "particle_count_per_image.png", dpi=150)
    plt.close(fig)
    hist, _, _ = np.histogram2d([r["relative_center_x"] for r in particles],
                               [r["relative_center_y"] for r in particles],
                               bins=20, range=((0, 1), (0, 1)))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    heat = ax.imshow(hist.T, extent=(0, 1, 1, 0), origin="upper", cmap="magma")
    for edge in (1/3, 2/3):
        ax.axvline(edge, color="white", linestyle="--", linewidth=0.6)
        ax.axhline(edge, color="white", linestyle="--", linewidth=0.6)
    ax.set(xlabel="Center x / image width", ylabel="Center y / image height",
           title="Particle bbox centers (20 x 20 bins; y=0 at image top)")
    fig.colorbar(heat, ax=ax, label="Particle annotation count")
    fig.savefig(OUT / "particle_position_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    require(iou((0, 0, 2, 2), (2, 0, 2, 2)) == 0, "IoU touch helper failure")
    require(iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1, "IoU identical helper failure")
    require(abs(iou((0, 0, 2, 2), (1, 1, 2, 2)) - 1/7) < 1e-15, "IoU helper failure")
    source, coco, images, by_id, categories, per_image, protected, provenance = load()
    particles, image_rows, grouped, cooccurrence = measure(coco, images, by_id, categories, per_image)
    selections, high_count = samples(particles, image_rows)
    counts = [r["particle_count"] for r in image_rows]
    bins = {"0": sum(n == 0 for n in counts), "1": sum(n == 1 for n in counts),
            "2-5": sum(2 <= n <= 5 for n in counts), "6-10": sum(6 <= n <= 10 for n in counts),
            "11-20": sum(11 <= n <= 20 for n in counts), "21+": sum(n >= 21 for n in counts)}
    summary = {
        "source": provenance, "total_images": 629, "particle_annotation_count": len(particles),
        "particle_image_count": 620, "images_without_particle": 9,
        "image_count_statistics_all_629": describe(counts),
        "image_count_statistics_particle_images_only": describe(n for n in counts if n > 0),
        "particle_count_image_distribution": bins,
        "exact_particle_count_image_distribution": dict(sorted(Counter(counts).items())),
        "bbox_statistics": {field: describe(r[field] for r in particles) for field in GEOMETRY},
        "spatial": {
            "horizontal_thirds": {k: fraction_record(sum(r["horizontal_region"] == k for r in particles))
                                  for k in ("left", "center", "right")},
            "vertical_thirds": {k: fraction_record(sum(r["vertical_region"] == k for r in particles))
                                for k in ("top", "center", "bottom")},
            "edge_near": fraction_record(sum(r["edge_near_center"] for r in particles)),
            "center_region": fraction_record(sum(r["center_region"] for r in particles)),
            "definitions": {
                "thirds": "[0,1/3), [1/3,2/3), [2/3,1]; based on bbox center, not object extent",
                "edge_near": "min(relative center x, 1-x, relative center y, 1-y) <= 0.10",
                "center_region": "1/3 <= relative center x < 2/3 AND 1/3 <= relative center y < 2/3",
                "edge_region_image_area_fraction": 0.36,
                "center_region_image_area_fraction": 1/9,
                "purpose": "Descriptive spatial bins only, never defect subtype or operational threshold",
            },
        },
        "metadata": metadata_groups(particles, image_rows), "cooccurrence": cooccurrence,
        "sample_selection": {
            "seed": SEED, "per_crop_sheet": 12, "high_count_images": 8,
            "rules": "Area/width-to-height rankings; near-square ranking abs(log(aspect_ratio)); ties by image/annotation ID. Random sample seed 42. No subtype labels.",
            "annotation_ids": {k: [r["annotation_id"] for r in v] for k, v in selections.items()},
            "high_count_image_ids": [r["image_id"] for r in high_count],
        },
        "methods": {
            "std": "Population standard deviation (ddof=0)",
            "percentiles": "NumPy linear interpolation; 25/50/75/90/95/99",
            "bbox_area": "COCO bbox width * height; not segmentation area, not physical units",
            "crop": "Bbox-tight, outward integer rounding: floor(x/y), ceil(x+w/y+h). No context margin, no clipping, native pixels saved losslessly as PNG.",
            "contact_sheets": "Display-only nearest-neighbor scaling to fit preview cells. Individual crop PNGs never resized.",
            "exposure": "Fixed manifest observed value only; blank becomes UNKNOWN",
            "no_particle_geometry": "Blank CSV fields / JSON null, never artificial zero size",
        },
    }
    require(sum(bins.values()) == 629 and sum(counts) == 5444, "Count reconciliation failed")
    require(sum(cooccurrence["exclusive_image_counts"].values()) == 620, "Co-occurrence reconciliation failed")
    OUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUT / "particle_manifest.csv", particles, list(particles[0]))
    write_csv(OUT / "particle_per_image.csv", image_rows, list(image_rows[0]))
    print("CSV_WRITTEN particle=5444; image=629", flush=True)
    verified = crops_and_sheets(source, images, by_id, grouped, selections, high_count)
    print(f"CROPS_VERIFIED {verified}", flush=True)
    plots(particles, image_rows)
    require(all(sha(p) == value for p, value in protected.items()), "Protected source/audit/split changed")
    with (OUT / "particle_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        readback = list(csv.DictReader(stream))
    with (OUT / "particle_per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
        image_readback = list(csv.DictReader(stream))
    require(len(readback) == len({r["annotation_id"] for r in readback}) == 5444, "Manifest readback mismatch")
    require(len(image_readback) == len({r["image_id"] for r in image_readback}) == 629, "Image readback mismatch")
    require(sum(int(r["particle_count"]) for r in image_readback) == 5444, "CSV count mismatch")
    summary["qa"] = {"source_audit_split_unchanged": True, "particle_csv_rows": 5444,
                     "per_image_csv_rows": 629, "duplicate_annotation_ids": 0,
                     "native_crop_png_count": verified, "crop_pixel_equality_verified": verified,
                     "contact_sheet_count": len(list((OUT / "contact_sheets").glob("*.png"))),
                     "unresolved_exposure_images": sum(r["exposure"] == "UNKNOWN" for r in image_rows),
                     "subtype_labels_created": False, "training_executed": False}
    (OUT / "particle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_reports(summary, particles, image_rows)
    print(json.dumps({k: summary[k] for k in ("particle_annotation_count", "particle_image_count",
        "image_count_statistics_all_629", "bbox_statistics", "spatial", "cooccurrence", "qa")},
        ensure_ascii=False, indent=2), flush=True)


def write_reports(summary, particles, image_rows):
    def fmt(value):
        return f"{value:.3f}"

    count = summary["image_count_statistics_all_629"]
    positive = summary["image_count_statistics_particle_images_only"]
    lines = [
        "# Particle analysis v1", "",
        "## Dataset", "",
        f"- Particle bbox annotation: {len(particles):,}; 포함 이미지: 620 / 629; p 없음: 9장.",
        "- Class는 COCO category name p로 선택. bbox만 분석하며 기존 class/annotation은 변경하지 않음.",
        "- p 없는 9장을 모두 정상으로 해석하지 않음. 전체 zero-label 정상 negative는 별도 확인된 6장.", "",
        "## Count distribution", "",
        "| 대상 | Min | Max | Mean | Median | Std (population) |",
        "|---|---:|---:|---:|---:|---:|",
        "| 전체 629장 | " + " | ".join(fmt(count[k]) for k in
            ("min", "max", "mean", "median", "std_population")) + " |",
        "| p 포함 620장 | " + " | ".join(fmt(positive[k]) for k in
            ("min", "max", "mean", "median", "std_population")) + " |",
        "",
        "전체 이미지 percentile: " + ", ".join(
            f"P{q}={count['percentiles'][str(q)]:.2f}" for q in PERCENTILES) + ".", "",
        "| 이미지당 p 개수 | 이미지 수 |", "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary["particle_count_image_distribution"].items()]
    lines += ["", "## Size / Shape", "",
              "| BBox 지표 | Min | Max | Mean | Median | Std (population) | P25 | P75 | P90 | P95 | P99 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for field, label in (("bbox_width", "Width (px)"), ("bbox_height", "Height (px)"),
                         ("bbox_area", "Area (px2)"), ("aspect_ratio", "Width/height")):
        stat = summary["bbox_statistics"][field]
        values = [stat[k] for k in ("min", "max", "mean", "median", "std_population")]
        values += [stat["percentiles"][str(q)] for q in (25, 75, 90, 95, 99)]
        lines.append("| " + label + " | " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "Area는 width×height이며 실제 object의 mask 면적이 아님. 단위는 pixel이며 calibration을 적용하지 않음.",
              "전체 normalized width/height/area 및 중심좌표 통계는 particle_summary.json에 기록.",
              "smallest/largest 및 low/high aspect ratio의 실제 annotation을 contact sheet로 제공.",
              "정사각형 bbox에 가까운 샘플은 abs(log(width/height))가 작은 순서로 제공하며 subtype이 아님.", "",
              "## Position", "",
              "BBox 중심 기준의 통계이며 object가 실제 이미지 경계와 접하는지에 대한 판정은 아님.",
              "좌우/상하를 동일 3등분. Edge-near는 바깥 10%, center region은 중앙 1/3×1/3.", "",
              "| 영역 | Particle 수 | 비율 |", "|---|---:|---:|"]
    for axis in ("horizontal_thirds", "vertical_thirds"):
        for name, value in summary["spatial"][axis].items():
            lines.append(f"| {axis}/{name} | {value['count']} | {value['fraction']:.2%} |")
    for name in ("edge_near", "center_region"):
        value = summary["spatial"][name]
        lines.append(f"| {name} | {value['count']} | {value['fraction']:.2%} |")
    lines += ["", "Edge-near 영역은 전체 면적의 36%, center region은 11.11%이므로 단순 개수만으로 집중/원인을 단정하지 않음.",
              "Heatmap은 정규화된 중심좌표의 20×20 bin count이며 image y=0이 위쪽.", "",
              "## Metadata", ""]
    for field, entries in summary["metadata"].items():
        lines += [f"### {field}", "", "| 값 | 전체 이미지 | p 포함 이미지 | p annotation | p/image mean |",
                  "|---|---:|---:|---:|---:|"]
        for name, value in entries.items():
            lines.append(f"| {name} | {value['images']} | {value['particle_images']} | "
                         f"{value['particle_annotations']} | {value['count_per_image']['mean']:.3f} |")
        lines.append("")
    lines += ["Exposure는 확인된 216장만 60/70 값을 사용하며 결측 413장은 UNKNOWN. 다른 metadata 결측도 추정하지 않음.",
              "각 metadata group의 area/aspect ratio 통계는 JSON에 기록.", "",
              "## Co-occurrence", "",
              "아래 표는 p 포함 이미지 620장을 배타적으로 구분한 image-level 동시 존재 현황.", "",
              "| Class 조합 | 이미지 수 |", "|---|---:|"]
    co = summary["cooccurrence"]
    lines += [f"| {k} | {v} |" for k, v in co["exclusive_image_counts"].items()]
    lines += ["", "Inclusive 동시 존재(중복 집계 가능): " +
              ", ".join(f"{k}={v}" for k, v in co["inclusive_image_counts"].items()) + ".", "",
              "Overlap은 같은 이미지의 p bbox와 다른 defect bbox 간 IoU>0인 annotation pair 수.",
              "Edge-only touch는 제외. BBox overlap은 실제 object overlap이나 동일 object를 증명하지 않음.", "",
              "| 다른 class | IoU>0 pair | 해당 p 고유 수 | 이미지 수 |",
              "|---|---:|---:|---:|"]
    for cls, value in co["overlap_by_other_class"].items():
        lines.append(f"| {cls} | {value['positive_iou_pair_count']} | "
                     f"{value['unique_particle_count']} | {value['image_count']} |")
    lines += ["", f"전체 overlap pair={co['total_overlap_pairs']}; 다른 class와 겹치는 고유 p="
              f"{co['unique_particles_with_any_other_overlap']}; 해당 이미지={co['images_with_any_other_overlap']}.",
              "", "## 다음 단계 참고", "",
              "- Size/AR의 연속적 분포와 extreme bbox는 수치·원본 crop을 함께 확인할 근거로만 사용.",
              "- BBox margin, 작은 원본 pixel 수, 주변 패턴 포함이 실제 형태와 어떻게 다른지는 육안 확인 필요.",
              "- 형태 차이의 육안 구분 가능 여부는 제공된 native crop/contact sheet에서 확인할 항목이며, 본 분석에서 subtype 가능성을 확정하지 않음.",
              "- Subtype, cluster, 분류 경계, 새 annotation을 정의하지 않음.", ""]
    (OUT / "particle_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# Particle analysis v1", "",
        "COCO category name p의 bbox와 image metadata에 대한 descriptive audit.",
        "Subtype 정의, clustering, 재라벨링, classifier/detector 학습은 수행하지 않음.", "",
        "## Sources", "",
        "- COCO: " + summary["source"]["coco_archive"],
        "- Image metadata: data/derived/fixed_split_v1/split_manifest.csv",
        "- Existing audit: data/derived/gt_629_audit_v1/",
        "- Source ZIP, audit, fixed split은 SHA-256으로 전후 변경 없음 검증.", "",
        "## Results", "",
        "- particle_summary.json: 전체 수치, percentile, metadata group 통계, sample ID와 QA.",
        "- particle_analysis.md: 간단한 분석 보고서.",
        "- particle_manifest.csv: p annotation 5,444행, COCO bbox 원본 값과 계산값/원본 추적 정보.",
        "- particle_per_image.csv: 전체 이미지 629행. p=0인 9장 포함, geometry는 공란.",
        "- 4개 distribution PNG 및 contact_sheets/의 7개 시각적 검토용 sheet.",
        "- crops/: 5,444개 native-resolution lossless PNG.",
        "", "## Definitions", "",
        "- BBox area=width×height(px2); aspect_ratio=width/height. Object mask area/physical size가 아님.",
        "- Normalized width/height는 image width/height, area는 image area로 나눔.",
        "- Std는 population(ddof=0), percentile은 linear interpolation(P25/50/75/90/95/99).",
        "- Count 통계는 전체 629장 및 p 포함 620장을 구분.",
        "- 좌우/상하 3등분은 bbox center 기준. Outer 10%가 edge-near, 중앙 1/3×1/3이 center region.",
        "- 위치 bin은 설명용이며 subtype/operational threshold가 아님.",
        "- Exposure 결측은 UNKNOWN. Wafer/position/split도 기존 manifest만 사용.",
        "- Co-occurrence image 표는 배타적 조합. Inclusive 값은 별도로 중복 가능성을 명시.",
        "- BBox overlap=IoU>0인 다른 class와의 annotation pair. Boundary touch는 제외.", "",
        "## Crops and review samples", "",
        "- Bbox-tight 한 방식만 사용. floor(x/y), ceil(x+w/y+h)의 outward integer bounds.",
        "- Context margin, bbox clipping/correction, resize, augmentation, color correction 없음.",
        "- 원본 decoded pixels를 그대로 crop하여 PNG 저장; 5,444개 모두 저장 후 pixel equality 검증.",
        "- Crop filename은 image_id, annotation_id 및 original filename을 포함.",
        "- Manifest에 정확한 COCO bbox와 integer crop bounds를 별도로 기록하며 원본 bbox는 변경하지 않음.",
        "- Contact sheet에서만 display fit을 위해 nearest-neighbor scaling. 개별 crop 파일은 native 해상도 유지.",
        "- Area 최소/최대, AR 최소/최대, abs(log(AR)) 최소, seed42 random을 각 12개 선정.",
        "- p count 상위 이미지 8장을 별도 sheet에 원래 p bbox overlay로 표시.",
        "- Sample 그룹은 검토 순위일 뿐 subtype/class annotation이 아님.", "",
        "## Reproduction", "",
        ".venv/Scripts/python.exe -B scripts/analyze_particle_v1.py",
        "기존 v1 출력이 있으면 덮어쓰지 않고 중단. NumPy/Pillow/Matplotlib 및 표준 CSV만 사용.",
        "원본 image/annotation, 고정 split, production source/config/dashboard 변경 없음.",
        "학습, subtype 정의, clustering, augmentation, oversampling, synthetic, commit/push 없음.", "",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()

