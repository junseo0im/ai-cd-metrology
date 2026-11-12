"""COCO bbox-only export under immutable fixed split; never train or edit sources.

Run: .venv/Scripts/python.exe -B scripts/convert_coco_detection_fixed_v1.py
Recheck without writing: add --verify-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "data/derived/fixed_split_v1"
OUT = ROOT / "data/derived/yolov8_detection_fixed_v1"
SOURCE = ROOT / "data/incoming/roboflow_current/gt_629_raw/coco"
CLASS_IDS = {"o": 0, "p": 1, "s": 2, "sc": 3}
EXPECTED_IMAGES = {"train": 505, "val": 62, "test": 62}
EXPECTED_CLASSES = {
    "train": {"p": 4352, "o": 2, "s": 5, "sc": 399},
    "val": {"p": 546, "o": 0, "s": 1, "sc": 50},
    "test": {"p": 546, "o": 1, "s": 1, "sc": 50},
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode_bbox(bbox, width, height):
    x, y, w, h = map(float, bbox)
    require(all(math.isfinite(v) for v in (x, y, w, h)), "Non-finite source bbox")
    require(width > 0 and height > 0 and w > 0 and h > 0, "Nonpositive size")
    require(x >= 0 and y >= 0 and x + w <= width and y + h <= height,
            "Out-of-bounds source bbox; never clip or repair")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(args.verify_only or not OUT.exists(), "Output exists; never overwrite v1")
    manifest = FIXED / "split_manifest.csv"
    frozen = json.loads((FIXED / "split_summary.json").read_text(encoding="utf-8"))
    require(sha256(manifest) == frozen["split_manifest_sha256"], "Fixed manifest hash changed")
    protected = {p: sha256(p) for p in FIXED.iterdir() if p.is_file()}
    archives = sorted(SOURCE.glob("*.zip"))
    require(len(archives) == 1, "Expected exactly one authoritative COCO source ZIP")
    source = archives[0]
    protected[source] = sha256(source)
    expected_source_hash = frozen["source"]["input_sha256"][source.relative_to(ROOT).as_posix()]
    require(protected[source] == expected_source_hash, "Source differs from frozen source")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len({r["image_id"] for r in rows}) == 629, "Manifest identity mismatch")
    require(dict(Counter(r["split"] for r in rows)) == EXPECTED_IMAGES, "Unexpected split counts")
    require(len({r["filename"] for r in rows}) == 629, "Filename collision")
    require(len({Path(r["filename"]).stem for r in rows}) == 629, "Label filename collision")
    require(len({r["sha256"] for r in rows}) == 629, "Duplicate image hash")
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group_key"]].add(row["split"])
        require(Path(row["filename"]).name == row["filename"] and
                "/" not in row["filename"] and "\\" not in row["filename"],
                "Unsafe filename")
    require(all(len(v) == 1 for v in group_splits.values()), "Group leakage")

    with zipfile.ZipFile(source) as archive:
        json_members = [n for n in archive.namelist() if n.endswith("_annotations.coco.json")]
        require(len(json_members) == 1, "Expected one COCO JSON")
        coco_bytes = archive.read(json_members[0])
        coco = json.loads(coco_bytes)
        images = {str(im["id"]): im for im in coco["images"]}
        require(len(images) == len(coco["images"]) == 629, "COCO image count/IDs changed")
        require(set(images) == {r["image_id"] for r in rows}, "Missing/extra manifest identities")
        require(len(coco["annotations"]) == 5953, "COCO annotation total changed")
        require(len({ann["id"] for ann in coco["annotations"]}) == 5953, "Duplicate annotation ID")
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for ann in coco["annotations"]:
            require(str(ann["image_id"]) in images, "Unknown annotation image reference")
            require(categories.get(ann["category_id"]) in CLASS_IDS, "Unknown annotated class")
            by_image[str(ann["image_id"])].append(ann)
        members = defaultdict(list)
        for entry in archive.infolist():
            if not entry.is_dir():
                members[PurePosixPath(entry.filename).name].append(entry.filename)
        records = []
        split_counts = {s: Counter() for s in EXPECTED_IMAGES}
        zeros = Counter()
        # Validate all inputs before creating the output directory.
        for row in sorted(rows, key=lambda r: int(r["image_id"])):
            image = images[row["image_id"]]
            filename, split = row["filename"], row["split"]
            require(image["file_name"] == filename, "Manifest filename differs from COCO")
            require(image["extra"]["name"] == row["original_filename"], "Original identity mismatch")
            require(len(members[filename]) == 1, "Missing or ambiguous source image")
            member = members[filename][0]
            raw = archive.read(member)
            require(hashlib.sha256(raw).hexdigest() == row["sha256"], "Source image hash mismatch")
            with Image.open(io.BytesIO(raw)) as decoded:
                require(decoded.size == (image["width"], image["height"]),
                        "COCO dimensions differ from actual image")
            annotations = sorted(by_image[row["image_id"]], key=lambda a: a["id"])
            count = Counter()
            label_lines = []
            for ann in annotations:
                name = categories[ann["category_id"]]
                normalized = encode_bbox(ann["bbox"], image["width"], image["height"])
                label_lines.append(str(CLASS_IDS[name]) + " " +
                                   " ".join(format(v, ".17g") for v in normalized))
                count[name] += 1
            require(all(count[c] == int(row[c + "_count"]) for c in CLASS_IDS),
                    "Manifest/COCO per-image class count mismatch")
            require(sum(count.values()) == int(row["total_annotations"]), "Annotation count mismatch")
            require((not annotations) == (row["zero_annotation"] == "True"), "Negative identity mismatch")
            split_counts[split].update(count)
            zeros[split] += not annotations
            content = ("\n".join(label_lines) + ("\n" if label_lines else "")).encode("ascii")
            records.append((row, image, member, annotations, content))
        require({s: {c: split_counts[s][c] for c in CLASS_IDS} for s in split_counts}
                == EXPECTED_CLASSES, "Split class count mismatch")
        require(dict(zeros) == {"train": 4, "val": 1, "test": 1}, "Negative split changed")

        yaml_content = (
            "path: " + json.dumps(OUT.as_posix(), ensure_ascii=False) +
            "\ntrain: train/images\nval: val/images\ntest: test/images\n"
            "nc: 4\nnames:\n  0: o\n  1: p\n  2: s\n  3: sc\n"
        ).encode("utf-8")
        if not args.verify_only:
            OUT.mkdir(parents=True, exist_ok=False)
            for split in EXPECTED_IMAGES:
                for kind in ("images", "labels"):
                    (OUT / split / kind).mkdir(parents=True, exist_ok=False)
            for row, image, member, annotations, content in records:
                with (OUT / row["split"] / "images" / row["filename"]).open("xb") as stream:
                    stream.write(archive.read(member))
                with (OUT / row["split"] / "labels" / (Path(row["filename"]).stem + ".txt")).open("xb") as stream:
                    stream.write(content)
            with (OUT / "data.yaml").open("xb") as stream:
                stream.write(yaml_content)

        # Independently parse the files actually saved, including empty negative labels.
        actual_images, actual_labels = {}, {}
        actual_class_counts = {s: Counter() for s in EXPECTED_IMAGES}
        invalid_rows = polygon_rows = actual_annotations = 0
        max_error = 0.0
        for split in EXPECTED_IMAGES:
            image_files = list((OUT / split / "images").iterdir())
            label_files = list((OUT / split / "labels").iterdir())
            expected_names = {r["filename"] for r in rows if r["split"] == split}
            expected_labels = {Path(name).stem + ".txt" for name in expected_names}
            require({p.name for p in image_files} == expected_names, "Missing/extra/misassigned image")
            require({p.name for p in label_files} == expected_labels, "Missing/extra/misassigned label")
            actual_images[split], actual_labels[split] = len(image_files), len(label_files)
        for row, image, member, annotations, content in records:
            image_path = OUT / row["split"] / "images" / row["filename"]
            label_path = OUT / row["split"] / "labels" / (Path(row["filename"]).stem + ".txt")
            require(sha256(image_path) == row["sha256"], "Copied image bytes changed")
            require(label_path.read_bytes() == content, "Saved label differs from exact conversion")
            lines = label_path.read_text(encoding="ascii").splitlines()
            require(len(lines) == len(annotations), "Annotation missing or deleted")
            for line, ann in zip(lines, annotations):
                tokens = line.split()
                polygon_rows += len(tokens) > 5
                if len(tokens) != 5:
                    invalid_rows += 1
                    continue
                cls = int(tokens[0])
                cx, cy, nw, nh = map(float, tokens[1:])
                if (cls not in CLASS_IDS.values() or
                    not all(math.isfinite(v) for v in (cx, cy, nw, nh)) or
                    not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1)):
                    invalid_rows += 1
                require(cls == CLASS_IDS[categories[ann["category_id"]]], "Class-name mapping mismatch")
                w, h = nw * image["width"], nh * image["height"]
                restored = (cx * image["width"] - w / 2,
                            cy * image["height"] - h / 2, w, h)
                max_error = max(max_error, *(abs(a - b) for a, b in zip(restored, ann["bbox"])))
                actual_class_counts[row["split"]][categories[ann["category_id"]]] += 1
                actual_annotations += 1
        require(actual_annotations == 5953 and invalid_rows == polygon_rows == 0,
                "Detection label validation failed")
        require(max_error < 1e-8, "Round-trip error exceeds floating-point tolerance")
        require(actual_images == actual_labels == EXPECTED_IMAGES, "Output counts mismatch")
        require({s: {c: actual_class_counts[s][c] for c in CLASS_IDS} for s in actual_class_counts}
                == EXPECTED_CLASSES, "Saved class counts mismatch")
        require((OUT / "data.yaml").read_bytes() == yaml_content, "data.yaml changed")
        dataset_yaml = yaml.safe_load((OUT / "data.yaml").read_text(encoding="utf-8"))
        require(dataset_yaml["names"] == {v: k for k, v in CLASS_IDS.items()}, "YAML class order mismatch")
        for split in EXPECTED_IMAGES:
            require(Path(dataset_yaml["path"]) / dataset_yaml[split] == OUT / split / "images",
                    "YAML path mismatch")
        require(all(sha256(p) == value for p, value in protected.items()),
                "Source archive or fixed split changed")
        summary = {
            "source_path": source.relative_to(ROOT).as_posix(),
            "source_coco_json": json_members[0],
            "source_coco_json_sha256": hashlib.sha256(coco_bytes).hexdigest(),
            "source_archive_sha256": protected[source],
            "output_path": OUT.relative_to(ROOT).as_posix(),
            "fixed_split_manifest": manifest.relative_to(ROOT).as_posix(),
            "fixed_split_manifest_sha256": protected[manifest],
            "total_images": sum(actual_images.values()), "total_annotations": actual_annotations,
            "split_image_counts": actual_images, "split_label_counts": actual_labels,
            "split_class_counts": EXPECTED_CLASSES,
            "class_counts": {c: sum(actual_class_counts[s][c] for s in EXPECTED_IMAGES) for c in CLASS_IDS},
            "class_mapping_by_name": CLASS_IDS,
            "source_category_id_to_name": categories,
            "zero_label_count": sum(zeros.values()), "split_zero_label_counts": dict(zeros),
            "invalid_yolo_rows": invalid_rows, "polygon_rows": polygon_rows,
            "missing_images": 0, "missing_labels": 0, "split_mismatch": 0,
            "image_overlap": 0, "label_overlap": 0, "group_leakage": 0,
            "all_image_bytes_match_source_sha256": True,
            "max_bbox_round_trip_error_px": max_error,
            "round_trip_tolerance_px": 1e-8,
            "image_placement": "Byte-for-byte copy of COCO ZIP member; no decoding/re-encoding",
            "segmentation_used": False, "source_and_fixed_split_unchanged": True,
            "conversion_success": True, "training_executed": False,
            "label_precision": "17 significant digits; source annotation ID order within each image",
        }
        readme = "\n".join([
            "# YOLOv8 Detection fixed dataset v1", "",
            "COCO bbox [x,y,width,height] converted to exactly five-token YOLO Detection labels.",
            "Class mapping by category name: 0=o, 1=p, 2=s, 3=sc. No class merge or annotation removal.",
            "Segmentation/polygon data is not read for conversion and is not emitted.", "",
            "Source: " + summary["source_path"],
            "Assignment source: data/derived/fixed_split_v1/split_manifest.csv.",
            "The fixed split is unchanged: Train 505, Val 62, Test 62.",
            "Images are byte-for-byte copies from the source COCO ZIP. SHA-256 matches all 629 originals.",
            "No image resize, crop, rotation, recompression, or source modification occurred.", "",
            "Labels: x_center=(x+w/2)/W, y_center=(y+h/2)/H, norm_width=w/W, norm_height=h/H.",
            "Source bboxes are not clipped, repaired, moved or deleted. Output precision is 17 significant digits.",
            "All 629 images have one label TXT, including six empty human-confirmed normal negatives (4/1/1).",
            "All 5953 annotations retained: p=5444, o=3, s=7, sc=499.",
            "Invalid YOLO rows=0, polygon rows=0, missing images/labels=0, assignment mismatches=0.",
            "Cross-split image/label overlap and group leakage=0.",
            f"Maximum bbox round-trip error: {max_error:.17g} px (tolerance 1e-8 px).",
            "data.yaml points to this dataset and uses the fixed class order.", "",
            "Reproducible converter: scripts/convert_coco_detection_fixed_v1.py.",
            "Existing output is never overwritten. Use --verify-only to recheck without writes.",
            "No training, augmentation, oversampling or synthetic generation was performed.", "",
        ]).encode("utf-8")
        reports = {
            "conversion_summary.json": (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            "README.md": readme,
        }
        for name, content in reports.items():
            path = OUT / name
            if args.verify_only:
                require(path.read_bytes() == content, "Report differs from verified conversion: " + name)
            else:
                with path.open("xb") as stream:
                    stream.write(content)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

