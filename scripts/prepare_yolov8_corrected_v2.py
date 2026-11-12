"""Prepare corrected_v2 COCO bboxes for YOLO Detection; never train or edit sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/incoming/augmentation_current/corrected_v2.zip"
OUT = ROOT / "data/derived/yolov8_detection_corrected_v2"
DRIVE = "/content/drive/MyDrive/ai-cd-metrology"
DRIVE_DATA = DRIVE + "/datasets/yolov8_detection_corrected_v2"
SOURCE_SHA256 = "890c9df87d04c952066fd4e9587a4ed93bdd8ccd591d7100d733ac7ff6ba0314"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
COUNTS = {"train": 534, "val": 67, "test": 66}
CLASSES = {"o": 0, "p": 1, "s": 2, "sc": 3}
CLASS_COUNTS = {
    "train": {"o": 35, "p": 4534, "s": 40, "sc": 427},
    "val": {"o": 4, "p": 593, "s": 4, "sc": 54},
    "test": {"o": 6, "p": 580, "s": 5, "sc": 53},
}
BASE = ROOT / "data/derived/yolov8_baseline_v1/baseline_summary.json"
AUG = ROOT / "data/derived/yolov8_v1_augmentation/augmentation_summary.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def encode_bbox(bbox, width, height):
    require(len(bbox) == 4, "Malformed source bbox")
    x, y, w, h = map(float, bbox)
    require(all(math.isfinite(v) for v in (x, y, w, h)), "Non-finite bbox")
    require(width > 0 and height > 0 and w > 0 and h > 0, "Nonpositive bbox/image size")
    require(x >= 0 and y >= 0 and x+w <= width and y+h <= height,
            "Out-of-bounds bbox; no clipping or repair is permitted")
    return ((x+w/2)/width, (y+h/2)/height, w/width, h/height)


def baseline_reference():
    old = json.loads(BASE.read_text(encoding="utf-8"))
    aug = json.loads(AUG.read_text(encoding="utf-8"))
    require(old["training_success"] and aug["training_success"], "Completed baselines required")
    ignored = {"data", "project", "name", "save_dir"}
    a, b = old["effective_training_args"], aug["effective_training_args"]
    require(all(a.get(k) == b.get(k) for k in (a.keys() | b.keys()) - ignored),
            "V1 baseline and V1 augmentation settings disagree")
    require(old["pretrained_weights_sha256"] == aug["pretrained_weights_sha256"],
            "Pretrained checkpoint changed")
    return {
        "reference_source": BASE.relative_to(ROOT).as_posix(),
        "reference_source_sha256": digest(BASE),
        "augmentation_crosscheck_sha256": digest(AUG),
        "ultralytics_version": old["environment"]["ultralytics"],
        "baseline_environment": old["environment"],
        "requested_train_args": old["requested_train_args"],
        "effective_training_args": old["effective_training_args"],
        "pretrained_weights_sha256": old["pretrained_weights_sha256"],
        "optimizer_policy": "Retain baseline optimizer=auto policy under the same pinned Ultralytics.",
        "expected_optimizer": {"name": "AdamW", "lr": 0.00125, "betas": [0.9, 0.999]},
        "note": "CPU to CUDA execution differs; identical seeds do not guarantee bitwise-identical results.",
    }


def source_plan(archive):
    members = archive.namelist()
    require(len(members) == len(set(members)), "Duplicate ZIP member")
    read = lambda name: archive.read("corrected_v2/" + name)
    manifest = list(csv.DictReader(io.StringIO(read("reports/split_manifest.csv").decode("utf-8-sig"),
                                               newline=None)))
    by_id = {int(row["image_id"]): row for row in manifest}
    require(len(by_id) == len(manifest) == 667, "Source split manifest identity/count error")
    file_hashes = json.loads(read("dataset_sha256.json"))
    summary = json.loads(read("summary.json"))
    records, image_ids, annotation_ids = [], set(), set()
    hashes, pixel_hashes, groups = defaultdict(list), defaultdict(list), defaultdict(set)
    zero_counts, category_mappings = {}, {}
    for original_split, split in SPLITS.items():
        coco_member = "corrected_v2/" + original_split + "/_annotations.coco.json"
        coco = json.loads(archive.read(coco_member))
        images = {im["id"]: im for im in coco["images"]}
        require(len(images) == len(coco["images"]) == COUNTS[split], "Source image count mismatch")
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        require(len(cats) == 4 and set(cats.values()) == set(CLASSES), "Unexpected source categories")
        category_mappings[split] = cats
        annotations = defaultdict(list)
        for ann in coco["annotations"]:
            require(ann["id"] not in annotation_ids, "Duplicate annotation ID")
            annotation_ids.add(ann["id"])
            require(ann["image_id"] in images and ann["category_id"] in cats, "Invalid annotation reference")
            require(ann.get("iscrowd", 0) == 0, "Crowd annotation needs explicit review")
            annotations[ann["image_id"]].append(ann)
        split_counts, zero = Counter(), 0
        expected_members = set()
        for image in sorted(images.values(), key=lambda im: im["id"]):
            iid, name = image["id"], image["file_name"]
            require(iid not in image_ids, "Duplicate image ID")
            image_ids.add(iid)
            require(Path(name).name == name and "/" not in name and "\\" not in name, "Unsafe image filename")
            row = by_id[iid]
            require(row["split"] == original_split and row["file_name"] == name, "Source split assignment mismatch")
            group = row["group_id"]
            require(group and image["extra"]["split_group"] == group and
                    image["extra"]["assigned_split"] == original_split, "Declared group mismatch")
            member = "corrected_v2/" + original_split + "/" + name
            expected_members.add(member)
            raw = archive.read(member)
            sha = hashlib.sha256(raw).hexdigest()
            require(file_hashes[original_split + "/" + name] == sha, "Source image SHA256 mismatch")
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.load()
                require(decoded.size == (image["width"], image["height"]), "COCO/actual size mismatch")
                require(decoded.getexif().get(274, 1) == 1, "EXIF orientation needs explicit review")
                rgb = decoded.convert("RGB")
                pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            hashes[sha].append(member)
            pixel_hashes[pixel_sha].append(member)
            groups[group].add(split)
            lines, counts, normalized_seen = [], Counter(), set()
            ordered = sorted(annotations[iid], key=lambda ann: ann["id"])
            for ann in ordered:
                cls = cats[ann["category_id"]]
                box = encode_bbox(ann["bbox"], image["width"], image["height"])
                label_row = (CLASSES[cls], *box)
                require(label_row not in normalized_seen, "Exact duplicate bbox annotation")
                normalized_seen.add(label_row)
                lines.append(str(CLASSES[cls]) + " " + " ".join(format(v, ".17g") for v in box))
                counts[cls] += 1
            require(all(counts[c] == int(row[c+"_boxes"]) for c in CLASSES), "Source class manifest mismatch")
            content = ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")
            split_counts.update(counts)
            zero += not ordered
            records.append({
                "image_id": iid, "filename": name, "source_split": original_split, "split": split,
                "group_key": group, "identity_status": row["identity_status"],
                "width": image["width"], "height": image["height"],
                "image_sha256": sha, "label_sha256": hashlib.sha256(content).hexdigest(),
                "o_count": counts["o"], "p_count": counts["p"], "s_count": counts["s"], "sc_count": counts["sc"],
                "annotation_count": len(ordered), "zero_annotation": not ordered,
                "_member": member, "_annotations": ordered, "_labels": content,
            })
        actual_members = {n for n in members if n.startswith("corrected_v2/"+original_split+"/")
                          and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}
        require(actual_members == expected_members, "Missing or extra source image")
        require({c: split_counts[c] for c in CLASSES} == CLASS_COUNTS[split], "Source class counts mismatch")
        require(summary[original_split]["images"] == COUNTS[split] and
                summary[original_split]["annotations"] == sum(split_counts.values()), "Archive summary mismatch")
        zero_counts[split] = zero
    require(len(image_ids) == 667 and len(annotation_ids) == 6335, "Source totals mismatch")
    require(len(hashes) == len(pixel_hashes) == 667, "Duplicate image bytes or decoded pixels")
    require(all(len(v) == 1 for v in groups.values()), "Declared group leakage")
    require(len({r["filename"] for r in records}) == len({Path(r["filename"]).stem for r in records}) == 667,
            "Image or label filename collision")
    return records, zero_counts, category_mappings, len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Validate existing output without writing")
    args = parser.parse_args()
    require(args.verify_only or not OUT.exists(), "Output exists; never overwrite a dataset")
    require(digest(SOURCE) == SOURCE_SHA256, "corrected_v2 source ZIP changed")
    protected = {p: digest(p) for p in (SOURCE, BASE, AUG)}
    reference = baseline_reference()
    with zipfile.ZipFile(SOURCE) as archive:
        records, zeros, cats, group_count = source_plan(archive)
        # No output is created until every source and baseline check above has passed.
        if not args.verify_only:
            OUT.mkdir(parents=True, exist_ok=False)
            for split in COUNTS:
                for kind in ("images", "labels"):
                    (OUT/split/kind).mkdir(parents=True, exist_ok=False)
            for r in records:
                (OUT/r["split"]/"images"/r["filename"]).write_bytes(archive.read(r["_member"]))
                (OUT/r["split"]/"labels"/(Path(r["filename"]).stem+".txt")).write_bytes(r["_labels"])
        max_error = 0.0
        for split in COUNTS:
            expected = {r["filename"] for r in records if r["split"] == split}
            require({p.name for p in (OUT/split/"images").iterdir()} == expected, "Output image set mismatch")
            require({p.name for p in (OUT/split/"labels").iterdir()} ==
                    {Path(n).stem+".txt" for n in expected}, "Output label set mismatch")
        for r in records:
            image = OUT/r["split"]/"images"/r["filename"]
            label = OUT/r["split"]/"labels"/(Path(r["filename"]).stem+".txt")
            require(digest(image) == r["image_sha256"] and label.read_bytes() == r["_labels"], "Output bytes mismatch")
            for line, ann in zip(label.read_text(encoding="ascii").splitlines(), r["_annotations"]):
                tokens = line.split()
                require(len(tokens) == 5 and int(tokens[0]) in CLASSES.values(), "Non-detection label row")
                cx, cy, w, h = map(float, tokens[1:])
                require(all(math.isfinite(v) for v in (cx, cy, w, h)) and
                        0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1, "Invalid YOLO coordinates")
                restored = ((cx-w/2)*r["width"], (cy-h/2)*r["height"], w*r["width"], h*r["height"])
                max_error = max(max_error, *(abs(a-b) for a,b in zip(restored, ann["bbox"])))
        require(max_error < 1e-8, "BBox round-trip error")
    public = [{k:v for k,v in r.items() if not k.startswith("_")} for r in records]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(public[0]))
    writer.writeheader()
    writer.writerows(public)
    manifest = stream.getvalue().encode("utf-8-sig")
    base_ref = json_bytes(reference)
    yaml_data = {"train": "train/images", "val": "val/images", "test": "test/images",
                 "nc": 4, "names": {v:k for k,v in CLASSES.items()}}
    reports = {
        "image_manifest.csv": manifest,
        "baseline_reference.json": base_ref,
        "data.yaml": yaml.safe_dump({"path": OUT.as_posix(), **yaml_data}, sort_keys=False).encode("utf-8"),
        "data_colab.yaml": yaml.safe_dump({"path": DRIVE_DATA, **yaml_data}, sort_keys=False).encode("utf-8"),
        "requirements_colab.txt": ("ultralytics==" + reference["ultralytics_version"] + "\n").encode("ascii"),
    }
    qa = {
        "conversion_success": True, "dataset_qa": "PASS", "training_executed": False,
        "source_path": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": SOURCE_SHA256,
        "total_images": 667, "total_annotations": 6335, "split_image_counts": COUNTS,
        "split_class_counts": CLASS_COUNTS, "class_mapping_by_name": CLASSES,
        "source_category_id_to_name": cats, "zero_label_counts": zeros, "zero_label_count": sum(zeros.values()),
        "invalid_bbox": 0, "invalid_yolo_rows": 0, "polygon_rows": 0, "missing_images": 0, "missing_labels": 0,
        "split_mismatch": 0, "exact_image_duplicates": 0, "decoded_pixel_duplicates": 0,
        "duplicate_annotation_rows": 0, "declared_group_leakage": 0, "declared_group_count": group_count,
        "physical_source_identity_verified": False,
        "identity_note": "Source-provided filename-inferred groups; no independent acquisition identity claim.",
        "max_bbox_round_trip_error_px": max_error, "all_source_image_bytes_preserved": True,
        "image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "baseline_reference_sha256": hashlib.sha256(base_ref).hexdigest(),
        "preparation_script_sha256": digest(Path(__file__)),
        "placement": "Byte-for-byte copy of ZIP members; valid directory is renamed val without changing membership.",
        "segmentation_used": False, "additional_augmentation": False,
    }
    reports["conversion_summary.json"] = json_bytes(qa)
    readme = [
        "# corrected_v2 YOLO Detection / Colab preparation", "",
        "COCO category names -> 0=o, 1=p, 2=s, 3=sc. Bboxes only; polygons are not used.",
        "Original corrected_v2 split preserved: Train534 / Val67 / Test66, annotations6335.",
        "Six zero-annotation images are retained with empty TXT labels (Train5 / Val0 / Test1).",
        "Source pixels and annotations are never edited; no resizing, clipping, augmentation or re-splitting.",
        "QA PASS: missing/invalid/duplicate/group-crossing counts0. Bbox round-trip error <1e-8px.",
        "Declared groups are filename-inferred; physical acquisition identity was not independently verified.",
        "V2 split and labels differ from V1; V1 scores are not a controlled same-Test comparison.", "",
        "## Colab", "",
        "Copy this entire folder to " + DRIVE_DATA + ".",
        "Copy scripts/train_yolov8_corrected_v2_colab.py to " + DRIVE + "/scripts/.",
        "Copy outputs/pretrained/yolov8n.pt to " + DRIVE + "/models/yolov8n.pt.",
        "Mount Google Drive, select a CUDA GPU runtime, then run:",
        "```python", "from google.colab import drive", "drive.mount('/content/drive')", "```",
        "```bash",
        "pip install -r " + DRIVE_DATA + "/requirements_colab.txt",
        "# If Albumentations is installed, remove it to preserve the baseline's no-extra-augmentation condition:",
        "pip uninstall -y albumentations",
        "python " + DRIVE + "/scripts/train_yolov8_corrected_v2_colab.py --check-only",
        "python " + DRIVE + "/scripts/train_yolov8_corrected_v2_colab.py", "```", "",
        "data.yaml is local; the Colab script uses data_colab.yaml (no Windows paths).",
        "Training runs only when the user executes the saved script on CUDA device0; no CPU fallback.",
        "No training was performed during this preparation.",
        "Baseline settings, including optimizer=auto, AMP=False and all online augmentation OFF, are preserved.",
        "Pinned Ultralytics version and matching official initial weights are required.",
        "PyTorch/CUDA/GPU versions and actual optimizer groups are recorded; CPU/CUDA are not bitwise-equivalent.",
        "Run path: " + DRIVE + "/runs/v2_baseline/yolov8n_corrected_v2",
        "The training script validates Val during fitting; it does not run a separate Test evaluation.", "",
    ]
    reports["README.md"] = ("\n".join(readme)).encode("utf-8")
    for name, content in reports.items():
        path = OUT/name
        if args.verify_only:
            require(path.read_bytes() == content, "Metadata changed: " + name)
        else:
            with path.open("xb") as f:
                f.write(content)
    require(all(digest(p) == h for p,h in protected.items()), "Source/baseline changed")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
