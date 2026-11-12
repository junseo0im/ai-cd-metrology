"""Convert existing general_v2 COCO bboxes and verify the corrected_v2 comparison."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import yaml
import prepare_yolov8_corrected_v2 as corrected

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/incoming/augmentation_current/general_v2.zip"
OUT = ROOT/"data/derived/yolov8_detection_general_v2"
CORRECTED = ROOT/"data/derived/yolov8_detection_corrected_v2"
DRIVE = "/content/drive/MyDrive/ai-cd-metrology"
SOURCE_SHA256 = "9c9d5b31af2923b92a42a8f3e39caf5f6dbc90f00a2ca0c1b005ebd5e270a28d"
COUNTS = {"train":604, "val":67, "test":66}
CLASS_COUNTS = {
    "train":{"o":105,"p":5054,"s":120,"sc":511},
    "val":corrected.CLASS_COUNTS["val"], "test":corrected.CLASS_COUNTS["test"],
}
CLASSES = corrected.CLASSES
require, digest, json_bytes = corrected.require, corrected.digest, corrected.json_bytes


def plan(archive, original_archive):
    originals, _, original_cats, group_count = corrected.source_plan(original_archive)
    original_by_id = {r["image_id"]:r for r in originals}
    for r in originals:
        require(digest(CORRECTED/r["split"]/"images"/r["filename"]) == r["image_sha256"] and
                (CORRECTED/r["split"]/"labels"/(Path(r["filename"]).stem+".txt")).read_bytes() == r["_labels"],
                "Existing corrected YOLO dataset differs from its source")
    members = archive.namelist()
    require(len(members) == len(set(members)), "Duplicate ZIP member")
    aug_bytes = archive.read("general_v2/augmentation_manifest.json")
    aug = json.loads(aug_bytes)
    aug_by_id = {r["image_id"]:r for r in aug["records"]}
    require(len(aug_by_id) == len(aug["records"]) == 70 and aug["seed"] == 42 and
            aug["copies_per_selected_image"] == 2 and set(aug["focus"]) == {"o","s"}, "Augmentation manifest changed")
    parents = {r["parent_image_id"] for r in aug_by_id.values()}
    require(len(parents) == 35, "Expected exactly 35 existing augmentation parents")
    source_hashes = json.loads(archive.read("general_v2/dataset_sha256.json"))
    source_summary = json.loads(archive.read("general_v2/summary.json"))
    records, image_ids, annotation_ids = [], set(), set()
    hashes, pixels, groups, axes = {}, set(), defaultdict(set), defaultdict(list)
    zeros, raw_coco_comparison = {}, {}
    for source_split, split in corrected.SPLITS.items():
        member = "general_v2/"+source_split+"/_annotations.coco.json"
        raw_coco = archive.read(member)
        original_raw = original_archive.read("corrected_v2/"+source_split+"/_annotations.coco.json")
        if split in {"val","test"}:
            require(raw_coco == original_raw, "Val/Test COCO bytes differ")
            raw_coco_comparison[split] = hashlib.sha256(raw_coco).hexdigest()
        coco = json.loads(raw_coco)
        require(len(coco["images"]) == COUNTS[split], "Source split image count mismatch")
        cats = {c["id"]:c["name"] for c in coco["categories"]}
        require(len(cats) == 4 and set(cats.values()) == set(CLASSES), "Class mapping mismatch")
        annotations = defaultdict(list)
        image_map = {im["id"]:im for im in coco["images"]}
        require(len(image_map) == COUNTS[split], "Duplicate source image ID")
        for ann in coco["annotations"]:
            require(ann["id"] not in annotation_ids and ann["image_id"] in image_map and
                    ann["category_id"] in cats and ann.get("iscrowd",0) == 0, "Annotation identity/reference error")
            annotation_ids.add(ann["id"])
            annotations[ann["image_id"]].append(ann)
        total, zero, expected_members = Counter(), 0, set()
        for image in sorted(coco["images"],key=lambda r:r["id"]):
            iid, name = image["id"], image["file_name"]
            require(iid not in image_ids and Path(name).name == name and "/" not in name and "\\" not in name,
                    "Duplicate ID or unsafe filename")
            image_ids.add(iid)
            im_member = "general_v2/"+source_split+"/"+name
            expected_members.add(im_member)
            raw = archive.read(im_member)
            sha = hashlib.sha256(raw).hexdigest()
            require(source_hashes[source_split+"/"+name] == sha and sha not in hashes,
                    "Image checksum mismatch or exact duplicate")
            hashes[sha] = im_member
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.load()
                require(decoded.size == (image["width"],image["height"]) and decoded.getexif().get(274,1) == 1,
                        "Image geometry/EXIF mismatch")
                rgb = decoded.convert("RGB")
                pixel_sha = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                require(pixel_sha not in pixels, "Duplicate decoded image pixels")
                pixels.add(pixel_sha)
            ordered = sorted(annotations[iid],key=lambda r:r["id"])
            is_aug = iid not in original_by_id
            if not is_aug:
                r = original_by_id[iid]
                require(r["filename"] == name and r["split"] == split and r["image_sha256"] == sha and
                        r["_annotations"] == ordered and (r["width"],r["height"]) == (image["width"],image["height"]),
                        "Corrected original image, split or annotation changed")
                group, parent_id, parameters = r["group_key"], "", {}
            else:
                require(split == "train" and iid in aug_by_id, "Augmentation outside Train or unlisted image")
                a = aug_by_id[iid]
                parent_id, parameters = a["parent_image_id"], a["parameters"]
                parent = original_by_id[parent_id]
                require(parent["split"] == "train" and a["file_name"] == name and
                        image["extra"]["parent_image_id"] == parent_id and
                        (image["width"],image["height"]) == (parent["width"],parent["height"]),
                        "Augmentation parent or geometry mismatch")
                require(parent["o_count"]+parent["s_count"] > 0, "Non-o/s augmentation parent")
                h, v = parameters["horizontal"], parameters["vertical"]
                require(type(h) is bool and type(v) is bool and h != v and
                        .93 <= parameters["brightness"] <= 1.07 and .95 <= parameters["contrast"] <= 1.05,
                        "Existing augmentation parameters differ from audited ranges")
                require(all(image["extra"]["augmentation"][k] == val for k,val in parameters.items()),
                        "COCO/augmentation manifest parameters disagree")
                axes[parent_id].append((h,v))
                transformed = []
                for ann in parent["_annotations"]:
                    x,y,w,hh = ann["bbox"]
                    transformed.append((original_cats[parent["split"]][ann["category_id"]],
                                        parent["width"]-x-w if h else x,
                                        parent["height"]-y-hh if v else y,w,hh))
                require(Counter(transformed) == Counter((cats[a["category_id"]],*a["bbox"]) for a in ordered),
                        "Existing augmented bbox geometry does not match parent")
                group = parent["group_key"]
            require(image["extra"]["assigned_split"] == source_split and image["extra"]["split_group"] == group,
                    "Declared group/split changed")
            groups[group].add(split)
            lines, counts, seen = [], Counter(), set()
            for ann in ordered:
                cls = cats[ann["category_id"]]
                box = corrected.encode_bbox(ann["bbox"],image["width"],image["height"])
                item = (CLASSES[cls],*box)
                require(item not in seen, "Duplicate annotation bbox")
                seen.add(item)
                lines.append(str(CLASSES[cls])+" "+" ".join(format(x,".17g") for x in box))
                counts[cls] += 1
            content = ("\n".join(lines)+("\n" if lines else "")).encode("ascii")
            if not is_aug:
                require(content == original_by_id[iid]["_labels"], "Original YOLO label changed")
            total.update(counts)
            zero += not ordered
            records.append({
                "image_id":iid,"filename":name,"source_split":source_split,"split":split,"group_key":group,
                "parent_image_id":parent_id,"existing_augmentation":is_aug,"width":image["width"],"height":image["height"],
                "image_sha256":sha,"label_sha256":hashlib.sha256(content).hexdigest(),
                "o_count":counts["o"],"p_count":counts["p"],"s_count":counts["s"],"sc_count":counts["sc"],
                "annotation_count":len(ordered),"zero_annotation":not ordered,
                "_member":im_member,"_annotations":ordered,"_labels":content,
            })
        actual_members = {n for n in members if n.startswith("general_v2/"+source_split+"/") and
                          Path(n).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
        require(expected_members == actual_members, "Missing/extra ZIP image")
        require({c:total[c] for c in CLASSES} == CLASS_COUNTS[split] and
                source_summary[source_split]["annotations"] == sum(total.values()), "Source annotation count mismatch")
        zeros[split] = zero
    require(len(records) == len(image_ids) == len(hashes) == len(pixels) == 737 and len(annotation_ids) == 7089,
            "General dataset totals mismatch")
    require(image_ids-original_by_id.keys() == aug_by_id.keys() and set(axes) == parents and
            all(sorted(v) == [(False,True),(True,False)] for v in axes.values()), "Augmentation identity/axes mismatch")
    require(all(len(v) == 1 for v in groups.values()) and len(groups) == group_count == 134, "Declared group leakage")
    require(len({r["filename"] for r in records}) == len({Path(r["filename"]).stem for r in records}) == 737,
            "Output filename collision")
    require(zeros == {"train":5,"val":0,"test":1}, "Zero-label preservation mismatch")
    return records, zeros, aug_bytes, raw_coco_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    require(args.verify_only or not OUT.exists(), "Output exists; never overwrite")
    require(digest(SOURCE) == SOURCE_SHA256 and digest(corrected.SOURCE) == corrected.SOURCE_SHA256,
            "Source ZIP changed")
    corrected_script = ROOT/"scripts/train_yolov8_corrected_v2_colab.py"
    ref_path = CORRECTED/"baseline_reference.json"
    protected = {p:digest(p) for p in [SOURCE,corrected.SOURCE,ref_path,corrected_script,CORRECTED/"image_manifest.csv"]}
    with zipfile.ZipFile(SOURCE) as z,zipfile.ZipFile(corrected.SOURCE) as original:
        records,zeros,aug_bytes,coco_hashes = plan(z,original)
        if not args.verify_only:
            OUT.mkdir(parents=True,exist_ok=False)
            for split in COUNTS:
                for kind in ("images","labels"):
                    (OUT/split/kind).mkdir(parents=True)
            for r in records:
                (OUT/r["split"]/"images"/r["filename"]).write_bytes(z.read(r["_member"]))
                (OUT/r["split"]/"labels"/(Path(r["filename"]).stem+".txt")).write_bytes(r["_labels"])
    max_error = 0.0
    for split in COUNTS:
        names = {r["filename"] for r in records if r["split"] == split}
        require({p.name for p in (OUT/split/"images").iterdir()} == names and
                {p.name for p in (OUT/split/"labels").iterdir()} == {Path(n).stem+".txt" for n in names},
                "Output image/label completeness mismatch")
    for r in records:
        require(digest(OUT/r["split"]/"images"/r["filename"]) == r["image_sha256"] and
                (OUT/r["split"]/"labels"/(Path(r["filename"]).stem+".txt")).read_bytes() == r["_labels"],
                "Copied output differs")
        for line,ann in zip(r["_labels"].decode("ascii").splitlines(),r["_annotations"]):
            tokens = line.split()
            require(len(tokens) == 5,"Non-detection label")
            x,y,w,h = map(float,tokens[1:])
            restored = ((x-w/2)*r["width"],(y-h/2)*r["height"],w*r["width"],h*r["height"])
            max_error = max(max_error,*(abs(a-b) for a,b in zip(restored,ann["bbox"])))
    require(max_error < 1e-8,"BBox round-trip error")
    public = [{k:v for k,v in r.items() if not k.startswith("_")} for r in records]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=list(public[0]));writer.writeheader();writer.writerows(public)
    manifest = stream.getvalue().encode("utf-8-sig")
    old = json.loads(ref_path.read_text(encoding="utf-8"))
    ref = {k:old[k] for k in ["requested_train_args","effective_training_args","ultralytics_version","pretrained_weights_sha256"]}
    ref.update(corrected_script_sha256=protected[corrected_script],
               corrected_run="runs/v2_baseline/yolov8n_corrected_v2",
               completed_gpu_run_verification="REQUIRED_ON_COLAB: read actual experiment_info.json and args.yaml",
               expected_reported_gpu="Tesla T4",expected_reported_actual_epochs=55,expected_reported_best_epoch=35)
    ref_bytes = json_bytes(ref)
    original_manifest = (CORRECTED/"image_manifest.csv").read_bytes()
    comparison = {
        "corrected_originals_preserved":667,"corrected_train_preserved":534,"existing_augmented_images":70,
        "augmentation_parents":35,"augmentation_train_only":True,
        "val_images_identical":True,"test_images_identical":True,
        "val_labels_identical":True,"test_labels_identical":True,"val_test_coco_sha256":coco_hashes,
        "declared_group_leakage":0,"physical_source_identity_verified":False,
        "gpu_run_settings_verification":"Deferred to mandatory Colab preflight, as authorized by user",
    }
    summary = {
        "conversion_success":True,"dataset_qa":"PASS","training_executed":False,"source_sha256":SOURCE_SHA256,
        "total_images":737,"total_annotations":7089,"split_image_counts":COUNTS,"split_class_counts":CLASS_COUNTS,
        "class_mapping_by_name":CLASSES,"zero_label_counts":zeros,"zero_label_count":6,
        "invalid_bbox":0,"invalid_yolo_rows":0,"polygon_rows":0,"missing_images":0,"missing_labels":0,
        "split_mismatch":0,"exact_image_duplicates":0,"decoded_pixel_duplicates":0,"duplicate_annotation_rows":0,
        "declared_group_leakage":0,"max_bbox_round_trip_error_px":max_error,
        "image_manifest_sha256":hashlib.sha256(manifest).hexdigest(),
        "baseline_reference_sha256":hashlib.sha256(ref_bytes).hexdigest(),
        "corrected_manifest_sha256":hashlib.sha256(original_manifest).hexdigest(),
        "augmentation_manifest_sha256":hashlib.sha256(aug_bytes).hexdigest(),
        "comparison":comparison,"preparation_script_sha256":digest(Path(__file__)),
        "image_placement":"Exact existing ZIP member bytes, including existing augmentation; no image generation",
    }
    yaml_body = {"train":"train/images","val":"val/images","test":"test/images","nc":4,"names":{v:k for k,v in CLASSES.items()}}
    reports = {
        "image_manifest.csv":manifest,"baseline_reference.json":ref_bytes,
        "corrected_image_manifest.csv":original_manifest,"augmentation_manifest.json":aug_bytes,
        "conversion_summary.json":json_bytes(summary),"comparison_qa.json":json_bytes(comparison),
        "data.yaml":yaml.safe_dump({"path":OUT.as_posix(),**yaml_body},sort_keys=False).encode(),
        "data_colab.yaml":yaml.safe_dump({"path":DRIVE+"/datasets/yolov8_detection_general_v2",**yaml_body},sort_keys=False).encode(),
        "requirements_colab.txt":b"ultralytics==8.4.174\n",
    }
    readme = [
        "# general_v2 / Colab GPU preparation","",
        "Train604 / Val67 / Test66, 7089 bbox annotations; 0=o,1=p,2=s,3=sc.",
        "All corrected originals and labels preserved: Train534 plus existing Train-only augmentation70 (35 parents).",
        "Val/Test images, YOLO labels and original COCO JSON are byte-identical to corrected_v2.",
        "No augmentation generation, source edits, clipping, re-split, local training or evaluation.",
        "Six zero-label negatives are preserved. Exact/decoded duplicates, invalid/missing labels and declared group leakage0.",
        "Source grouping is filename-inferred; physical acquisition identity was not independently verified.","",
        "## Drive copy and execution","",
        "Copy this entire folder to "+DRIVE+"/datasets/yolov8_detection_general_v2.",
        "Copy scripts/train_yolov8_general_v2_colab.py to "+DRIVE+"/scripts/.",
        "Use the unchanged official outputs/pretrained/yolov8n.pt at "+DRIVE+"/models/yolov8n.pt.",
        "Keep the completed corrected run at "+DRIVE+"/runs/v2_baseline/yolov8n_corrected_v2/.",
        "Mandatory baseline artifacts: experiment_info.json, args.yaml, archived training script, results.csv and best.pt.",
        "The actual corrected GPU run exists on Drive only; its settings/environment are checked there before any training.",
        "Use the same Colab GPU/package environment. Preflight rejects version, GPU, optimizer or dataset mismatches.",
        "Mount Drive; install requirements_colab.txt without changing the corrected baseline's Torch/CUDA environment.",
        "Albumentations must remain absent, matching the archived corrected script's preflight.","",
        "```bash",
        "python "+DRIVE+"/scripts/train_yolov8_general_v2_colab.py --check-only",
        "python "+DRIVE+"/scripts/train_yolov8_general_v2_colab.py",
        "```","",
        "Output: "+DRIVE+"/runs/v2_augmented/yolov8n_general_v2/.",
        "No CPU fallback; pinned Ultralytics8.4.174, device0, AMPFalse and all online augmentation OFF.",
        "Train settings come from the completed corrected run, not from new defaults.",
        "After fitting, best.pt is evaluated once on Test66; Test is never used for model selection.",
        "Existing corrected Test metrics are compared only when a valid saved metric artifact is available; no re-evaluation.",
        "To reproduce this local conversion, keep scripts/prepare_yolov8_corrected_v2.py alongside the general converter.","",
    ]
    reports["README.md"] = "\n".join(readme).encode("utf-8")
    for name,content in reports.items():
        path = OUT/name
        if args.verify_only:
            require(path.read_bytes() == content,"Output metadata changed: "+name)
        else:
            with path.open("xb") as f:f.write(content)
    require(all(digest(p) == value for p,value in protected.items()),"Protected corrected/source artifact changed")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
