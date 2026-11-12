"""E3 combined augmentation from E0 originals; deterministic rejection, validation-only experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/derived/yolov8_detection_corrected_v2"
E1 = ROOT/"data/derived/yolov8_detection_general_v2"
E2 = ROOT/"data/derived/yolov8_detection_geometric_v2"
OUT = ROOT/"data/derived/yolov8_detection_combined_v2"
NAMES = {0:"o",1:"p",2:"s",3:"sc"}
COUNTS = {"train":534,"val":67,"test":66}
FINAL_COUNTS = {"train":604,"val":67,"test":66}
SOURCE_MANIFEST_SHA256 = "f6f884d2079e9154e164da7a4548f08dcca23f86fab5cfe8268a8b08808fde9f"
E1_AUG_MANIFEST_SHA256 = "2f537e1db78385cb0671a3965f97a36e27982a524adad0e520dd34f035b9e4c7"
E2_AUG_MANIFEST_SHA256 = "4650f2fd909672b2ef3252c786610160e4e8b9ff3f000d565a837cb9420306f0"
SEED = 42
MAX_ATTEMPTS = 100  # Initial proposal included; retry_count is accepted_attempt - 1.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+"\n").encode("utf-8")


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def labels(path):
    rows = []
    for index,line in enumerate(path.read_text(encoding="ascii").splitlines(),1):
        tokens = line.split()
        require(len(tokens) == 5,"Non-detection label")
        cid = int(tokens[0])
        x,y,w,h = map(float,tokens[1:])
        require(cid in NAMES and str(cid) == tokens[0] and all(math.isfinite(v) for v in [x,y,w,h]) and
                0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                x-w/2 >= -1e-12 and y-h/2 >= -1e-12 and x+w/2 <= 1+1e-12 and y+h/2 <= 1+1e-12,
                "Invalid original label")
        rows.append((index,cid,x,y,w,h))
    return rows


def matrix_for(width, height, angle, tx, ty):
    require(all(math.isfinite(v) for v in [width,height,angle,tx,ty]) and width > 0 and height > 0,
            "Non-finite affine parameter")
    matrix = cv2.getRotationMatrix2D((width/2,height/2),angle,1.0)
    matrix[:,2] += [tx,ty]
    require(np.allclose(matrix[:,:2] @ matrix[:,:2].T,np.eye(2),atol=1e-12) and
            math.isclose(np.linalg.det(matrix[:,:2]),1.0,abs_tol=1e-12),
            "Scale/shear/reflection is forbidden")
    return matrix


def transform_bbox(row, width, height, matrix, flip_type):
    _,cid,x,y,w,h = row
    x0,x1,y0,y1 = (x-w/2)*width,(x+w/2)*width,(y-h/2)*height,(y+h/2)*height
    corners = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]],dtype=np.float64)
    require(flip_type in {"horizontal","vertical"},"Invalid flip type")
    flip = np.array([[-1,0,width],[0,1,0],[0,0,1]],dtype=np.float64) if flip_type == "horizontal" else \
           np.array([[1,0,0],[0,-1,height],[0,0,1]],dtype=np.float64)
    points = (corners @ flip.T) @ matrix.T
    require(np.isfinite(points).all(),"Non-finite transformed bbox")
    raw = np.array([points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()])
    clipped = np.clip(raw,[0,0,0,0],[width,height,width,height])
    valid = bool(np.isfinite(clipped).all() and 0 <= clipped[0] < clipped[2] <= width and
                 0 <= clipped[1] < clipped[3] <= height)
    result = None
    if valid:
        x0,y0,x1,y1 = clipped
        result = (cid,(x0+x1)/(2*width),(y0+y1)/(2*height),(x1-x0)/width,(y1-y0)/height)
    return result,raw.tolist(),clipped.tolist(),not np.array_equal(raw,clipped)

def source_data():
    require(digest(SOURCE/"image_manifest.csv") == SOURCE_MANIFEST_SHA256,"E0 manifest changed")
    require(digest(E1/"augmentation_manifest.json") == E1_AUG_MANIFEST_SHA256,"E1 parent manifest changed")
    with (SOURCE/"image_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        originals = list(csv.DictReader(f))
    require(len(originals) == len({r["image_id"] for r in originals}) == 667 and
            dict(Counter(r["split"] for r in originals)) == COUNTS,"E0 identity/split count mismatch")
    by_id = {int(r["image_id"]):r for r in originals}
    aug = json.loads((E1/"augmentation_manifest.json").read_text(encoding="utf-8-sig"))
    parents = sorted({r["parent_image_id"] for r in aug["records"]})
    require(len(aug["records"]) == 70 and len(parents) == 35 and
            all(by_id[i]["split"] == "train" for i in parents),"E1 parent set mismatch")
    require(all(sum(a["parent_image_id"] == i for a in aug["records"]) == 2 for i in parents),
            "E1 parent repeat count changed")
    class_counts = {s:Counter() for s in COUNTS}
    source_hashes,group_splits = {},defaultdict(set)
    for row in originals:
        image = SOURCE/row["split"]/"images"/row["filename"]
        label = SOURCE/row["split"]/"labels"/(Path(row["filename"]).stem+".txt")
        require(digest(image) == row["image_sha256"] and digest(label) == row["label_sha256"],
                "E0 image/label bytes changed")
        require(row["image_sha256"] not in source_hashes,"Exact duplicate E0 image")
        source_hashes[row["image_sha256"]] = row["filename"]
        parsed = labels(label)
        class_counts[row["split"]].update(NAMES[r[1]] for r in parsed)
        require(len(parsed) == int(row["annotation_count"]),"E0 label count mismatch")
        group_splits[row["group_key"]].add(row["split"])
        if row["split"] in {"val","test"}:
            require(digest(E1/row["split"]/"images"/row["filename"]) == row["image_sha256"] and
                    digest(E1/row["split"]/"labels"/label.name) == row["label_sha256"],
                    "E0/E1 Val/Test image/annotation differs")
    require(all(len(v) == 1 for v in group_splits.values()),"Original group leakage")
    for iid in parents:
        require(int(by_id[iid]["o_count"])+int(by_id[iid]["s_count"]) > 0,"Non-o/s E1 parent")
    require(digest(E2/"augmentation_manifest.json") == E2_AUG_MANIFEST_SHA256,"E2 parent manifest changed")
    e2_manifest = json.loads((E2/"augmentation_manifest.json").read_text(encoding="utf-8"))
    e2_parents = {r["parent_image_id"] for r in e2_manifest["records"]}
    require(len(e2_manifest["records"]) == 70 and e2_parents == set(parents),"E1/E2 parent set mismatch")
    for row in originals:
        if row["split"] in {"val","test"}:
            require(digest(E2/row["split"]/"images"/row["filename"]) == row["image_sha256"] and
                    digest(E2/row["split"]/"labels"/(Path(row["filename"]).stem+".txt")) == row["label_sha256"],
                    "E2 Val/Test differs from E0/E1")
    return originals,by_id,parents,{s:{n:c[n] for n in NAMES.values()} for s,c in class_counts.items()}


def build_plan(by_id, parents, max_attempts=MAX_ATTEMPTS):
    rng = random.Random(SEED)
    manifest,plans,rejected,failures = [],[],[],[]
    clipped_count,augmented_classes = 0,Counter()
    for iid in parents:
        parent = by_id[iid]
        width,height = int(parent["width"]),int(parent["height"])
        parent_labels = labels(SOURCE/"train/labels"/(Path(parent["filename"]).stem+".txt"))
        for variant in ("A","B"):
            flip_type = "horizontal" if variant == "A" else "vertical"
            for attempt in range(1,max_attempts+1):
                angle = rng.uniform(-5,5)
                while angle == 0:
                    angle = rng.uniform(-5,5)
                tx = 0.0 if variant == "A" else rng.uniform(-.03,.03)*width
                ty = 0.0 if variant == "A" else rng.uniform(-.03,.03)*height
                brightness,contrast = rng.uniform(.93,1.07),rng.uniform(.95,1.05)
                matrix = matrix_for(width,height,angle,tx,ty)
                info = {
                    "parent_filename":parent["filename"],"parent_image_id":iid,
                    "augmented_filename":f"combined_{iid}_{variant}.png","variant":variant,
                    "flip_type":flip_type,"brightness_factor":brightness,"contrast_factor":contrast,
                    "angle_deg":angle,"translate_x_px":tx,"translate_y_px":ty,
                    "retry_count":attempt-1,"rejected_transform_count":attempt-1,"seed":SEED,
                    "o_present":int(parent["o_count"]) > 0,"s_present":int(parent["s_count"]) > 0,
                    "annotation_count":len(parent_labels),"width":width,"height":height,
                    "transform_order":"flip -> affine -> brightness -> contrast",
                }
                transformed,errors,attempt_clipped = [],[],0
                for row in parent_labels:
                    try:
                        encoded,raw,clipped,did_clip = transform_bbox(row,width,height,matrix,flip_type)
                        require(encoded is not None,"nonpositive bbox after clipping")
                        require(encoded[0] == row[1] and all(math.isfinite(v) for v in encoded),
                                "Class mismatch or non-finite bbox")
                        transformed.append(encoded)
                        attempt_clipped += did_clip
                    except ValueError as error:
                        errors.append({"label_row":row[0],"class_id":row[1],"reason":str(error)})
                if errors or len(transformed) != len(parent_labels):
                    rejected.append({
                        "parent_filename":parent["filename"],"parent_image_id":iid,"variant":variant,
                        "attempt":attempt,"flip_type":flip_type,"angle_deg":angle,
                        "translate_x_px":tx,"translate_y_px":ty,
                        "brightness_factor":brightness,"contrast_factor":contrast,"seed":SEED,
                        "failure_reason":json.dumps(errors),
                    })
                    continue
                manifest.append(info)
                plans.append({"info":info,"matrix":matrix,"rows":transformed})
                clipped_count += attempt_clipped
                augmented_classes.update(NAMES[r[1]] for r in parent_labels)
                break
            else:
                failures.append({"parent_image_id":iid,"parent_filename":parent["filename"],
                                 "variant":variant,"attempts":max_attempts,"reason":"transform attempts exhausted"})
                return manifest,plans,rejected,failures,clipped_count,dict(augmented_classes)
    require(len(plans) == len(manifest) == 70,"Augmentation budget mismatch")
    return manifest,plans,rejected,failures,clipped_count,{n:augmented_classes[n] for n in NAMES.values()}


def combined_pixels(info, matrix):
    # Always start from E0 original pixels, never from E1/E2 augmented pixels.
    with Image.open(SOURCE/"train/images"/info["parent_filename"]) as image:
        require(image.size == (info["width"],info["height"]),"Parent dimensions changed")
        image = image.convert("RGB")
        image = ImageOps.mirror(image) if info["flip_type"] == "horizontal" else ImageOps.flip(image)
        affine = cv2.warpAffine(np.asarray(image),matrix,(info["width"],info["height"]),
                               flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    image = ImageEnhance.Brightness(Image.fromarray(affine)).enhance(info["brightness_factor"])
    image = ImageEnhance.Contrast(image).enhance(info["contrast_factor"])
    return np.asarray(image)

def materialize(originals, plans):
    for split in COUNTS:
        for kind in ("images","labels"):
            (OUT/split/kind).mkdir(parents=True)
    for r in originals:
        for kind,name in [("images",r["filename"]),("labels",Path(r["filename"]).stem+".txt")]:
            shutil.copy2(SOURCE/r["split"]/kind/name,OUT/r["split"]/kind/name)
    for plan in plans:
        info = plan["info"]
        result = combined_pixels(info,plan["matrix"])
        path = OUT/"train/images"/info["augmented_filename"]
        Image.fromarray(result).save(path,format="PNG")
        text = "".join(str(r[0])+" "+" ".join(format(v,".17g") for v in r[1:])+"\n" for r in plan["rows"])
        (OUT/"train/labels"/(path.stem+".txt")).write_text(text,encoding="ascii")


def verify_output(originals, plans):
    source_by_name = {r["filename"]:r for r in originals}
    plan_by_name = {p["info"]["augmented_filename"]:p for p in plans}
    image_rows = []
    file_hashes,pixel_hashes,groups = {},{},defaultdict(set)
    counts = {s:Counter() for s in COUNTS}
    zeros = Counter()
    max_label_error = 0.0
    for split in COUNTS:
        expected = {r["filename"] for r in originals if r["split"] == split}
        if split == "train":
            expected |= set(plan_by_name)
        files = {p.name:p for p in (OUT/split/"images").iterdir()}
        labs = {p.name:p for p in (OUT/split/"labels").iterdir()}
        require(set(files) == expected and len(files) == FINAL_COUNTS[split] and
                set(labs) == {Path(n).stem+".txt" for n in expected},"Missing/extra image-label pair")
        for name,path in sorted(files.items()):
            label = labs[path.stem+".txt"]
            parsed = labels(label)
            per_image = Counter(NAMES[r[1]] for r in parsed)
            counts[split].update(per_image)
            zeros[split] += not parsed
            sha,labsha = digest(path),digest(label)
            require(sha not in file_hashes,"Exact duplicate image anomaly")
            file_hashes[sha] = path
            with Image.open(path) as im:
                im.load()
                width,height = im.size
                rgb = np.asarray(im.convert("RGB"))
            pixel_sha = hashlib.sha256(str((width,height)).encode()+rgb.tobytes()).hexdigest()
            require(pixel_sha not in pixel_hashes,"Duplicate decoded image anomaly")
            pixel_hashes[pixel_sha] = path
            if name in source_by_name:
                parent = source_by_name[name]
                require(parent["split"] == split and sha == parent["image_sha256"] and
                        labsha == parent["label_sha256"] and
                        len(parsed) == int(parent["annotation_count"]),"Original/Val/Test changed")
                iid,source_split,parent_id,is_aug = parent["image_id"],parent["source_split"],"",False
            else:
                require(split == "train","Augmentation outside Train")
                plan = plan_by_name[name]
                info = plan["info"]
                parent = source_by_name[info["parent_filename"]]
                require(parent["split"] == "train" and len(parsed) == int(parent["annotation_count"]) ==
                        len(plan["rows"]),"Annotation removed or parent split changed")
                for actual,target in zip(parsed,plan["rows"]):
                    require(actual[1] == target[0],"Class ID mismatch")
                    max_label_error = max(max_label_error,*(abs(a-b) for a,b in zip(actual[2:],target[1:])))
                expected_pixels = combined_pixels(info,plan["matrix"])
                require(np.array_equal(rgb,expected_pixels),"Saved image differs from the recorded affine transform")
                iid,source_split,parent_id,is_aug = path.stem,"train",int(parent["image_id"]),True
            require((width,height) == (int(parent["width"]),int(parent["height"])),"Native canvas changed")
            groups[parent["group_key"]].add(split)
            image_rows.append({
                "image_id":iid,"filename":name,"source_split":source_split,"split":split,
                "group_key":parent["group_key"],"parent_image_id":parent_id,"is_augmentation":is_aug,
                "width":width,"height":height,"image_sha256":sha,"label_sha256":labsha,
                "o_count":per_image["o"],"p_count":per_image["p"],"s_count":per_image["s"],"sc_count":per_image["sc"],
                "annotation_count":len(parsed),"zero_annotation":not parsed,
            })
    require(max_label_error < 1e-12,"Saved bbox differs from four-corner affine AABB")
    require(len(image_rows) == len(file_hashes) == len(pixel_hashes) == 737 and
            all(len(v) == 1 for v in groups.values()),"Image/hash/group split leakage")
    actual_counts = {s:{n:counts[s][n] for n in NAMES.values()} for s in COUNTS}
    e1_summary = json.loads((E1/"conversion_summary.json").read_text(encoding="utf-8"))
    require(actual_counts == e1_summary["split_class_counts"],"E1/E2 annotation budget or classes differ")
    require(dict(zeros) == {"train":5,"val":0,"test":1},"Zero-label membership changed")
    return image_rows,{
        "split_image_counts":FINAL_COUNTS,"split_class_counts":actual_counts,
        "total_images":737,"total_annotations":sum(sum(c.values()) for c in actual_counts.values()),
        "zero_label_counts":dict(zeros),"invalid_bbox":0,"invalid_yolo_rows":0,"polygon_rows":0,
        "missing_images":0,"missing_labels":0,"split_mismatch":0,
        "exact_image_duplicates":0,"decoded_pixel_duplicates":0,"declared_group_leakage":0,
        "max_saved_label_error":max_label_error,"all_originals_preserved":True,
        "val_identical":True,"test_identical":True,"annotation_preservation":True,"class_mismatch":0,
    }


def make_reports(manifest, rejected, failures, clipped, source_counts, aug_classes, image_rows, qa):
    import importlib.metadata
    import platform
    passed = qa is not None and not failures
    manifest_bytes = csv_bytes(manifest,list(manifest[0]) if manifest else [
        "parent_filename","parent_image_id","augmented_filename","variant","flip_type","brightness_factor",
        "contrast_factor","angle_deg","translate_x_px","translate_y_px","retry_count","rejected_transform_count",
        "seed","o_present","s_present","annotation_count","transform_order"])
    reject_fields = ["parent_filename","parent_image_id","variant","attempt","flip_type","angle_deg",
                     "translate_x_px","translate_y_px","brightness_factor","contrast_factor","seed","failure_reason"]
    rejected_bytes = csv_bytes(rejected,reject_fields)
    summary = {
        "experiment":"E3","dataset_qa":"PASS" if passed else "FAIL","conversion_success":passed,
        "training_executed":False,"source_dataset":SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256":SOURCE_MANIFEST_SHA256,"source_augmented_images_used":False,
        "original_train_count":534,"augmentation_count":70,"final_train_count":604,
        "val_count":67,"test_count":66,"source_parent_count":35,
        "parent_ids":sorted({r["parent_image_id"] for r in manifest}),
        "parent_match_E1":True,"parent_match_E2":True,
        "variant_counts":dict(Counter(r["variant"] for r in manifest)),
        "rejected_transform_attempts":len(rejected),
        "max_retry_count_used":max((r["retry_count"] for r in manifest),default=0),
        "max_transform_attempts":MAX_ATTEMPTS,"seed":SEED,
        "transform_order":"flip -> affine -> brightness -> contrast",
        "augmentation_parameters":{
            "A":{"flip":"horizontal","angle_deg":[-5,5],"translation":False},
            "B":{"flip":"vertical","angle_deg":[-5,5],"translation_fraction":[-.03,.03]},
            "brightness_range":[.93,1.07],"contrast_range":[.95,1.05],
            "sampling":"random.Random(42), parent_id ascending, A then B; per attempt angle, B tx/ty, brightness, contrast",
            "rotation_center":"(width/2,height/2)","canvas":"native dimensions",
            "interpolation":"OpenCV INTER_LINEAR","border":"constant RGB(0,0,0)",
            "bbox":"flip four corners -> affine -> enclosing AABB -> boundary clipping",
            "scale":1.0,"shear":False,"perspective":False,"blur":False,"noise":False,"gamma":False,
            "copy_paste":False,"mosaic":False,"mixup":False,"online_augmentation":False,
            "rejection_policy":"Reject whole transform if any bbox invalid; no annotation deletion or parent replacement",
        },
        "evaluation_policy":{"selection_split":"val","validation_images":67,"test_images_integrity_only":66,
                             "test_evaluation_allowed":False,"test_used_for_selection":False,
                             "reason":"Test66 was exposed by E1; official experiment comparison is validation-only"},
        "class_mapping":NAMES,"original_class_counts":source_counts,"augmented_class_counts":aug_classes,
        "unresolved_transform_failures":failures,"clipped_bbox_count":clipped,
        "materialized_augmented_count":70 if passed else 0,
        "E1_parent_manifest_sha256":E1_AUG_MANIFEST_SHA256,"E2_parent_manifest_sha256":E2_AUG_MANIFEST_SHA256,
        "augmentation_csv_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
        "rejected_transforms_sha256":hashlib.sha256(rejected_bytes).hexdigest(),
        "preparation_script_sha256":digest(Path(__file__)),
        "generation_environment":{"python":platform.python_version(),"opencv":cv2.__version__,
                                  "numpy":np.__version__,"pillow":importlib.metadata.version("pillow")},
    }
    reports = {
        "augmentation_manifest.csv":manifest_bytes,"rejected_transforms.csv":rejected_bytes,
        "qa_failures.csv":csv_bytes(failures,["parent_image_id","parent_filename","variant","attempts","reason"]),
    }
    if passed:
        summary.update(qa)
        image_bytes = csv_bytes(image_rows,list(image_rows[0]))
        original_bytes = (SOURCE/"image_manifest.csv").read_bytes()
        e1_bytes = (E1/"augmentation_manifest.json").read_bytes()
        e2_bytes = (E2/"augmentation_manifest.json").read_bytes()
        reference = json.loads((E2/"baseline_reference.json").read_text(encoding="utf-8"))
        reference.update(
            geometric_run="runs/v2_geometric/yolov8n_geometric_v2",
            geometric_script_sha256=digest(ROOT/"scripts/train_yolov8_geometric_v2_colab.py"),
            geometric_manifest_sha256=json.loads((E2/"conversion_summary.json").read_text(encoding="utf-8"))["image_manifest_sha256"],
            geometric_image_counts=FINAL_COUNTS,
            evaluation_policy="VAL67 ONLY; Test66 integrity checks permitted, performance evaluation prohibited",
        )
        reference_bytes = json_bytes(reference)
        param_fields = ["variant","flip_type","brightness_factor","contrast_factor","angle_deg",
                        "translate_x_px","translate_y_px","retry_count","rejected_transform_count","seed","transform_order"]
        aug = {"seed":42,"max_attempts":100,"transform_order":summary["transform_order"],
               "source_E1_parent_manifest_sha256":E1_AUG_MANIFEST_SHA256,
               "source_E2_parent_manifest_sha256":E2_AUG_MANIFEST_SHA256,
               "records":[{"image_id":Path(r["augmented_filename"]).stem,"parent_image_id":r["parent_image_id"],
                           "file_name":r["augmented_filename"],"parameters":{k:r[k] for k in param_fields}}
                          for r in manifest]}
        aug_bytes = json_bytes(aug)
        summary.update(
            image_manifest_sha256=hashlib.sha256(image_bytes).hexdigest(),
            baseline_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            corrected_manifest_sha256=hashlib.sha256(original_bytes).hexdigest(),
            augmentation_manifest_sha256=hashlib.sha256(aug_bytes).hexdigest(),
            class_counts=qa["split_class_counts"],
        )
        body = {"train":"train/images","val":"val/images","test":"test/images","nc":4,"names":NAMES}
        reports.update({
            "image_manifest.csv":image_bytes,"corrected_image_manifest.csv":original_bytes,
            "E1_parent_manifest.json":e1_bytes,"E2_parent_manifest.json":e2_bytes,
            "baseline_reference.json":reference_bytes,"augmentation_manifest.json":aug_bytes,
            "data.yaml":yaml.safe_dump({"path":OUT.as_posix(),**body},sort_keys=False).encode(),
            "data_colab.yaml":yaml.safe_dump({
                "path":"/content/drive/MyDrive/ai-cd-metrology/datasets/yolov8_detection_combined_v2",**body},
                sort_keys=False).encode(),
            "requirements_colab.txt":(E2/"requirements_colab.txt").read_bytes(),
        })
    reports["conversion_summary.json"] = json_bytes(summary)
    reports["qa_report.json"] = json_bytes({
        k:summary.get(k) for k in ["dataset_qa","parent_match_E1","parent_match_E2","split_image_counts",
        "split_class_counts","annotation_preservation","class_mismatch","invalid_bbox","missing_images","missing_labels",
        "exact_image_duplicates","decoded_pixel_duplicates","declared_group_leakage","val_identical","test_identical",
        "rejected_transform_attempts","max_retry_count_used","evaluation_policy"]})
    readme = [
        "# E3 combined augmentation / validation-only comparison","",
        "QA: "+summary["dataset_qa"]+". Train604 / Val67 / Test66; originals534 + augmentation70.",
        "Exactly the same35 o/s-positive parents as E1/E2, two variants each.",
        "All new pixels start from E0 corrected original Train; E1/E2 augmented images are never re-augmented.",
        "A: horizontal flip + rotation[-5,+5]deg, translation OFF.",
        "B: vertical flip + rotation[-5,+5]deg + translation x/y[-3,+3]percent.",
        "Both: brightness[0.93,1.07], contrast[0.95,1.05].",
        "Order: flip -> affine -> PIL brightness -> PIL contrast.",
        "Four-corner bbox geometry uses the same flip and affine; enclosing AABB is clipped to image boundaries.",
        "Seed42; parent_id ascending, A then B; draw angle, optional tx/ty, brightness, contrast each proposal.",
        "Reject entire invalid transform; at most100 proposals including initial. No bbox deletion, parent change or range shrink.",
        "Native canvas, linear interpolation, constant black border, RGB, lossless augmented PNG.",
        "No scale/shear/perspective/blur/noise/gamma/copy-paste/mosaic/mixup/online augmentation.",
        "Rejected transform attempts: "+str(len(rejected))+"; maximum accepted retries: "+str(summary["max_retry_count_used"])+".",
        "Every original image/label is retained byte-for-byte; Val/Test also match E1/E2.","",
        "## Test policy","",
        "VAL67 ONLY for model/experiment selection and official metric comparison.",
        "Test66 was exposed by E1. It is copied and checked for integrity only.",
        "The E3 training script MUST NOT run Test inference/evaluation or read historical Test metrics.",
        "After fitting, only best.pt validation results are saved as comparison metrics.",
        "No training, validation inference or Test evaluation was executed during preparation.","",
        "## Colab","",
        "Copy this dataset folder to /content/drive/MyDrive/ai-cd-metrology/datasets/yolov8_detection_combined_v2/.",
        "Copy scripts/train_yolov8_combined_v2_colab.py to the existing Drive scripts folder.",
        "Reuse the original models/yolov8n.pt and completed E0/E1/E2 run folders.",
        "Preflight requires matching saved args and Python/Torch/CUDA/Ultralytics/GPU environment.",
        "Ultralytics8.4.174, YOLOv8n, epochs100, imgsz640, batch8, seed42, patience20, workers0, deterministicTrue.",
        "Optimizer=auto (actual AdamW/lr0.00125), AMPFalse, every YOLO online augmentation OFF.",
        "Run: /content/drive/MyDrive/ai-cd-metrology/runs/v2_combined/yolov8n_combined_v2/.",
        "Use --check-only before the saved script's training invocation; no CPU fallback.",
        "Local reproduction check without writes: python scripts/prepare_yolov8_combined_v2.py --verify-only.","",
    ]
    reports["README.md"] = "\n".join(readme).encode("utf-8")
    return reports,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    require(args.verify_only or not OUT.exists(),"Output exists; never overwrite E3 or prior experiments")
    originals,by_id,parents,counts = source_data()
    protected_paths = [
        SOURCE/"image_manifest.csv",E1/"augmentation_manifest.json",E2/"augmentation_manifest.json",
        E1/"baseline_reference.json",E2/"baseline_reference.json",
        ROOT/"scripts/train_yolov8_corrected_v2_colab.py",
        ROOT/"scripts/train_yolov8_general_v2_colab.py",ROOT/"scripts/train_yolov8_geometric_v2_colab.py",
        ROOT/"scripts/prepare_yolov8_geometric_v2.py",
    ]
    protected = {p:digest(p) for p in protected_paths}
    manifest,plans,rejected,failures,clipped,aug_classes = build_plan(by_id,parents)
    if not args.verify_only:
        OUT.mkdir(parents=True,exist_ok=False)
        if not failures:
            materialize(originals,plans)
    image_rows,qa = (None,None) if failures else verify_output(originals,plans)
    reports,summary = make_reports(manifest,rejected,failures,clipped,counts,aug_classes,image_rows,qa)
    for name,content in reports.items():
        path = OUT/name
        if args.verify_only:
            require(path.read_bytes() == content,"Generated metadata changed: "+name)
        else:
            with path.open("xb") as stream:
                stream.write(content)
    require(all(digest(p) == value for p,value in protected.items()),"E0/E1/E2 protected artifact changed")
    source_data()
    print(json.dumps(summary,indent=2))
    return 0 if summary["dataset_qa"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

