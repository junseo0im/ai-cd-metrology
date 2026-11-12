"""E6: read-only E1 base + four diverse variants from the same 35 original parents.\nNo training, model inference, Test inspection or external-image access.\n"""
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
from PIL import Image, ImageEnhance
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/derived/yolov8_detection_general_v2"
OUT = ROOT/"data/derived/yolov8_detection_rare_diverse_v2"
REPORTS = ROOT/"data/derived/yolov8_rare_diverse_v2"
NAMES = {0:"o",1:"p",2:"s",3:"sc"}
COUNTS = {"train":604,"val":67,"test":66}
FINAL_COUNTS = {"train":744,"val":67,"test":66}
SOURCE_MANIFEST_SHA256 = "b0ce5cd816ef2dd3a0a8d8652785844d009a66dcf06505cb03485e56ad9f1b11"
E1_AUG_MANIFEST_SHA256 = "2f537e1db78385cb0671a3965f97a36e27982a524adad0e520dd34f035b9e4c7"
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


def transform_bbox(row, width, height, matrix):
    _,cid,x,y,w,h = row
    x0,x1,y0,y1 = (x-w/2)*width,(x+w/2)*width,(y-h/2)*height,(y+h/2)*height
    corners = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]],dtype=np.float64)
    points = corners @ matrix.T
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

CLASS_COUNTS = {
    "train":{"o":105,"p":5054,"s":120,"sc":511},
    "val":{"o":4,"p":593,"s":4,"sc":54},
    "test":{"o":6,"p":580,"s":5,"sc":53},
}
PARAMETERS = {
    "A":{"family":"brightness_contrast","brightness":[.90,1.10],"contrast":[.92,1.08]},
    "B":{"family":"rotation","angle_deg":[-5.,5.]},
    "C":{"family":"translation","translate_x_fraction":[-.03,.03],"translate_y_fraction":[-.03,.03]},
    "D":{"family":"mild_blur_photometric","kernel_size":3,"sigma_px":[.3,.7],
         "brightness":[.95,1.05],"contrast":[.97,1.03]},
}


def source_snapshot():
    return {p.relative_to(SOURCE).as_posix():digest(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()}


def source_data():
    require(digest(SOURCE/"image_manifest.csv") == SOURCE_MANIFEST_SHA256,"Frozen E1 manifest changed")
    require(digest(SOURCE/"augmentation_manifest.json") == E1_AUG_MANIFEST_SHA256,"Frozen E1 parent set changed")
    cfg = yaml.safe_load((SOURCE/"data_colab.yaml").read_text(encoding="utf-8"))
    require(cfg["nc"] == 4 and cfg["names"] == NAMES,"Class mapping changed")
    with (SOURCE/"image_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        originals = list(csv.DictReader(f))
    require(len(originals) == len({r["image_id"] for r in originals}) == len({r["filename"] for r in originals}) == 737
            and dict(Counter(r["split"] for r in originals)) == COUNTS,"E1 identity/split mismatch")
    by_id = {int(r["image_id"]):r for r in originals}
    aug = json.loads((SOURCE/"augmentation_manifest.json").read_text(encoding="utf-8-sig"))
    parents = sorted({int(r["parent_image_id"]) for r in aug["records"]})
    require(len(aug["records"]) == 70 and len(parents) == 35 and
            all(sum(int(a["parent_image_id"]) == i for a in aug["records"]) == 2 for i in parents),
            "E1 frozen augmentation parent records mismatch")
    expected_parents = sorted(int(r["image_id"]) for r in originals if r["split"] == "train" and
                              r["existing_augmentation"] == "False" and int(r["o_count"])+int(r["s_count"]) > 0)
    require(parents == expected_parents,"Do not select new parents or augment an existing augmented image")
    groups,hashes = defaultdict(set),set()
    totals = {s:Counter() for s in COUNTS}
    for split in COUNTS:
        selected = [r for r in originals if r["split"] == split]
        require({p.name for p in (SOURCE/split/"images").iterdir()} == {r["filename"] for r in selected}
                and {p.name for p in (SOURCE/split/"labels").iterdir()} ==
                {Path(r["filename"]).stem+".txt" for r in selected},"E1 missing/extra file")
        for row in selected:
            image = SOURCE/split/"images"/row["filename"]
            label = SOURCE/split/"labels"/(image.stem+".txt")
            require(digest(image) == row["image_sha256"] and digest(label) == row["label_sha256"],
                    "E1 original image/label hash changed")
            require(row["image_sha256"] not in hashes,"Exact source duplicate")
            hashes.add(row["image_sha256"])
            groups[row["group_key"]].add(split)
            # Test labels are opaque; metadata counts are already bound by frozen hashes.
            actual = Counter({n:int(row[n+"_count"]) for n in NAMES.values()}) if split == "test" else \
                     Counter(NAMES[r[1]] for r in labels(label))
            require(all(actual[n] == int(row[n+"_count"]) for n in NAMES.values()),"E1 annotation mismatch")
            totals[split].update(actual)
    require(all(len(s) == 1 for s in groups.values()),"E1 group leakage")
    counts = {s:{n:totals[s][n] for n in NAMES.values()} for s in COUNTS}
    require(counts == CLASS_COUNTS,"E1 class counts changed")
    for iid in parents:
        parent = by_id[iid]
        require(parent["split"] == "train" and parent["existing_augmentation"] == "False" and
                int(parent["o_count"])+int(parent["s_count"]) > 0,"Invalid E1 original parent")
    return originals,by_id,parents


def sample_parameters(rng,variant,width,height):
    require(variant in PARAMETERS,"Unknown family")
    values = dict(variant=variant,family=PARAMETERS[variant]["family"],angle_deg=0.,
                  translate_x_px=0.,translate_y_px=0.,brightness_factor=1.,contrast_factor=1.,
                  blur_kernel_size=0,blur_sigma_px=0.,flip=False)
    if variant == "A":
        values.update(brightness_factor=rng.uniform(.90,1.10),contrast_factor=rng.uniform(.92,1.08))
    elif variant == "B":
        angle = rng.uniform(-5.,5.)
        while angle == 0.:
            angle = rng.uniform(-5.,5.)
        values["angle_deg"] = angle
    elif variant == "C":
        tx,ty = rng.uniform(-.03,.03)*width,rng.uniform(-.03,.03)*height
        while tx == 0. and ty == 0.:
            tx,ty = rng.uniform(-.03,.03)*width,rng.uniform(-.03,.03)*height
        values.update(translate_x_px=tx,translate_y_px=ty)
    else:
        values.update(blur_kernel_size=3,blur_sigma_px=rng.uniform(.3,.7),
                      brightness_factor=rng.uniform(.95,1.05),contrast_factor=rng.uniform(.97,1.03))
    return values


def build_plan(by_id,parents,max_attempts=MAX_ATTEMPTS):
    rng = random.Random(SEED)
    manifest,plans,rejected = [],[],[]
    clipped_total = 0
    for iid in sorted(parents):
        parent = by_id[iid]
        width,height = int(parent["width"]),int(parent["height"])
        rows = labels(SOURCE/"train/labels"/(Path(parent["filename"]).stem+".txt"))
        require(any(r[1] in {0,2} for r in rows),"Parent must be o/s-positive")
        for variant in PARAMETERS:
            for attempt in range(1,max_attempts+1):
                params = sample_parameters(rng,variant,width,height)
                matrix = matrix_for(width,height,params["angle_deg"],params["translate_x_px"],params["translate_y_px"])
                transformed,errors,clipped_count = [],[],0
                for row in rows:
                    if variant in {"A","D"}:
                        encoded,was_clipped = (row[1],*row[2:]),False
                    else:
                        encoded,raw,clipped,was_clipped = transform_bbox(row,width,height,matrix)
                    if encoded is None or encoded[0] != row[1]:
                        errors.append({"label_row":row[0],"class_id":row[1],"reason":"nonpositive bbox after clipping"})
                    else:
                        transformed.append(encoded)
                        clipped_count += was_clipped
                if errors:
                    rejected.append(dict(parent_image_id=iid,parent_filename=parent["filename"],
                                         attempt=attempt,seed=SEED,**params,failure_reason=json.dumps(errors)))
                    continue
                info = dict(parent_image_id=iid,parent_filename=parent["filename"],
                            augmented_filename=f"rare_diverse_{iid}_{variant}.png",
                            width=width,height=height,parent_group_key=parent["group_key"],
                            o_present=int(parent["o_count"]) > 0,s_present=int(parent["s_count"]) > 0,
                            annotation_count=len(rows),seed=SEED,retry_count=attempt-1,
                            rejected_transform_count=attempt-1,clipped_bbox_count=clipped_count,**params)
                manifest.append(info)
                plans.append({"info":info,"matrix":matrix,"rows":transformed})
                clipped_total += clipped_count
                break
            else:
                raise ValueError(f"QA FAIL: parent {iid} variant {variant}, {max_attempts} invalid transforms; no bbox removed")
    require(len(plans) == len(manifest) == 140,"Expected four variants per 35 parents")
    return manifest,plans,rejected,clipped_total


def augmented_pixels(info,matrix):
    # Only the named E1 original TRAIN parent can be opened here; never external data.
    with Image.open(SOURCE/"train/images"/info["parent_filename"]) as image:
        require(image.size == (info["width"],info["height"]),"Parent dimensions changed")
        image = image.convert("RGB")
        rgb = np.asarray(image)
    if info["variant"] in {"B","C"}:
        rgb = cv2.warpAffine(rgb,matrix,(info["width"],info["height"]),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    if info["variant"] == "D":
        rgb = cv2.GaussianBlur(rgb,(3,3),sigmaX=info["blur_sigma_px"],sigmaY=info["blur_sigma_px"],
                               borderType=cv2.BORDER_REFLECT_101)
    if info["variant"] in {"A","D"}:
        image = ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(info["brightness_factor"])
        rgb = np.asarray(ImageEnhance.Contrast(image).enhance(info["contrast_factor"]))
    return rgb


def encoded_label(plan):
    if plan["info"]["variant"] in {"A","D"}:
        return (SOURCE/"train/labels"/(Path(plan["info"]["parent_filename"]).stem+".txt")).read_bytes()
    return "".join(str(r[0])+" "+" ".join(format(v,".17g") for v in r[1:])+"\n" for r in plan["rows"]).encode("ascii")


def materialize(originals,plans):
    for split in COUNTS:
        for kind in ("images","labels"):
            (OUT/split/kind).mkdir(parents=True)
    for row in originals:
        for kind,name in [("images",row["filename"]),("labels",Path(row["filename"]).stem+".txt")]:
            shutil.copy2(SOURCE/row["split"]/kind/name,OUT/row["split"]/kind/name)
    for plan in plans:
        name = plan["info"]["augmented_filename"]
        Image.fromarray(augmented_pixels(plan["info"],plan["matrix"])).save(OUT/"train/images"/name,format="PNG")
        (OUT/"train/labels"/(Path(name).stem+".txt")).write_bytes(encoded_label(plan))


def verify_output(originals,plans):
    source_by_name = {r["filename"]:r for r in originals}
    plan_by_name = {p["info"]["augmented_filename"]:p for p in plans}
    require(not set(source_by_name)&set(plan_by_name),"Output filename collision")
    image_rows,files_seen,pixels_seen,groups = [],{}, {},defaultdict(set)
    counts = {s:Counter() for s in COUNTS}
    zeros = Counter()
    max_label_error = 0.
    for split in COUNTS:
        expected = {r["filename"] for r in originals if r["split"] == split}
        if split == "train":
            expected |= set(plan_by_name)
        images = {p.name:p for p in (OUT/split/"images").iterdir()}
        labs = {p.name:p for p in (OUT/split/"labels").iterdir()}
        require(set(images) == expected and len(images) == FINAL_COUNTS[split] and
                set(labs) == {Path(n).stem+".txt" for n in expected},"Missing/extra image-label pair")
        for name,path in sorted(images.items()):
            lab = labs[path.stem+".txt"]
            sha,labsha = digest(path),digest(lab)
            require(sha not in files_seen,"Exact file duplicate anomaly")
            files_seen[sha] = path
            # Pixel decoding is used solely for identity hashing, never qualitative Test inspection.
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
                width,height = im.size
            pixel_sha = hashlib.sha256(str((width,height)).encode()+rgb.tobytes()).hexdigest()
            require(pixel_sha not in pixels_seen,"Exact decoded-pixel duplicate anomaly")
            pixels_seen[pixel_sha] = path
            if name in source_by_name:
                parent = source_by_name[name]
                require(parent["split"] == split and sha == parent["image_sha256"] and
                        labsha == parent["label_sha256"],"Existing E1 image/label/split changed")
                row = dict(parent,e6_augmentation=False,augmentation_family="")
                if split == "test":
                    parsed = None  # Test annotation content remains opaque; frozen identity only.
                    per_class = Counter({n:int(parent[n+"_count"]) for n in NAMES.values()})
                    zero = parent["zero_annotation"] == "True"
                else:
                    parsed = labels(lab)
                    per_class = Counter(NAMES[r[1]] for r in parsed)
                    zero = not parsed
                require(all(per_class[n] == int(parent[n+"_count"]) for n in NAMES.values()),"Base class mismatch")
            else:
                require(split == "train","Augmentation leaked outside Train")
                plan = plan_by_name[name]
                info = plan["info"]
                parent = source_by_name[info["parent_filename"]]
                require(parent["split"] == "train" and parent["existing_augmentation"] == "False",
                        "Not an original rare TRAIN parent")
                parsed = labels(lab)
                require(len(parsed) == len(plan["rows"]) == int(parent["annotation_count"]),"Annotation removed")
                require(lab.read_bytes() == encoded_label(plan),"Saved labels differ from deterministic transform")
                for actual,target in zip(parsed,plan["rows"]):
                    require(actual[1] == target[0],"Class mismatch")
                    max_label_error = max(max_label_error,*(abs(a-b) for a,b in zip(actual[2:],target[1:])))
                require(np.array_equal(rgb,augmented_pixels(info,plan["matrix"])),"Saved pixels differ from parameters")
                per_class = Counter(NAMES[r[1]] for r in parsed)
                zero = not parsed
                row = {k:"" for k in originals[0]}
                row.update(image_id=Path(name).stem,filename=name,source_split="train",split="train",
                           group_key=parent["group_key"],parent_image_id=parent["image_id"],
                           existing_augmentation=False,e6_augmentation=True,augmentation_family=info["family"],
                           width=width,height=height,image_sha256=sha,label_sha256=labsha,
                           annotation_count=len(parsed),zero_annotation=zero,
                           **{n+"_count":per_class[n] for n in NAMES.values()})
            require((width,height) == (int(parent["width"]),int(parent["height"])),"Native image dimensions changed")
            counts[split].update(per_class)
            zeros[split] += zero
            groups[parent["group_key"]].add(split)
            image_rows.append(row)
    expected_classes = {s:CLASS_COUNTS[s].copy() for s in COUNTS}
    for plan in plans:
        expected_classes["train"] = dict(Counter(expected_classes["train"])+Counter(NAMES[r[0]] for r in plan["rows"]))
    actual = {s:{n:counts[s][n] for n in NAMES.values()} for s in COUNTS}
    require(actual == expected_classes and dict(zeros) == {"train":5,"val":0,"test":1},
            "Annotation/zero-label count mismatch")
    require(len(image_rows) == len(files_seen) == len(pixels_seen) == 877 and
            all(len(s) == 1 for s in groups.values()),"Duplicate/group leakage")
    require(max_label_error < 1e-12,"Serialized bbox differs from affine AABB")
    return image_rows,{
        "status":"PASS","split_image_counts":FINAL_COUNTS,"split_class_counts":actual,
        "total_images":877,"total_annotations":sum(sum(c.values()) for c in actual.values()),
        "zero_label_counts":dict(zeros),"invalid_bbox":0,"invalid_yolo_rows":0,"polygon_rows":0,
        "missing_image":0,"missing_label":0,"split_mismatch":0,"class_mismatch":0,
        "removed_annotations":0,"exact_image_duplicates":0,"decoded_pixel_duplicates":0,
        "declared_group_leakage":0,"max_saved_label_error":max_label_error,
        "all_E1_images_labels_preserved":True,"val_identical":True,"test_identical":True,
        "test_evaluated":False,"test_evaluation_calls":0,"external_20_used":False,
        "physical_group_identity_verified":False,
    }


def report_bytes(originals,by_id,parents,manifest,rejected,clipped,image_rows,qa):
    import importlib.metadata
    import platform
    parent_rows = [{k:by_id[iid][k] for k in [
        "image_id","filename","split","group_key","image_sha256","label_sha256","o_count","s_count","annotation_count"]}
                   for iid in parents]
    aug = json_bytes({"seed":SEED,"parameters":PARAMETERS,"max_attempts":MAX_ATTEMPTS,
                      "source":"E1 original train parents only","records":manifest})
    rejected_data = json_bytes(rejected)
    image_data = csv_bytes(image_rows,list(image_rows[0]))
    parent_data = csv_bytes(parent_rows,list(parent_rows[0]))
    recipe = {
        "order":"A brightness->contrast; B affine(rotation); C affine(translation); "
                "D GaussianBlur->brightness->contrast",
        "geometric":"native canvas, four corners -> affine -> enclosing AABB -> clip",
        "affine_interpolation":"OpenCV INTER_LINEAR, scale=1, constant RGB black border",
        "blur":"3x3 Gaussian, sigmaX=sigmaY, BORDER_REFLECT_101",
        "invalid_policy":"reject entire transform and resample, max100 proposals; no annotation removed",
        "photometric_labels":"A/D labels are byte copies; B/C normalized float64 with 17 significant digits",
        "sampling_order":"random.Random(42); numeric parent ID ascending; A,B,C,D; accepted/rejected draws recorded",
        "image_output":"lossless RGB PNG, native size; E1 base images/labels copied byte-for-byte",
    }
    summary = {
        "experiment":"E6","dataset_qa":"PASS","conversion_success":True,
        "original_train_count":604,"augmentation_count":140,"final_train_count":744,
        "val_count":67,"test_count":66,"source_parent_count":35,
        "family_counts":dict(Counter(r["variant"] for r in manifest)),
        "source_path":SOURCE.as_posix(),"output_path":OUT.as_posix(),
        "source_manifest_sha256":SOURCE_MANIFEST_SHA256,"E1_parent_manifest_sha256":E1_AUG_MANIFEST_SHA256,
        "seed":SEED,"augmentation_parameters":PARAMETERS,"recipe":recipe,
        "rejected_transform_attempts":len(rejected),"max_retry_count_used":max(r["retry_count"] for r in manifest),
        "clipped_bbox_count":clipped,
        "augmentation_manifest_sha256":hashlib.sha256(aug).hexdigest(),
        "augmentation_csv_sha256":hashlib.sha256(csv_bytes(manifest,list(manifest[0]))).hexdigest(),
        "parent_manifest_sha256":hashlib.sha256(parent_data).hexdigest(),
        "rejected_transforms_sha256":hashlib.sha256(rejected_data).hexdigest(),
        "image_manifest_sha256":hashlib.sha256(image_data).hexdigest(),
        "baseline_reference_sha256":digest(SOURCE/"baseline_reference.json"),
        "builder_script_sha256":digest(Path(__file__)),
        "python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__,
        "Pillow":importlib.metadata.version("Pillow"),
        "evaluation_policy":{"selection_split":"val","test_evaluation_allowed":False,
                             "test_used_for_selection":False,"external_20_used":False},
        "training_executed":False,"inference_executed":False,**qa,
    }
    return summary,{
        "augmentation_manifest.json":aug,
        "augmentation_manifest.csv":csv_bytes(manifest,list(manifest[0])),
        "parent_manifest.csv":parent_data,"rejected_transforms.json":rejected_data,
        "image_manifest.csv":image_data,"preparation_qa.json":json_bytes(summary),
        "conversion_summary.json":json_bytes(summary),
    }


def write_reports(originals,by_id,parents,manifest,rejected,clipped,image_rows,qa):
    summary,artifacts = report_bytes(originals,by_id,parents,manifest,rejected,clipped,image_rows,qa)
    for name,value in artifacts.items():
        (OUT/name).write_bytes(value)
    for source,target in [("image_manifest.csv","E1_image_manifest.csv"),
                          ("augmentation_manifest.json","E1_augmentation_manifest.json"),
                          ("baseline_reference.json","baseline_reference.json")]:
        shutil.copy2(SOURCE/source,OUT/target)
    shutil.copy2(SOURCE/"requirements_colab.txt",OUT/"requirements_colab.txt")
    drive = "/content/drive/MyDrive/ai-cd-metrology/datasets/yolov8_detection_rare_diverse_v2"
    for name,root in [("data.yaml",OUT.as_posix()),("data_colab.yaml",drive)]:
        (OUT/name).write_text(yaml.safe_dump(
            {"path":root,"train":"train/images","val":"val/images","test":"test/images","nc":4,"names":NAMES},
            sort_keys=False),encoding="utf-8")
    (OUT/"README.md").write_text(
        "# E6 rare-class diverse augmentation\n\n"
        "Read-only E1 Train604 (including its existing 70 augmentations) + 140 new images.\n"
        "Exactly the original E1 35 rare parents, four different families each; seed42.\n"
        "Train744 / Val67 / Test66. Val/Test images and annotations retain exact bytes.\n"
        "A brightness0.90-1.10/contrast0.92-1.08; B rotation[-5,+5]deg; C tx/ty±3%;\n"
        "D Gaussian3x3 sigma0.3-0.7px then brightness0.95-1.05/contrast0.97-1.03. No flips.\n"
        "Native canvas; deterministic rejection of any invalid transform; no labels removed.\n"
        "A/D labels are byte copies; B/C four-corner affine AABB with boundary clipping.\n"
        "Copy method: shutil.copy2; originals are never modified. No oversampling or class weighting.\n"
        "Model/loss and online-augmentation-OFF policy remain E1.\n"
        "Test integrity only, no inference/qualitative inspection. External20 never accessed.\n"
        "No training/inference was run in preparation. GPU and actual E1 run preflight deferred to Colab.\n"
        "QA: preparation_qa.json. All image/hash/class/annotation checks PASS; leakage0.\n",
        encoding="utf-8")
    REPORTS.mkdir(parents=True,exist_ok=False)
    for name in ["preparation_qa.json","augmentation_manifest.csv","augmentation_manifest.json",
                 "parent_manifest.csv","rejected_transforms.json"]:
        shutil.copy2(OUT/name,REPORTS/name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true",help="Read-only deterministic replay, no writes")
    args = parser.parse_args()
    before = source_snapshot()
    originals,by_id,parents = source_data()
    manifest,plans,rejected,clipped = build_plan(by_id,parents)
    if args.verify_only:
        require(OUT.is_dir() and REPORTS.is_dir(),"Completed output not found")
        image_rows,qa = verify_output(originals,plans)
        summary,artifacts = report_bytes(originals,by_id,parents,manifest,rejected,clipped,image_rows,qa)
        for name,value in artifacts.items():
            require((OUT/name).read_bytes() == value,"Reproducibility mismatch: "+name)
        require(before == source_snapshot(),"Source changed")
        print(json.dumps({"verify_only":"PASS",**qa}),flush=True)
        return
    require(not OUT.exists() and not REPORTS.exists(),"Never overwrite an existing E6 dataset/report")
    try:
        materialize(originals,plans)
        image_rows,qa = verify_output(originals,plans)
        require(before == source_snapshot(),"E1 source modified")
        summary = write_reports(originals,by_id,parents,manifest,rejected,clipped,image_rows,qa)
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        if not REPORTS.exists():
            REPORTS.mkdir(parents=True)
        (REPORTS/"generation_failure.json").write_bytes(json_bytes(
            {"status":"FAIL","error":str(error),"training_executed":False,"external_20_used":False}))
        raise


if __name__ == "__main__":
    main()
