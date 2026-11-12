"""Create lossless native GT bbox crops under the existing fixed split; never train."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import textwrap
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

import convert_coco_detection_fixed_v1 as detection

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/derived/mobilenet_bbox_crops_v1"
DATA=detection.OUT
FIXED=detection.FIXED
CLASSES=("p","o","s","sc")
SEED=42
ISSUE_FIELDS=("issue_type","split","class","source_image_id","annotation_id","filename","detail")
QA_KEYS=("nonpositive_crop_width","nonpositive_crop_height","invalid_bbox","bbox_outside_source",
         "unreadable_source","missing_source_image","source_size_mismatch","zero_byte_crop",
         "unreadable_crop","duplicate_output_filename","missing_class_folder","missing_crop",
         "crop_size_mismatch","crop_pixel_mismatch","split_leakage","annotation_count_mismatch")


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path,rows,fields=None):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def issue(issues,kind,row,detail):
    issues.append(dict(issue_type=kind,split=row.get("split",""),**{"class":row.get("class","")},
                       source_image_id=row.get("source_image_id",row.get("image_id","")),
                       annotation_id=row.get("annotation_id",""),
                       filename=row.get("crop_filename",row.get("filename","")),detail=detail))


def pixel_bounds(bbox):
    """Half-open bbox includes native pixel iff its center (index+0.5) lies inside."""
    x,y,w,h=map(float,bbox)
    return (math.ceil(x-0.5),math.ceil(y-0.5),
            math.ceil(x+w-0.5),math.ceil(y+h-0.5))


def pixel_sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def validate(rows,issues,assignments):
    """Independent saved-file readback against source pixels and frozen identities."""
    output_names=Counter(r["crop_filename"] for r in rows)
    for name,n in output_names.items():
        if n!=1:
            issue(issues,"duplicate_output_filename",{},"Duplicate filename: "+name)
    per_source=defaultdict(set)
    per_hash=defaultdict(set)
    per_group=defaultdict(set)
    by_image=defaultdict(list)
    for row in rows:
        sid=str(row["source_image_id"])
        split=row["split"]
        a=assignments[sid]
        per_source[sid].add(split)
        per_hash[a["sha256"]].add(split)
        if a["group_key"]:
            per_group[a["group_key"]].add(split)
        if split!=a["split"] or row["source_filename"]!=a["filename"]:
            issue(issues,"split_leakage",row,"Crop identity/split differs from fixed assignment")
        by_image[sid].append(row)
    for values in (per_source,per_hash,per_group):
        for key,splits in values.items():
            if len(splits)!=1:
                issue(issues,"split_leakage",{},"Multiple splits for source/hash/group: "+key)
    counts={s:Counter() for s in detection.EXPECTED_IMAGES}
    for i,(sid,rr) in enumerate(by_image.items()):
        a=assignments[sid]
        source=DATA/a["split"]/"images"/a["filename"]
        if not source.is_file():
            issue(issues,"missing_source_image",rr[0],str(source))
            continue
        with Image.open(source) as original:
            original.load()
            for row in rr:
                path=OUT/row["crop_relative_path"]
                counts[row["split"]][row["class"]]+=1
                if not path.is_file():
                    issue(issues,"missing_crop",row,str(path))
                    continue
                if path.stat().st_size==0:
                    issue(issues,"zero_byte_crop",row,"Empty file")
                    continue
                try:
                    with Image.open(path) as saved:
                        saved.load()
                        box=tuple(int(row[k]) for k in ("crop_x_start","crop_y_start","crop_x_end","crop_y_end"))
                        expected=original.crop(box)
                        if saved.size!=(int(row["crop_width"]),int(row["crop_height"])):
                            issue(issues,"crop_size_mismatch",row,str(saved.size))
                        if saved.size!=expected.size or saved.mode!=original.mode or saved.tobytes()!=expected.tobytes():
                            issue(issues,"crop_pixel_mismatch",row,"Saved native pixels differ from exact source window")
                        if pixel_sha(saved)!=row["crop_pixel_sha256"]:
                            issue(issues,"crop_pixel_mismatch",row,"Decoded pixel digest mismatch")
                    if detection.sha256(path)!=row["crop_file_sha256"]:
                        issue(issues,"crop_pixel_mismatch",row,"PNG file digest mismatch")
                except (OSError,ValueError) as exc:
                    issue(issues,"unreadable_crop",row,str(exc))
        if (i+1)%150==0:
            print(f"READBACK_SOURCES={i+1}/{len(by_image)}",flush=True)
    for split,expected in detection.EXPECTED_CLASSES.items():
        for name in CLASSES:
            folder=OUT/split/name
            if not folder.is_dir():
                issue(issues,"missing_class_folder",{"split":split,"class":name},str(folder))
            actual=len(list(folder.glob("*.png"))) if folder.is_dir() else 0
            if counts[split][name]!=expected[name] or actual!=expected[name]:
                issue(issues,"annotation_count_mismatch",{"split":split,"class":name},
                      f"Expected {expected[name]}, manifest {counts[split][name]}, files {actual}")
    return counts


def distribution(rows):
    result=[]
    for split in detection.EXPECTED_IMAGES:
        for name in CLASSES:
            selected=[r for r in rows if r["split"]==split and r["class"]==name]
            count=len(selected)
            stats={"split":split,"class":name,"crop_count":count,
                   "unique_source_images":len({r["source_image_id"] for r in selected}),
                   "percent_of_split":100*count/sum(r["split"]==split for r in rows)}
            for field,label in (("crop_width","width"),("crop_height","height")):
                values=[int(r[field]) for r in selected]
                stats.update({label+"_mean":float(np.mean(values)) if count else None,
                              label+"_median":float(np.median(values)) if count else None,
                              label+"_min":min(values) if count else None,
                              label+"_max":max(values) if count else None})
            for label,values in (
                ("aspect_ratio",[int(r["crop_width"])/int(r["crop_height"]) for r in selected]),
                ("bbox_area",[float(r["bbox_w"])*float(r["bbox_h"]) for r in selected])):
                stats[label+"_mean"]=float(np.mean(values)) if count else None
                stats[label+"_median"]=float(np.median(values)) if count else None
            result.append(stats)
    return result


def make_contact_sheets(rows):
    rng=random.Random(SEED)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",14)
    titlefont=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18)
    records=[]
    for name in CLASSES:
        pool=sorted((r for r in rows if r["class"]==name and r["split"]=="train"),
                    key=lambda r:(int(r["source_image_id"]),int(r["annotation_id"])))
        chosen=rng.sample(pool,min(25,len(pool))) if name in ("p","sc") else pool
        for page,start in enumerate(range(0,len(chosen),6),1):
            selected=chosen[start:start+6]
            imgs=[Image.open(OUT/r["crop_relative_path"]) for r in selected]
            cw=max(350,max(im.width+16 for im in imgs))
            heights=[max(170,max(im.height for im in imgs[i:i+2])+112)
                     for i in range(0,len(imgs),2)]
            canvas=Image.new("RGB",(cw*2,44+sum(heights)),"white")
            draw=ImageDraw.Draw(canvas)
            draw.text((10,10),f"TRAIN / {name} / page{page:02d} / NATIVE PIXELS; no resize",
                      font=titlefont,fill="black")
            y=44
            for row_index,hh in enumerate(heights):
                for col in range(2):
                    index=row_index*2+col
                    if index>=len(imgs):
                        continue
                    item,im=selected[index],imgs[index]
                    x=col*cw
                    caption=(f"image {item['source_image_id']} / ann {item['annotation_id']} / class {name}\n"
                             f"{item['crop_width']} x {item['crop_height']} px / {item['wafer']} {item['position']}\n"+
                             "\n".join(textwrap.wrap(item["source_filename"],width=max(42,(cw-16)//8))))
                    draw.multiline_text((x+8,y+5),caption,font=font,fill="black",spacing=1)
                    canvas.paste(im,(x+(cw-im.width)//2,y+104))
                    draw.rectangle((x,y,x+cw-1,y+hh-1),outline="#bbbbbb")
                y+=hh
            path=OUT/"contact_sheets"/f"train_{name}_{page:02d}.png"
            canvas.save(path)
            records.append({"path":path.relative_to(OUT).as_posix(),"class":name,
                            "annotation_ids":[int(r["annotation_id"]) for r in selected],
                            "count":len(selected),"native_pixels_no_resize":True})
            for im in imgs:
                im.close()
    return records


def main():
    if OUT.exists():
        raise FileExistsError("Never overwrite existing raw crop dataset v1")
    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/convert_coco_detection_fixed_v1.py"),
                    "--verify-only"],cwd=ROOT,check=True)
    frozen=json.loads((FIXED/"split_summary.json").read_text(encoding="utf-8"))
    assignments={r["image_id"]:r for r in read_csv(FIXED/"split_manifest.csv")}
    archives=list(detection.SOURCE.glob("*.zip"))
    assert len(archives)==1
    archive=archives[0]
    protected={archive.relative_to(ROOT).as_posix():detection.sha256(archive)}
    assert protected[archive.relative_to(ROOT).as_posix()]==frozen["source"]["input_sha256"][archive.relative_to(ROOT).as_posix()]
    for root in (FIXED,DATA):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".jpg",".jpeg",".png",".txt",".yaml",".csv",".json",".md"):
                protected[path.relative_to(ROOT).as_posix()]=detection.sha256(path)
    with zipfile.ZipFile(archive) as z:
        member=next(n for n in z.namelist() if n.endswith("_annotations.coco.json"))
        coco_bytes=z.read(member)
        data=json.loads(coco_bytes)
    images={str(im["id"]):im for im in data["images"]}
    categories={c["id"]:c["name"] for c in data["categories"]}
    assert len(assignments)==len(images)==629 and set(images)==set(assignments)
    assert Counter(a["split"] for a in assignments.values())==Counter(detection.EXPECTED_IMAGES)
    assert len(data["annotations"])==len({a["id"] for a in data["annotations"]})==5953
    ann_by_image=defaultdict(list)
    issues=[]
    rows=[]
    counts={s:Counter() for s in detection.EXPECTED_IMAGES}
    for ann in sorted(data["annotations"],key=lambda a:a["id"]):
        sid=str(ann["image_id"])
        a,im=assignments[sid],images[sid]
        name=categories[ann["category_id"]]
        assert name in CLASSES
        assert a["filename"]==im["file_name"]
        x,y,w,h=map(float,ann["bbox"])
        image_path=DATA/a["split"]/"images"/a["filename"]
        stem=Path(a["filename"]).stem
        filename=f"{stem}__ann_{ann['id']}__class_{name}.png"
        box=pixel_bounds(ann["bbox"])
        row={
            "crop_filename":filename,"split":a["split"],"class":name,"source_filename":a["filename"],
            "source_image_id":ann["image_id"],"annotation_id":ann["id"],
            "bbox_x":x,"bbox_y":y,"bbox_w":w,"bbox_h":h,
            "source_width":im["width"],"source_height":im["height"],
            "crop_width":box[2]-box[0],"crop_height":box[3]-box[1],
            "wafer":a["wafer"],"position":a["position"],
            "original_filename":a["original_filename"],"capture_group":a["group_key"],
            "exposure":a["exposure"],"source_path":image_path.relative_to(ROOT).as_posix(),
            "source_sha256":a["sha256"],
            "crop_relative_path":f"{a['split']}/{name}/{filename}",
            "crop_x_start":box[0],"crop_y_start":box[1],
            "crop_x_end":box[2],"crop_y_end":box[3],
            "bbox_fractional":any(v!=int(v) for v in (x,y,w,h)),
            "bbox_area":w*h,"crop_aspect_ratio":(box[2]-box[0])/(box[3]-box[1]) if box[3]>box[1] else None,
            "source_mode":"","crop_mode":"","crop_pixel_sha256":"","crop_file_sha256":"",
        }
        if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:
            issue(issues,"invalid_bbox",row,"Non-finite or nonpositive source bbox")
        if x<0 or y<0 or x+w>im["width"] or y+h>im["height"]:
            issue(issues,"bbox_outside_source",row,"Source bbox outside image; no clipping permitted")
        if box[2]<=box[0]:
            issue(issues,"nonpositive_crop_width",row,str(box))
        if box[3]<=box[1]:
            issue(issues,"nonpositive_crop_height",row,str(box))
        if not (0<=box[0]<box[2]<=im["width"] and 0<=box[1]<box[3]<=im["height"]):
            issue(issues,"bbox_outside_source",row,"Rasterized window outside image; no padding/clipping")
        if not image_path.is_file():
            issue(issues,"missing_source_image",row,str(image_path))
        counts[a["split"]][name]+=1
        ann_by_image[sid].append(row)
        rows.append(row)
    for split,expected in detection.EXPECTED_CLASSES.items():
        assert {name:counts[split][name] for name in CLASSES}==expected
    assert len({r["crop_filename"] for r in rows})==5953
    OUT.mkdir(parents=True)
    (OUT/"contact_sheets").mkdir()
    if issues:
        csv_write(OUT/"qa_issues.csv",issues,ISSUE_FIELDS)
        json_write(OUT/"dataset_summary.json",{"success":False,"qa_issues":issues})
        raise RuntimeError("Source preflight issues; no automatic fixes or crops generated")
    for split in detection.EXPECTED_IMAGES:
        for name in CLASSES:
            (OUT/split/name).mkdir(parents=True)
    for i,(sid,a) in enumerate(sorted(assignments.items(),key=lambda pair:int(pair[0]))):
        source=DATA/a["split"]/"images"/a["filename"]
        assert detection.sha256(source)==a["sha256"]
        try:
            with Image.open(source) as image:
                image.load()
                if image.size!=(images[sid]["width"],images[sid]["height"]):
                    issue(issues,"source_size_mismatch",{"source_image_id":sid},str(image.size))
                    raise RuntimeError("Source image dimensions differ from COCO")
                if image.mode!="RGB":
                    raise RuntimeError("Unexpected source mode; do not automatically change pixel channels")
                for row in ann_by_image[sid]:
                    box=tuple(row[k] for k in ("crop_x_start","crop_y_start","crop_x_end","crop_y_end"))
                    crop=image.crop(box)
                    target=OUT/row["crop_relative_path"]
                    assert not target.exists()
                    crop.save(target,format="PNG")
                    row.update(source_mode=image.mode,crop_mode=crop.mode,
                               crop_pixel_sha256=pixel_sha(crop),
                               crop_file_sha256=detection.sha256(target))
        except OSError as exc:
            issue(issues,"unreadable_source",{"source_image_id":sid},str(exc))
            raise
        if (i+1)%100==0:
            print(f"CROPPED_SOURCES={i+1}/629",flush=True)
    csv_write(OUT/"crop_manifest.csv",rows)
    saved=read_csv(OUT/"crop_manifest.csv")
    assert len(saved)==5953 and len({r["annotation_id"] for r in saved})==5953
    validate(saved,issues,assignments)
    stats=distribution(saved)
    csv_write(OUT/"class_distribution.csv",stats)
    sheets=make_contact_sheets(rows) if not issues else []
    csv_write(OUT/"qa_issues.csv",issues,ISSUE_FIELDS)
    qa_counts={key:sum(i["issue_type"]==key for i in issues) for key in QA_KEYS}
    preserved=all(detection.sha256(ROOT/path)==h for path,h in protected.items())
    assert preserved
    zero_images=[{"image_id":sid,"filename":a["filename"],"split":a["split"]}
                 for sid,a in assignments.items() if not ann_by_image[sid]]
    assert len(zero_images)==6
    widths=[r["crop_width"] for r in rows]
    heights=[r["crop_height"] for r in rows]
    fractional=sum(r["bbox_fractional"] for r in rows)
    byclass=Counter(r["class"] for r in rows)
    summary={
        "success":not issues,"status":"CREATED_PENDING_VISUAL_QA" if not issues else "FAILED_QA",
        "total_crops":len(rows),"source_image_count":629,
        "source_split_counts":detection.EXPECTED_IMAGES,
        "split_crop_counts":dict(Counter(r["split"] for r in rows)),
        "split_class_crop_counts":{s:{n:counts[s][n] for n in CLASSES} for s in counts},
        "class_counts":{n:byclass[n] for n in CLASSES},
        "class_percent":{n:100*byclass[n]/len(rows) for n in CLASSES},
        "unique_sources_with_crops":len(ann_by_image)-sum(not rr for rr in ann_by_image.values()),
        "source_images_with_crops_per_split":{s:sum(bool(ann_by_image[sid]) and a["split"]==s
                                                   for sid,a in assignments.items()) for s in counts},
        "zero_annotation_sources":zero_images,
        "zero_annotation_policy":"Six human-confirmed negative source images remain in the fixed split but have no defect bbox, hence no crop. No normal class or artificial negative crop is created.",
        "source_coco_archive":archive.relative_to(ROOT).as_posix(),
        "source_coco_member":member,"source_coco_json_sha256":hashlib.sha256(coco_bytes).hexdigest(),
        "source_coco_json_digest_note":"Exact unmodified COCO JSON ZIP-member bytes.",
        "fixed_split_manifest":(FIXED/"split_manifest.csv").relative_to(ROOT).as_posix(),
        "protected_input_sha256":protected,"all_protected_inputs_unchanged":preserved,
        "class_mapping":"COCO category NAME p/o/s/sc retained; original numeric IDs are not folder labels. No new classifier index is assigned.",
        "crop_format":"Lossless RGB PNG; exact decoded source pixel values verified for every crop.",
        "raster_rule":"Half-open source bbox [x,x+w) x [y,y+h). Select integer pixel indices whose centers (index+0.5) are inside: start=ceil(start_float-0.5), end=ceil(end_float-0.5), end exclusive. Original float bbox unchanged. No context margin, padding, resizing, interpolation, normalization, color conversion or clipping.",
        "fractional_bboxes":fractional,
        "max_endpoint_quantization_px":max(abs(float(row[k])-v) for row in rows
            for k,v in (("crop_x_start",row["bbox_x"]),("crop_y_start",row["bbox_y"]),
                        ("crop_x_end",row["bbox_x"]+row["bbox_w"]),("crop_y_end",row["bbox_y"]+row["bbox_h"]))),
        "crop_size_range":{"width_min":min(widths),"width_max":max(widths),
                           "height_min":min(heights),"height_max":max(heights)},
        "distribution":stats,"qa_error_counts":qa_counts,"qa_issue_count":len(issues),
        "pixel_exact_readback_verified":len(saved)-qa_counts["crop_pixel_mismatch"],
        "contact_sheets":sheets,"contact_sheet_selection_seed":SEED,
        "contact_sheet_rule":"Train only: seeded random25 p/sc, all2 open/all5 short. Pages up to6 crops; crop pixels pasted at native size, no resizing in sheets.",
        "training_performed":False,"oversampling":False,"augmentation":False,
        "warnings":[f"{fractional} fractional bboxes require pixel-center rasterization; original float bbox and actual integer boundaries both retained.",
                    "Val/o is an intentionally empty class folder (zero GT); no samples added.",
                    "Six zero-label source images produce no defect crops by design.",
                    "A crop keeps its source annotation class even if another annotated structure appears inside it; no relabeling or crop-content adjudication performed."],
    }
    json_write(OUT/"dataset_summary.json",summary)
    table=["| Split | p | o | s | sc | Total crops | Source images (all / with bbox) |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for split,expected in detection.EXPECTED_CLASSES.items():
        table.append(f"| {split} | {expected['p']} | {expected['o']} | {expected['s']} | {expected['sc']} | {sum(expected.values())} | {detection.EXPECTED_IMAGES[split]} / {summary['source_images_with_crops_per_split'][split]} |")
    disttable=["| Split/class | Count | Width mean / median / min / max [px] | Height mean / median / min / max [px] | Aspect ratio mean / median | Bbox area mean / median [px2] |",
               "|---|---:|---|---|---|---|"]
    for st in stats:
        f=lambda k:"N/A" if st[k] is None else f"{st[k]:.3f}"
        disttable.append(f"| {st['split']}/{st['class']} | {st['crop_count']} | "+
            " / ".join(f("width_"+k) for k in ("mean","median","min","max"))+" | "+
            " / ".join(f("height_"+k) for k in ("mean","median","min","max"))+" | "+
            f("aspect_ratio_mean")+" / "+f("aspect_ratio_median")+" | "+
            f("bbox_area_mean")+" / "+f("bbox_area_median")+" |")
    report=f"""# MobileNet raw GT bbox crop dataset v1

## Scope

One unchanged GT bbox -> one lossless native RGB crop -> one existing class name. No training, augmentation, oversampling, synthetic images, bbox edits, relabeling or new split.
Source: validated COCO bbox annotations and data/derived/yolov8_detection_fixed_v1/ image bytes. Assignment source of truth: data/derived/fixed_split_v1/split_manifest.csv.

## Counts and split integrity

{chr(10).join(table)}

Total:5953 crops, identical to original annotation counts. All629 source images retain fixed assignments;6 zero-label negatives have no bbox and therefore produce no crop. No artificial normal class is introduced.
Identity/source SHA/capture-group split checks and exact source-pixel readback were applied to every crop. No source, source hash or capture group spans multiple crop splits.

## Native rasterization and pixel preservation

PNG is used instead of JPEG to avoid lossy crop recompression. Stored RGB pixel arrays are exact source decode slices.
No padding, context margin, resize, interpolation, normalization, color conversion, rotation or correction was applied.
For integer bboxes, crop bounds are the same half-open integer rectangle. For {fractional} fractional bboxes, select pixels whose centers lie inside the original half-open bbox: ceil(x-0.5) to ceil(x+w-0.5), and likewise y. This is native rasterization, not a new expanded/contracted float annotation. Maximum endpoint quantization is {summary['max_endpoint_quantization_px']:.3f}px. Original float coordinates and actual integer crop bounds are both retained in the manifest.
The source bbox itself may be tight or contain other structures; visual QA checks extraction, not annotation validity or class purity. No annotation is corrected or rejudged.

## Distribution

Native crop widths:{min(widths)}..{max(widths)}px; heights:{min(heights)}..{max(heights)}px. These marginal extrema need not describe one crop.
Class imbalance is preserved: p5444 ({summary['class_percent']['p']:.3f}%), o3 ({summary['class_percent']['o']:.3f}%), s7 ({summary['class_percent']['s']:.3f}%), sc499 ({summary['class_percent']['sc']:.3f}%).
Width/height/aspect statistics use saved crop pixels. Bbox area uses original float width*height. Empty Val/open statistics are N/A, not fabricated zeros.

{chr(10).join(disttable)}

## QA and contact sheets

QA issue count:{len(issues)}. Individual error counts are in dataset_summary.json; qa_issues.csv contains issue rows only (header-only when no errors).
All12 split/class directories exist, including empty val/o.
Train-only visual examples: random25 p, random25 sc, all2 o, all5 s; seed42. Crops and contact-sheet pasting are native-pixel/no-resize; the viewer may downscale large sheets. Contact-sheet margins are canvas layout only and are not added to dataset crops.
Review status is recorded in dataset_summary.json. No claim of complete visual annotation adjudication is made.

## Files and integrity

- train/, val/, test/: lossless native crops in p/o/s/sc class folders.
- crop_manifest.csv: full source identity, annotation ID, class, float bbox, integer crop bounds, metadata and pixel/file hashes.
- class_distribution.csv: all12 split/class combinations and descriptive statistics.
- dataset_summary.json: rules, counts, preserved-source hashes, QA and visual sheet records.
- qa_issues.csv: errors only; no automatic clipping, deletion or fixes.
- contact_sheets/: native-pixel extraction review examples.

All{len(protected)} protected input files remain byte-identical. Previous results and production source/config/dashboard are untouched. No commit/push.
"""
    (OUT/"dataset_report.md").write_text(report,encoding="utf-8")
    assert all(detection.sha256(ROOT/path)==h for path,h in protected.items())
    print("CROP_DATASET="+json.dumps({k:summary[k] for k in (
        "success","status","total_crops","split_crop_counts","split_class_crop_counts",
        "crop_size_range","fractional_bboxes","qa_error_counts")}),flush=True)
    if issues:
        raise RuntimeError("Crop QA issues recorded; do not fix annotations automatically")


def finish_visual_review():
    """Record the completed extraction review; never relabel crop content."""
    summary_path=OUT/"dataset_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success"] and summary["status"]=="CREATED_PENDING_VISUAL_QA"
    assert summary["qa_issue_count"]==0 and summary["total_crops"]==5953
    assert all(detection.sha256(ROOT/p)==h for p,h in summary["protected_input_sha256"].items())
    assert len(summary["contact_sheets"])==12
    assert Counter({n:sum(s["count"] for s in summary["contact_sheets"] if s["class"]==n)
                    for n in CLASSES})==Counter({"p":25,"o":2,"s":5,"sc":25})
    for sheet in summary["contact_sheets"]:
        with Image.open(OUT/sheet["path"]) as im:
            im.verify()
    summary.update(status="COMPLETED",visual_qa={
        "reviewer":"assistant visual extraction review, not canonical GT re-adjudication",
        "contact_sheets_inspected":[s["path"] for s in summary["contact_sheets"]],
        "reviewed_train_crops":{"p":25,"o":2,"s":5,"sc":25},
        "scope":"Native crop visibility, layout and extraction consistency only. Full pixel/geometry/source QA covers all5953 crops. Source bbox tightness or mixed visible content is preserved, not repaired.",
        "extraction_issue_observed":False,
    })
    json_write(summary_path,summary)
    report_path=OUT/"dataset_report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8")+
        "\n## Completed visual QA\n\nAll12 native-pixel contact-sheet pages were inspected "
        "(Train p25/sc25/o2/s5). No extraction mismatch was observed. This is not a relabeling "
        "or defect-boundary re-annotation pass; source GT and tightness are preserved.\n",
        encoding="utf-8")
    print("VISUAL_QA_COMPLETE="+json.dumps(summary["visual_qa"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finish-visual-review",action="store_true")
    args=parser.parse_args()
    finish_visual_review() if args.finish_visual_review else main()
