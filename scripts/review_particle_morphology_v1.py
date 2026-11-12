"""Prepare native-pixel particle review sheets; observations are visual notes, never GT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/derived/particle_analysis_v1"
OUT = ROOT / "data/derived/particle_morphology_review_v1"
FIXED = ROOT / "data/derived/fixed_split_v1"
SEED = 42
MARGIN = 16


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iou(a, b):
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    area = x*y
    return area/(a[2]*a[3]+b[2]*b[3]-area) if area else 0.0


def write_csv(rows):
    with (OUT/"review_manifest.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def varied(pool, key, n=2):
    selected=[]
    for wafer in sorted({r["wafer"] for r in pool}):
        seen=set()
        for row in sorted((r for r in pool if r["wafer"]==wafer), key=key):
            if row["image_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["image_id"])
            if len(seen)==n:
                break
    return selected


def native_sheet(name, selected):
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",14)
    title_font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    crops=[Image.open(OUT/r["review_crop_path"]) for r in selected]
    widths=[max(340,im.width+20) for im in crops]
    column_width=max(widths)
    heights=[max(160,max(im.height for im in crops[i:i+2])+94)
             for i in range(0,len(crops),2)]
    sheet=Image.new("RGB",(column_width*2,46+sum(heights)),"white")
    draw=ImageDraw.Draw(sheet)
    draw.text((10,10),name+" | NATIVE PIXELS / no resize",fill="black",font=title_font)
    yy=46
    for row_index,height in enumerate(heights):
        for col in range(2):
            index=row_index*2+col
            if index>=len(crops):
                continue
            r,im=selected[index],crops[index]
            xx=col*column_width
            label=(f"ann {r['annotation_id']} / image {r['image_id']} / {r['wafer']} {r['position']}\n"
                   f"bbox {r['bbox_width']:.2f} x {r['bbox_height']:.2f} px; AR {r['aspect_ratio']:.3f}\n"
                   f"overlap_sc={r['overlap_sc']}; max IoU {r['max_sc_iou']:.4f}")
            draw.text((xx+10,yy+5),label,fill="black",font=f)
            sheet.paste(im,(xx+(column_width-im.width)//2,yy+84))
            draw.rectangle((xx,yy,xx+column_width-1,yy+height-1),outline="#bbbbbb")
        yy+=height
    path=OUT/"contact_sheets"/(name+".png")
    sheet.save(path)
    for im in crops:
        im.close()


def prepare():
    if OUT.exists():
        raise FileExistsError("Never overwrite existing morphology review v1")
    with (BASE/"particle_manifest.csv").open(encoding="utf-8-sig") as f:
        particle=list(csv.DictReader(f))
    with (FIXED/"split_manifest.csv").open(encoding="utf-8-sig") as f:
        mapping={int(r["image_id"]):r for r in csv.DictReader(f)}
    analysis=json.loads((BASE/"particle_summary.json").read_text(encoding="utf-8"))
    source=ROOT/analysis["source"]["coco_archive"]
    assert sha(source)==analysis["source"]["coco_archive_sha256"]
    protected={p.relative_to(ROOT).as_posix():sha(p) for parent in (BASE,FIXED)
               for p in parent.rglob("*") if p.is_file()}
    protected[source.relative_to(ROOT).as_posix()]=sha(source)
    with zipfile.ZipFile(source) as z:
        coco=json.loads(z.read(analysis["source"]["coco_json_member"]))
    images={i["id"]:i for i in coco["images"]}
    categories={c["id"]:c["name"] for c in coco["categories"]}
    annotations={a["id"]:a for a in coco["annotations"]}
    scratch=defaultdict(list)
    for a in coco["annotations"]:
        if categories[a["category_id"]]=="sc":
            scratch[a["image_id"]].append(a)
    rows=[]
    for original in particle:
        aid,iid=int(original["annotation_id"]),int(original["image_id"])
        a=annotations[aid]
        assert categories[a["category_id"]]=="p" and a["image_id"]==iid
        matches=[(s["id"],iou(a["bbox"],s["bbox"])) for s in scratch[iid]]
        matches=sorted((sid,score) for sid,score in matches if score>0)
        assert len(matches)==int(original["overlap_sc_count"])
        row={k:original[k] for k in ("annotation_id","image_id","filename","wafer","split",
                                    "bbox_width","bbox_height","bbox_area","aspect_ratio")}
        for k in ("annotation_id","image_id"):
            row[k]=int(row[k])
        for k in ("bbox_width","bbox_height","bbox_area","aspect_ratio"):
            row[k]=float(row[k])
        row.update(
            overlap_sc=bool(matches),max_sc_iou=max((v for _,v in matches),default=0.0),
            overlapping_sc_annotation_ids=";".join(str(sid) for sid,_ in matches),
            review_descriptor="",ambiguity_flag="",review_note="",review_status="NOT_REVIEWED",
            original_filename=original["original_filename"],position=original["position"],
            exposure=original["exposure"],group_key=mapping[iid]["group_key"],
            normalized_area=float(original["normalized_area"]),selection_groups="",
            source_tight_crop_path=(BASE/ original["crop_path"]).relative_to(ROOT).as_posix(),
            review_crop_path="",context_margin_px="",review_crop_bounds_px="",
        )
        rows.append(row)
    assert len(rows)==5444 and sum(r["overlap_sc"] for r in rows)==882
    clean=[r for r in rows if not r["overlap_sc"]]
    rng=random.Random(SEED)
    random_order={r["annotation_id"]:rng.random() for r in rows}
    tie=lambda r:(r["image_id"],r["annotation_id"])
    groups={
        "01_random_non_overlap":varied(clean,lambda r:random_order[r["annotation_id"]]),
        "02_smallest":varied(clean,lambda r:(r["bbox_area"],*tie(r))),
        "03_largest":varied(clean,lambda r:(-r["bbox_area"],*tie(r))),
        "04_low_aspect_ratio":varied(clean,lambda r:(r["aspect_ratio"],*tie(r))),
        "05_near_square_bbox":varied(clean,lambda r:(abs(math.log(r["aspect_ratio"])),*tie(r))),
        "06_high_aspect_ratio":varied(clean,lambda r:(-r["aspect_ratio"],*tie(r))),
        "07_low_normalized_area":varied(clean,lambda r:(r["normalized_area"],*tie(r))),
        "08_high_normalized_area":varied(clean,lambda r:(-r["normalized_area"],*tie(r))),
        "09_scratch_overlap":varied([r for r in rows if r["overlap_sc"]],
                                   lambda r:(-r["max_sc_iou"],*tie(r))),
    }
    per_image=Counter(r["image_id"] for r in rows)
    chosen_images=varied(clean,lambda r:(-per_image[r["image_id"]],r["image_id"],r["annotation_id"]))
    groups["10_high_count_images"]=chosen_images
    selected={r["annotation_id"]:r for group in groups.values() for r in group}
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"crops_review").mkdir()
    (OUT/"contact_sheets").mkdir()
    with zipfile.ZipFile(source) as z:
        per_image_rows=defaultdict(list)
        for r in selected.values():
            per_image_rows[r["image_id"]].append(r)
        for iid,pp in per_image_rows.items():
            im=images[iid]
            raw=z.read("train/"+im["file_name"])
            assert hashlib.sha256(raw).hexdigest()==mapping[iid]["sha256"]
            with Image.open(io.BytesIO(raw)) as source_image:
                source_image.load()
                for r in pp:
                    x,y,w,h=annotations[r["annotation_id"]]["bbox"]
                    box=(max(0,math.floor(x)-MARGIN),max(0,math.floor(y)-MARGIN),
                         min(im["width"],math.ceil(x+w)+MARGIN),
                         min(im["height"],math.ceil(y+h)+MARGIN))
                    safe=re.sub(r'[<>:"/\\|?*\x00-\x1f]',"_",r["original_filename"])
                    r["review_crop_path"]=f"crops_review/image_{iid}__ann_{r['annotation_id']}__{safe}__context16.png"
                    r["context_margin_px"]=MARGIN
                    r["review_crop_bounds_px"]=";".join(map(str,box))
                    crop=source_image.crop(box)
                    path=OUT/r["review_crop_path"]
                    crop.save(path,format="PNG")
                    with Image.open(path) as saved:
                        assert saved.mode==crop.mode and saved.size==crop.size and saved.tobytes()==crop.tobytes()
    for name,group in groups.items():
        for r in group:
            r["selection_groups"]+=";" if r["selection_groups"] else ""
            r["selection_groups"]+=name
        native_sheet(name,group)
    write_csv(rows)
    summary={
        "status":"AWAITING_VISUAL_REVIEW","total_particles":len(rows),
        "population_non_overlap":len(clean),"population_scratch_overlap":sum(r["overlap_sc"] for r in rows),
        "selected_unique_particles":len(selected),
        "selected_non_overlap":sum(not r["overlap_sc"] for r in selected.values()),
        "selected_scratch_overlap":sum(r["overlap_sc"] for r in selected.values()),
        "selection_groups":{name:[r["annotation_id"] for r in group] for name,group in groups.items()},
        "seed":SEED,"context_margin_px":MARGIN,
        "selection_rule":"2 distinct images per wafer per criterion; clean-first except scratch reference. Ranked within each wafer. Random uses seed42. Groups may overlap; unique annotation IDs deduplicated.",
        "crop_rule":"Native decoded pixels, floor/ceil bbox window +16px per side; context window bounded by image only. Original object bbox unchanged. No resizing, enhancement or color conversion. Native sheet pasting, no scaling.",
        "source_hashes":protected,
        "reviewer":"assistant_visual_review; not independent human-confirmed subtype annotation",
        "all_unreviewed_descriptors_blank":True,
        "population_prevalence_estimation":False,
    }
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    (OUT/"morphology_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="source_hashes"},ensure_ascii=False,indent=2))


def finalize():
    """Record only actually inspected examples; never infer descriptors from geometry."""
    summary=json.loads((OUT/"morphology_summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="AWAITING_VISUAL_REVIEW"
    protected=summary["source_hashes"]
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    with (OUT/"review_manifest.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    observations=json.loads((OUT/"visual_observations.json").read_text(encoding="utf-8"))["observations"]
    expected={aid for ids in summary["selection_groups"].values() for aid in ids}
    assert set(map(int,observations))==expected and len(expected)==78
    for row in rows:
        aid=row["annotation_id"]
        if aid in observations:
            row.update(observations[aid],review_status="VISUALLY_REVIEWED")
            if row["ambiguity_flag"]:
                row["review_descriptor"]+=";ambiguous"
        else:
            assert not any(row[k] for k in ("review_descriptor","ambiguity_flag","review_note"))
    write_csv(rows)
    with (OUT/"review_manifest.csv").open(encoding="utf-8-sig") as f:
        reread=list(csv.DictReader(f))
    reviewed=[r for r in reread if r["review_status"]=="VISUALLY_REVIEWED"]
    clean=[r for r in reviewed if r["overlap_sc"]=="False"]
    overlap=[r for r in reviewed if r["overlap_sc"]=="True"]
    assert len(reread)==5444 and len(reviewed)==78 and len(clean)==68 and len(overlap)==10
    assert len({r["annotation_id"] for r in reread})==5444
    assert all(not any(r[k] for k in ("review_descriptor","ambiguity_flag","review_note"))
               for r in reread if r["review_status"]=="NOT_REVIEWED")

    def statistics(pool):
        result={}
        tags=sorted({tag for r in pool for tag in r["review_descriptor"].split(";")})
        for tag in tags:
            matches=[r for r in pool if tag in r["review_descriptor"].split(";")]
            clear=[r for r in matches if r["ambiguity_flag"]=="False"]
            result[tag]={
                "examples":len(matches),"annotation_ids":[int(r["annotation_id"]) for r in matches],
                "unique_images":len({r["image_id"] for r in matches}),
                "unique_capture_groups":len({r["group_key"] for r in matches}),
                "unique_wafers":len({r["wafer"] for r in matches}),
                "wafer_counts":dict(sorted(Counter(r["wafer"] for r in matches).items())),
                "position_counts":dict(sorted(Counter(r["position"] for r in matches).items())),
                "unambiguous_examples":len(clear),
                "unambiguous_unique_images":len({r["image_id"] for r in clear}),
                "unambiguous_unique_capture_groups":len({r["group_key"] for r in clear}),
                "unambiguous_unique_wafers":len({r["wafer"] for r in clear}),
            }
        return result

    counts=lambda pool:{
        "reviewed":len(pool),"ambiguous":sum(r["ambiguity_flag"]=="True" for r in pool),
        "ambiguous_percent":100*sum(r["ambiguity_flag"]=="True" for r in pool)/len(pool),
        "unique_images":len({r["image_id"] for r in pool}),
        "unique_capture_groups":len({r["group_key"] for r in pool}),
        "wafer_counts":dict(sorted(Counter(r["wafer"] for r in pool).items())),
        "position_counts":dict(sorted(Counter(r["position"] for r in pool).items())),
    }
    summary.update(
        status="VISUAL_REVIEW_COMPLETE",reviewed=counts(reviewed),
        reviewed_non_overlap=counts(clean),reviewed_scratch_overlap=counts(overlap),
        unreviewed_count=len(rows)-len(reviewed),
        descriptors_non_overlap=statistics(clean),descriptors_scratch_overlap=statistics(overlap),
        subdivision_evidence="B. Weak / partial grouping evidence",
        ambiguity_definition="Reviewer cannot confidently isolate or characterize a single particle morphology at native pixels because of faintness, tiny size, multi-component extent or pattern attachment. This is not an annotation validity judgment.",
        descriptor_semantics="Nonexclusive visual review tags, not subtype classes, labels or population prevalence estimates. Dark/bright describe displayed RGB appearance, not photometric measurements.",
        independence_limit="Distinct filenames/images are not proof of independent acquisition. Also report wafer_die capture-group counts. Different wafer groups strengthen recurrence evidence without proving physical or acquisition independence.",
        overlap_caution="overlap_sc means positive bbox IoU only, not verified intersection of visible scratch pixels. The reference cohort is selected by bbox IoU, not confirmed scratch morphology.",
    )
    source=next(ROOT/p for p in protected if p.endswith(".zip"))
    old=json.loads((BASE/"particle_summary.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(source) as z:
        coco=json.loads(z.read(old["source"]["coco_json_member"]))
        images={i["id"]:i for i in coco["images"]}
        annotations={a["id"]:a for a in coco["annotations"]}
        categories={c["id"]:c["name"] for c in coco["categories"]}
        scratch=defaultdict(list)
        for a in coco["annotations"]:
            if categories[a["category_id"]]=="sc":
                scratch[a["image_id"]].append(a)
        for r in reread:
            a=annotations[int(r["annotation_id"])]
            assert categories[a["category_id"]]=="p" and a["image_id"]==int(r["image_id"])
            matches=sorted((s["id"],iou(a["bbox"],s["bbox"])) for s in scratch[a["image_id"]]
                           if iou(a["bbox"],s["bbox"])>0)
            assert r["overlap_sc"]==str(bool(matches))
            assert r["overlapping_sc_annotation_ids"]==";".join(str(sid) for sid,_ in matches)
            assert abs(float(r["max_sc_iou"])-max((v for _,v in matches),default=0))<1e-12
            assert float(r["bbox_width"])==a["bbox"][2] and float(r["bbox_height"])==a["bbox"][3]
        for r in reviewed:
            im=images[int(r["image_id"])]
            box=tuple(map(int,r["review_crop_bounds_px"].split(";")))
            with Image.open(io.BytesIO(z.read("train/"+im["file_name"]))) as original:
                crop=original.crop(box)
                with Image.open(OUT/r["review_crop_path"]) as saved:
                    assert saved.mode==crop.mode and saved.size==crop.size and saved.tobytes()==crop.tobytes()
    assert iou([0,0,10,10],[10,0,10,10])==0
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    summary["qa"]={
        "manifest_rows":len(reread),"unique_annotation_ids":5444,"reviewed_unique":78,
        "unreviewed_descriptors_blank":True,"all_particle_sc_flags_recomputed":True,
        "all_original_bbox_dimensions_unchanged":True,"native_crop_pixel_equality_checked":78,
        "contact_sheet_count":len(list((OUT/"contact_sheets").glob("*.png"))),
        "source_hashes_unchanged":True,"protected_file_count":len(protected),
        "touching_bbox_zero_iou_test":True,"identical_bbox_unit_iou_test":True,
    }
    (OUT/"morphology_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    table=["| Review descriptor (not subtype) | Clean examples | Images | Capture groups | Wafers | Clear examples | Wafer counts | Position counts |",
           "|---|---:|---:|---:|---:|---:|---|---|"]
    for tag in ("round-like","elongated-like","irregular-like","multi-component-like",
                "uniform-intensity-like","mixed-intensity-like","ring-like-appearance",
                "diffuse-edge-like","distinct-edge-like","pattern-attached-like"):
        d=summary["descriptors_non_overlap"].get(tag)
        if d:
            table.append(f"| {tag} | {d['examples']} | {d['unique_images']} | {d['unique_capture_groups']} | {d['unique_wafers']} | {d['unambiguous_examples']} | {d['wafer_counts']} | {d['position_counts']} |")
    group_table=["| Selection sheet | Cards | Ambiguous | Note |","|---|---:|---:|---|"]
    by_id={int(r["annotation_id"]):r for r in reviewed}
    for name,ids in summary["selection_groups"].items():
        n=sum(by_id[aid]["ambiguity_flag"]=="True" for aid in ids)
        note="same annotation set as sheet 02" if name.startswith("07") else "same annotation set as sheet 03" if name.startswith("08") else "within-wafer purposive selection"
        group_table.append(f"| {name} | {len(ids)} | {n} | {note} |")
    content=f"""# Particle morphology visual review v1

## Review scope

Actually inspected {len(reviewed)} unique particles in native-pixel contact sheets: {len(clean)} non-overlap and {len(overlap)} scratch-bbox-overlap references. Source population is 5,444: 4,562 non-overlap / 882 scratch-overlap. The remaining {len(rows)-len(reviewed)} particles have blank descriptors and ambiguity fields.
The ten sheets contain 100 cards but only 78 unique particles; repeated cards count once. Each criterion selects two distinct images per wafer. Extremes are ranked within wafer, not the global ten most extreme values. This is a diversity-oriented review, not a prevalence survey.

## Repeated morphology

Only the non-overlap cohort supports the main morphology observations below. Tags overlap and their counts do not sum to the sample size. Frequencies refer only to this selected review sample.

{chr(10).join(table)}

Small rounded darker centers with soft perimeters recur across all five wafers (examples 5724, 320, 158, 838, 509). Large line-connected irregular bodies or multi-component fields also recur across multiple wafers, but their isolated particle contours are uncertain (2994, 1831, 2293, 4845). These are visual descriptions, not physical material or subtype claims.
Clear isolated elongation is limited: 4523 is a horizontal soft-ended dash; 852 is faint, 3525 meets a pattern line, and 1226 includes separate marks. Therefore a strong independent elongated group is not established.
Ring-like bright pockets occur in 2994 and 1831, with overlap reference 4846, but all are pattern-connected networks. No confidently isolated hollow particle group is established. Tiny crops do not support reliable smooth/rough surface texture assessment.
Full per-tag annotation IDs, wafer/position distributions and unambiguous-subset image/group counts are in morphology_summary.json. Distinct images are not necessarily independent captures; wafer_die capture groups are reported as a conservative additional check, not proof of independence.

## Ambiguous cases

Total {summary['reviewed']['ambiguous']}/{len(reviewed)} ({summary['reviewed']['ambiguous_percent']:.1f}%) are ambiguous under the stated single-particle morphology definition.
Non-overlap: {summary['reviewed_non_overlap']['ambiguous']}/{len(clean)} ({summary['reviewed_non_overlap']['ambiguous_percent']:.1f}%).
Scratch reference: {summary['reviewed_scratch_overlap']['ambiguous']}/{len(overlap)} ({summary['reviewed_scratch_overlap']['ambiguous_percent']:.1f}%).
These are sample review uncertainty rates, not population rates or annotation error rates.
Tiny or faint marks (18, 5110, 1151), boundary-connected bulges (2772, 3141) and multi-mark boxes (1623, 4076, 4159) limit reproducibility of a single outline.
Scratch-overlap references are not pooled into main morphology counts. 5558 and 1702 show oblique multi-spot traces; 4846 is a looped pattern-connected network. In 5209 and 168 a discrete dot remains visible despite bbox overlap; bbox IoU alone does not prove a scratch crosses the particle. No automatic overlap-contaminated morphology label is assigned.

## Geometry relationship

High aspect ratios do not consistently imply a single elongated body: 3356 and 1610 are faint background regions, 341 has multiple marks, and 2653/3452 contain pattern turns or connections. A real elongated-looking dash exists in 4523, but does not validate an aspect-ratio classifier.
Large boxes contain both faint multi-mark areas (1623, 4076, 2101) and large red line-attached bodies (4845, 1239). Large area is not equivalent to clustered morphology.
Near-square boxes include isolated rounded dots (320, 158), pattern bulges (72, 395) and nearly unresolved marks (1151); near-square does not establish a round subtype.

{chr(10).join(group_table)}

## Subdivision evidence

B. Weak / partial grouping evidence.

## Reason

1. Rounded dark-center appearances repeat in multiple images, capture groups and all five wafers.
2. Irregular or multi-component appearances recur, but pattern attachment and faintness obscure a stable single-particle outline.
3. Elongated and ring-like appearances lack sufficient clean, unambiguous independent repetitions in this sample.
4. Geometry and visible morphology are not one-to-one, with continuous edge/contrast variation and substantial review ambiguity.
5. The sample is purposive and one assistant visual review, not independent human reproducibility evidence; no subtype count or class name is fixed.
"""
    (OUT/"morphology_review.md").write_text(content,encoding="utf-8")
    readme=f"""# Particle morphology review v1

Purpose: visually assess repeated morphology only. Conclusion: B, weak / partial grouping evidence. No final subtype, class change or training.

- Sources: data/derived/particle_analysis_v1/particle_manifest.csv and existing native bbox-tight crops; the COCO archive and fixed split are inherited from that analysis.
- Population: 5,444 particle annotations; all have recomputed scratch bbox overlap flags, max IoU and matching sc annotation IDs.
- Actual review: 78 unique particles, 68 non-overlap / 10 scratch-overlap, selected with seed 42 and two distinct images per wafer per criterion. All five wafers are represented. Unreviewed 5,366 rows retain blank descriptors, ambiguity and notes.
- Files: review_manifest.csv (all particles and selected-only notes), morphology_summary.json (counts, recurrence, provenance, QA), morphology_review.md (findings), visual_observations.json (explicit manual observation registry), contact_sheets/ (10 sheets), crops_review/ (78 selected crops).
- Existing tight crop paths remain traceable in source_tight_crop_path. For pattern context, selected-only review crops use floor/ceil bbox bounds plus a fixed 16 native pixels per side, bounded by image dimensions. This changes only the review window, never the annotation bbox.
- No crop resizing, augmentation, enhancement, sharpening or color correction. Contact sheets paste native pixels without scaling. The image viewer may scale a large sheet on screen; stored crops retain native pixels.
- Ten sheets have 100 cards but 78 unique particles. Low/high normalized-area selections exactly repeat the corresponding absolute-area selections here and are deduplicated for every review count.
- Descriptors are nonexclusive assistant visual review tags, not canonical annotation, subtype labels or physical material/intensity measurements. Uncertainty flags concern descriptor visibility, not GT validity.
- Scratch overlap means bbox IoU > 0 only; it is not automatic confirmation of visible scratch contamination. Main and overlap-reference counts are separate.
- Selection favors wafer diversity and extremes; reviewed frequencies cannot estimate population morphology frequencies. Capture group counts supplement filename/image counts without proving acquisition independence.
- QA: all 5,444 original p bbox dimensions and scratch flags checked against COCO; all 78 review crop pixel arrays equal the corresponding native source windows; {len(protected)} protected source/analysis/split file hashes unchanged.
- Reproduce preparation in a fresh output directory only: .venv/Scripts/python.exe -B scripts/review_particle_morphology_v1.py. After actual visual inspection and manually recording observations, --finalize records reviewed-only notes and checks integrity. Existing output is protected against accidental prepare/finalize reruns.

No raw/annotation/split modification, clustering, classifier training, detector retraining or commit/push was performed.
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    print(json.dumps({k:summary[k] for k in ("status","reviewed","reviewed_non_overlap",
          "reviewed_scratch_overlap","unreviewed_count","subdivision_evidence","qa")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize",action="store_true",help="Record explicit manually reviewed observations.")
    args=parser.parse_args()
    finalize() if args.finalize else prepare()

