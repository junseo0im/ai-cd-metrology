"""One V1-compatible YOLOv8n train-only o/s augmentation experiment."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import contextlib
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import yaml
import train_yolov8_detection_baseline_v1 as baseline

ROOT=baseline.ROOT
SOURCE=baseline.DATA
DATA=ROOT/"data/derived/yolov8_detection_fixed_v1_augmented"
REPORT=ROOT/"data/derived/yolov8_v1_augmentation"
RUN=ROOT/"runs/augmentation/yolov8n_v1_augmented"
SUMMARY=REPORT/"augmentation_summary.json"
MANIFEST=ROOT/"data/derived/fixed_split_v1/split_manifest.csv"
BASE_REPORT=ROOT/"data/derived/yolov8_baseline_v1"
SCRIPT=Path(__file__).resolve()
NAMES=baseline.NAMES
EXPECTED_IMAGES={"train":517,"val":62,"test":62}
EXPECTED_ANNOTATIONS={
    "train":{"o":6,"p":4482,"s":15,"sc":433},
    "val":baseline.CLASS_COUNTS["val"],"test":baseline.CLASS_COUNTS["test"],
}


def read_labels(path):
    rows=[]
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        tokens=line.split()
        assert len(tokens)==5,f"V1 must remain bbox-only detection format: {path}"
        cid=int(tokens[0])
        assert str(cid)==tokens[0] and cid in NAMES
        x,y,w,h=map(float,tokens[1:])
        assert all(math.isfinite(v) for v in [x,y,w,h])
        assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
        assert x-w/2>=-1e-9 and y-h/2>=-1e-9 and x+w/2<=1+1e-9 and y+h/2<=1+1e-9
        rows.append((cid,x,y,w,h))
    return rows


def transformed_rows(rows,axis):
    return [(cid,1-x if axis=="horizontal" else x,1-y if axis=="vertical" else y,w,h)
            for cid,x,y,w,h in rows]


def transformed_image(image,axis,brightness,contrast):
    image=ImageOps.mirror(image) if axis=="horizontal" else ImageOps.flip(image)
    image=ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def protected_snapshot():
    result=baseline.immutable_dataset()
    for folder in [baseline.RUN,BASE_REPORT,ROOT/"git_release_v1"]:
        for path in folder.rglob("*"):
            if path.is_file():
                result[path.relative_to(ROOT).as_posix()]=baseline.digest(path)
    for path in [MANIFEST,baseline.WEIGHTS,Path(baseline.__file__).resolve()]:
        result[path.relative_to(ROOT).as_posix()]=baseline.digest(path)
    return result


def unchanged(snapshot):
    return all((ROOT/name).is_file() and baseline.digest(ROOT/name)==value
               for name,value in snapshot.items())


def verify_dataset(summary):
    assert unchanged(summary["source_hashes"]),"V1 source/split/baseline/release changed"
    records=summary["augmentation_records"]
    aug_by_name={r["augmented_filename"]:r for r in records}
    assert len(records)==12 and len(aug_by_name)==12
    with MANIFEST.open(encoding="utf-8-sig",newline="") as file:
        fixed=list(csv.DictReader(file))
    fixed_by_name={r["filename"]:r for r in fixed}
    assert len(fixed_by_name)==629
    assert Counter(r["split"] for r in fixed)==baseline.COUNTS
    groups=defaultdict(set);hash_splits=defaultdict(set)
    image_counts={};class_counts={};all_output_hashes={}
    for split in ["train","val","test"]:
        originals={p.name:p for p in (SOURCE/split/"images").iterdir() if p.is_file()}
        images={p.name:p for p in (DATA/split/"images").iterdir() if p.is_file()}
        expected=set(originals)|(set(aug_by_name) if split=="train" else set())
        assert set(images)==expected and len(images)==EXPECTED_IMAGES[split],f"Image list mismatch {split}"
        labs={p.name:p for p in (DATA/split/"labels").glob("*.txt")}
        assert {p.stem for p in images.values()}=={p.stem for p in labs.values()}
        counts=Counter()
        for name,p in images.items():
            label=DATA/split/"labels"/(p.stem+".txt")
            rows=read_labels(label);counts.update(NAMES[row[0]] for row in rows)
            with Image.open(p) as image:
                image.load()
                size=image.size
            digest=baseline.digest(p)
            hash_splits[digest].add(split)
            all_output_hashes[p.relative_to(DATA).as_posix()]=digest
            all_output_hashes[label.relative_to(DATA).as_posix()]=baseline.digest(label)
            if name in originals:
                source=originals[name]
                assert baseline.digest(source)==digest,"Original image changed"
                assert (SOURCE/split/"labels"/label.name).read_bytes()==label.read_bytes(),"Original label changed"
                assignment=fixed_by_name[name]
                assert assignment["split"]==split and assignment["sha256"]==digest
                groups[assignment["group_key"]].add(split)
            else:
                assert split=="train"
                record=aug_by_name[name]
                parent=SOURCE/"train/images"/record["source_filename"]
                assert fixed_by_name[parent.name]["split"]=="train"
                source_rows=read_labels(SOURCE/"train/labels"/(parent.stem+".txt"))
                assert any(row[0] in {0,2} for row in source_rows),"Non-o/s parent"
                expected_rows=transformed_rows(source_rows,record["axis"])
                assert len(rows)==len(expected_rows)
                for actual,target in zip(rows,expected_rows):
                    assert actual[0]==target[0] and all(abs(a-b)<1e-12 for a,b in zip(actual[1:],target[1:]))
                with Image.open(parent) as im:
                    im=im.convert("RGB")
                    assert size==im.size
                    expected_image=transformed_image(im,record["axis"],record["brightness"],record["contrast"])
                    with Image.open(p) as saved:
                        assert np.array_equal(np.asarray(expected_image),np.asarray(saved.convert("RGB")))
                groups[fixed_by_name[parent.name]["group_key"]].add(split)
        assert {n:counts[n] for n in NAMES.values()}==EXPECTED_ANNOTATIONS[split]
        image_counts[split]=len(images);class_counts[split]={n:counts[n] for n in NAMES.values()}
    assert all(len(x)==1 for x in groups.values()),"Group leakage"
    assert all(len(x)==1 for x in hash_splits.values()),"Exact-hash split leakage"
    config=yaml.safe_load((DATA/"data.yaml").read_text(encoding="utf-8"))
    assert config["names"]==NAMES and Path(config["path"]).resolve()==DATA.resolve()
    for split in ["train","val","test"]:
        assert config[split]==split+"/images"
    all_output_hashes["data.yaml"]=baseline.digest(DATA/"data.yaml")
    if "derived_image_label_sha256" in summary:
        assert all_output_hashes==summary["derived_image_label_sha256"],"Derived training files changed"
    return {"status":"PASS","image_counts":image_counts,"class_counts":class_counts,
            "val_test_byte_identical":True,"original_train_byte_identical":True,
            "augmentation_train_only":True,"all_sources_os_positive":True,
            "image_label_missing":0,"invalid_coordinate_rows":0,"polygon_rows":0,
            "source_group_leakage":0,"exact_hash_split_leakage":0,
            "image_label_sha256":all_output_hashes}


def prepare():
    for path in [DATA,REPORT,RUN]:
        if path.exists():
            raise FileExistsError(f"Never overwrite an experiment: {path}")
    old=json.loads((BASE_REPORT/"baseline_summary.json").read_text(encoding="utf-8"))
    assert old["training_success"] and old["status"]=="COMPLETED"
    assert old["image_counts"]==baseline.COUNTS
    assert old["environment"]["ultralytics"]==importlib.metadata.version("ultralytics")
    assert old["environment"]["torch"]==importlib.metadata.version("torch")
    assert baseline.digest(baseline.WEIGHTS)==old["pretrained_weights_sha256"]
    assert baseline.immutable_dataset()==old["dataset_files_sha256"]
    with MANIFEST.open(encoding="utf-8-sig",newline="") as file:
        fixed=list(csv.DictReader(file))
    assert len(fixed)==629 and len({r["filename"] for r in fixed})==629
    by_name={r["filename"]:r for r in fixed}
    selected=[];original_counts={}
    for split in baseline.COUNTS:
        images=sorted((SOURCE/split/"images").iterdir())
        assert len(images)==baseline.COUNTS[split]
        counts=Counter()
        for image in images:
            rows=read_labels(SOURCE/split/"labels"/(image.stem+".txt"))
            counts.update(NAMES[r[0]] for r in rows)
            assert by_name[image.name]["split"]==split and baseline.digest(image)==by_name[image.name]["sha256"]
            with Image.open(image) as im:
                im.load()
            if split=="train" and any(r[0] in {0,2} for r in rows):
                classes=Counter(NAMES[r[0]] for r in rows)
                selected.append({"filename":image.name,"classes":dict(classes),"group_key":by_name[image.name]["group_key"]})
        assert {n:counts[n] for n in NAMES.values()}==baseline.CLASS_COUNTS[split]
        original_counts[split]=dict(counts)
    source_stats={
        "o_positive":sum(bool(r["classes"].get("o")) for r in selected),
        "s_positive":sum(bool(r["classes"].get("s")) for r in selected),
        "both":sum(bool(r["classes"].get("o")) and bool(r["classes"].get("s")) for r in selected),
        "unique":len(selected)}
    assert source_stats=={"o_positive":2,"s_positive":4,"both":0,"unique":6}
    print("V1_AUG_SOURCE_COUNTS="+json.dumps(source_stats),flush=True)
    snapshot=protected_snapshot()
    DATA.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    for split in baseline.COUNTS:
        for kind in ["images","labels"]:
            (DATA/split/kind).mkdir(parents=True)
            for source in sorted((SOURCE/split/kind).iterdir()):
                if source.is_file() and (kind=="images" or source.suffix==".txt"):
                    shutil.copy2(source,DATA/split/kind/source.name)
    rng=random.Random(42);records=[]
    for selected_source in selected:
        source=SOURCE/"train/images"/selected_source["filename"]
        rows=read_labels(SOURCE/"train/labels"/(source.stem+".txt"))
        for axis in ["horizontal","vertical"]:
            brightness=rng.uniform(.93,1.07);contrast=rng.uniform(.95,1.05)
            filename=source.stem+("__aug_h.png" if axis=="horizontal" else "__aug_v.png")
            with Image.open(source) as im:
                result=transformed_image(im.convert("RGB"),axis,brightness,contrast)
                result.save(DATA/"train/images"/filename,format="PNG")
            label=DATA/"train/labels"/(Path(filename).stem+".txt")
            transformed=transformed_rows(rows,axis)
            label.write_text("".join(str(row[0])+" "+" ".join(format(v,".16g") for v in row[1:])+"\n" for row in transformed),encoding="utf-8")
            records.append({"source_filename":source.name,"source_sha256":baseline.digest(source),
                "source_group_key":selected_source["group_key"],"source_classes":selected_source["classes"],
                "augmented_filename":filename,"axis":axis,"brightness":brightness,"contrast":contrast,"split":"train"})
    yaml_config={"path":DATA.as_posix(),"train":"train/images","val":"val/images","test":"test/images","nc":4,"names":NAMES}
    (DATA/"data.yaml").write_text(yaml.safe_dump(yaml_config,sort_keys=False),encoding="utf-8")
    manifest={"seed":42,"photometric":{"brightness_range":[.93,1.07],"contrast_range":[.95,1.05],
              "order":"PIL RGB -> flip -> ImageEnhance.Brightness -> ImageEnhance.Contrast",
              "note":"Same ranges as general_v2; exact historical operation implementation was not available. No V2 image/annotation/split is used."},
              "source_counts":source_stats,"sources":selected,"records":records}
    baseline.write_json(DATA/"augmentation_manifest.json",manifest)
    summary={"model":old["model"],"status":"PREPARED","training_success":False,"error":None,
        "started_utc":datetime.now(timezone.utc).isoformat(),"requested_epochs":100,"actual_epochs":0,"best_epoch":None,
        "imgsz":640,"batch":8,"seed":42,"source_hashes":snapshot,"source_counts":source_stats,
        "original_image_counts":baseline.COUNTS,"effective_train_samples":517,
        "effective_train_class_counts":EXPECTED_ANNOTATIONS["train"],"augmentation_records":records,
        "augmentation_manifest_sha256":baseline.digest(DATA/"augmentation_manifest.json"),
        "source_dataset":SOURCE.as_posix(),"dataset":DATA.as_posix(),"run":RUN.as_posix(),"report":REPORT.as_posix(),
        "script_path":str(SCRIPT),"script_sha256":baseline.digest(SCRIPT),
        "pretrained_weights_sha256":baseline.digest(baseline.WEIGHTS),"loader_validation":{},
        "test_evaluation_calls":0,"nan_detected":False,"loader_exclusions":0,"warnings":[]}
    qa=verify_dataset(summary)
    summary["derived_image_label_sha256"]=qa.pop("image_label_sha256")
    summary["dataset_qa"]=qa
    RUN.mkdir(parents=True)
    shutil.copy2(SCRIPT,RUN/SCRIPT.name)
    baseline.write_json(SUMMARY,summary)
    baseline.write_json(REPORT/"dataset_qa.json",qa)
    (DATA/"README.md").write_text(
        "# V1-compatible train-only augmentation\n\nOriginal V1 images and labels are copied byte-for-byte.\n"
        "Train505+12=517; Val62/Test62 unchanged. Sources: open-positive2/short-positive4/both0/unique6.\n"
        "Each source has one horizontal and one vertical flip, brightness0.93-1.07, contrast0.95-1.05, seed42.\n"
        "PIL RGB -> flip -> brightness -> contrast. No resize/crop, no V2 data, no annotation edits.\n"
        "V1 labels are bbox-only Detection rows; no polygon conversion is performed.\n",encoding="utf-8")
    print("V1_AUG_DATASET_QA="+json.dumps(qa),flush=True)
    return summary,old


def check_loader(dataset,split,summary):
    expected=summary["effective_train_samples"] if split=="train" else baseline.COUNTS[split]
    expected_counts=summary["effective_train_class_counts"] if split=="train" else baseline.CLASS_COUNTS[split]
    assert len(dataset)==expected and len(dataset.labels)==expected, f"{split}: loader exclusion"
    actual=Counter()
    for label in dataset.labels:
        for cid in label["cls"].reshape(-1):
            actual[NAMES[int(cid)]]+=1
    assert {n:actual[n] for n in NAMES.values()}==expected_counts, f"{split}: loader changed labels"
    return {"images":len(dataset),"class_counts":expected_counts,"excluded_images":0}


def train(summary,old):
    runtime=RUN/"runtime_config"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"]=str(runtime)
    os.environ["YOLO_AUTOINSTALL"]="false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import SETTINGS,USER_CONFIG_DIR
    assert torch.__version__==old["environment"]["torch"]
    assert ultralytics.__version__==old["environment"]["ultralytics"]
    SETTINGS.update({k:False for k in ("sync","wandb","mlflow","comet","clearml",
                                     "dvc","neptune","raytune","tensorboard","hub") if k in SETTINGS})
    font=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/arial.ttf"
    if font.is_file() and not (USER_CONFIG_DIR/"Arial.ttf").exists():
        shutil.copy2(font,USER_CONFIG_DIR/"Arial.ttf")
    settings=old["requested_train_args"].copy()
    settings.update(data=str(DATA/"data.yaml"),project=str(RUN.parent),name=RUN.name)
    allowed={"data","project","name"}
    assert {k for k in settings if settings[k]!=old["requested_train_args"][k]}==allowed
    for key in old["augmentation_disabled"]:
        assert key in DEFAULT_CFG_DICT and settings[key]==0
    assert settings["auto_augment"] is None and not settings["augment"]
    assert (settings["imgsz"],settings["batch"],settings["seed"],settings["patience"],
            settings["epochs"],settings["deterministic"])==(640,8,42,20,100,True)
    assert str(settings["device"])==old["device"]
    if settings["device"]!="cpu":
        assert torch.cuda.is_available(), "Baseline device unavailable; do not silently switch"
    if importlib.util.find_spec("albumentations") is not None:
        raise RuntimeError("Unexpected Albumentations installation versus baseline; do not change pipeline")
    summary.update(status="TRAINING",requested_train_args=settings,
                   device=str(settings["device"]),workers=settings["workers"],
                   environment={"python":platform.python_version(),"torch":torch.__version__,
                                "ultralytics":ultralytics.__version__},
                   baseline_train_args_differences=sorted(allowed))
    baseline.write_json(SUMMARY,summary)
    (REPORT/"training_config.yaml").write_text(yaml.safe_dump({"model":str(baseline.WEIGHTS),
                                              **settings},sort_keys=False),encoding="utf-8")
    model=YOLO(str(baseline.WEIGHTS),task="detect")
    assert model.task=="detect"
    started=time.perf_counter()

    def on_start(trainer):
        summary["loader_validation"]["train"]=check_loader(trainer.train_loader.dataset,"train",summary)
        summary["loader_validation"]["val"]=check_loader(trainer.test_loader.dataset,"val",summary)
        summary["effective_training_args"]=vars(trainer.args).copy()
        # Auto optimizer must resolve exactly as baseline, not silently change due to dataset size.
        baseline_effective=old["effective_training_args"]
        for key in ("optimizer","lr0","lrf","momentum","weight_decay","warmup_epochs",
                    "warmup_momentum","warmup_bias_lr","cos_lr"):
            assert getattr(trainer.args,key)==baseline_effective[key], f"Unexpected optimizer/scheduler change: {key}"
        assert type(trainer.optimizer).__name__==baseline_effective["optimizer"]
        assert all(abs(g["initial_lr"]-baseline_effective["lr0"])<1e-12
                   for g in trainer.optimizer.param_groups)
        summary["resolved_optimizer_matches_baseline"]=True
        (REPORT/"training_config.yaml").write_text(yaml.safe_dump(vars(trainer.args),sort_keys=False),encoding="utf-8")
        baseline.write_json(SUMMARY,summary)

    def on_epoch_start(trainer):
        summary["actual_epochs"]=int(trainer.epoch)+1

    def on_fit_end(trainer):
        if int(trainer.epoch)+1!=summary.get("actual_epochs"):
            return
        metrics={k:float(v) for k,v in trainer.metrics.items()}
        assert all(math.isfinite(v) for v in metrics.values()), "Non-finite train metrics"
        summary.update(latest_epoch_metrics=metrics,elapsed_training_seconds=time.perf_counter()-started)
        baseline.write_json(SUMMARY,summary)
        print("V1_AUG_PROGRESS="+json.dumps({"epoch":summary["actual_epochs"],
              "best_epoch":summary["best_epoch"],"metrics":metrics}),flush=True)

    def on_save(trainer):
        if trainer.best_fitness==trainer.fitness:
            summary["best_epoch"]=int(trainer.epoch)+1

    for name,callback in (("on_train_start",on_start),("on_train_epoch_start",on_epoch_start),
                          ("on_fit_epoch_end",on_fit_end),("on_model_save",on_save)):
        model.add_callback(name,callback)
    model.train(**settings)
    summary["training_time_seconds"]=time.perf_counter()-started
    assert Path(model.trainer.save_dir).resolve()==RUN.resolve()
    best,last=RUN/"weights/best.pt",RUN/"weights/last.pt"
    for path in (best,last,RUN/"results.csv",RUN/"results.png",RUN/"args.yaml"):
        assert path.is_file(), f"Missing training artifact: {path}"
    assert summary["best_epoch"]
    with (RUN/"results.csv").open(encoding="utf-8-sig") as f:
        epochs=list(csv.DictReader(f))
    assert len(epochs)==summary["actual_epochs"]
    summary.update(status="EVALUATING",best_pt_path=str(best),last_pt_path=str(last))
    baseline.write_json(SUMMARY,summary)
    best_model=YOLO(str(best),task="detect")
    for split in ("val","test"):
        def on_val_start(validator,split=split):
            summary["loader_validation"][split+"_best"]=check_loader(validator.dataloader.dataset,split,summary)
        best_model.reset_callbacks()
        best_model.add_callback("on_val_start",on_val_start)
        if split=="test":
            assert summary["test_evaluation_calls"]==0
            summary["test_evaluation_calls"]=1
            baseline.write_json(SUMMARY,summary)
        metrics=best_model.val(data=str(DATA/"data.yaml"),split=split,imgsz=640,batch=8,
                              device=settings["device"],workers=0,plots=True,augment=False,
                              project=str(RUN),name=split+"_best",exist_ok=False,verbose=True)
        summary[split]=baseline.evaluation_result(metrics,split)
        baseline.write_json(SUMMARY,summary)
    finish(summary,old)


def finish(summary,old):
    qa=verify_dataset(summary)
    qa.pop("image_label_sha256")
    assert qa==summary["dataset_qa"]
    assert baseline.digest(DATA/"augmentation_manifest.json")==summary["augmentation_manifest_sha256"]
    assert baseline.digest(SCRIPT)==summary["script_sha256"]==baseline.digest(RUN/SCRIPT.name)
    summary["test_delta_vs_baseline"]={k:summary["test"]["overall"][k]-old["test"]["overall"][k]
        for k in ["precision","recall","map50","map50_95"]}
    summary["test_per_class_comparison"]={name:{k:{
        "baseline":old["test"]["per_class"][name][k],"augmented":summary["test"]["per_class"][name][k],
        "delta":summary["test"]["per_class"][name][k]-old["test"]["per_class"][name][k]}
        for k in ["precision","recall","ap50","ap50_95"]} for name in ["p","o","s","sc"]}
    summary["rare_class_observations"]={name:{
        "test_gt_instances":baseline.CLASS_COUNTS["test"][name],
        "reported_recall":summary["test"]["per_class"][name]["recall"],
        "ap50":summary["test"]["per_class"][name]["ap50"],
        "note":"One GT instance only; numeric observation, not generalization evidence."}
        for name in ["o","s"]}
    text=(REPORT/"training.log").read_text(encoding="utf-8",errors="replace")
    warnings=sorted(set(line.strip() for line in text.splitlines()
        if "WARNING" in line or "warning:" in line.lower()))
    summary.update(status="COMPLETED",training_success=True,error=None,warnings=warnings,
        nan_detected=False,loader_exclusions=0,test_evaluation_completed=True,
        source_dataset_and_baseline_unchanged=True,val_test_byte_identical=True,
        source_group_leakage=0,script_and_config_verified=True,
        ended_utc=datetime.now(timezone.utc).isoformat())
    baseline.write_json(SUMMARY,summary)
    rows=[]
    for key in ["precision","recall","map50","map50_95"]:
        rows.append({"metric":key,"baseline":old["test"]["overall"][key],
                     "augmented":summary["test"]["overall"][key],"change":summary["test_delta_vs_baseline"][key]})
    for name in ["o","s"]:
        rows.append({"metric":name+"_recall","baseline":old["test"]["per_class"][name]["recall"],
                     "augmented":summary["test"]["per_class"][name]["recall"],
                     "change":summary["test_per_class_comparison"][name]["recall"]["delta"]})
    with (REPORT/"performance_comparison.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# V1-compatible YOLOv8n augmentation","",
        "- Baseline images Train505 / Val62 / Test62; augmented Train517 / Val62 / Test62.",
        "- Original V1 images/labels and Val/Test preserved byte-for-byte; fixed group split unchanged.",
        "- Source counts: o-positive2 / s-positive4 / both0 / unique6; two variants each, total12.",
        "- Horizontal/vertical flips + brightness0.93-1.07 and contrast0.95-1.05; seed42.",
        "- Photometric implementation: PIL RGB -> flip -> Brightness -> Contrast. Same V2 ranges, but no V2 image/label/split is used.",
        "- No oversampling multiplier, weighted loss, focal loss, synthetic data or extra online augmentation.",
        "- Same official yolov8n.pt SHA256 and baseline requested settings except data/project/name.",
        "- Same epochs100, imgsz640, batch8, seed42, patience20, workers0, CPU, deterministic=True.",
        "- Baseline auto optimizer policy retained; effective AdamW/lr0=0.00125 and other optimizer/scheduler args verified.",
        f"- Actual epochs {summary['actual_epochs']}; best epoch {summary['best_epoch']}.",
        "- Best selection uses unchanged installed Ultralytics detection fitness, not Test metrics.",
        "- Same standalone Val/Test evaluation args and GT; no threshold tuning.","",
        "## Test comparison","",
        "| Metric | Baseline | Augmented | Change |","|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['baseline']:.8f} | {row['augmented']:.8f} | {row['change']:+.8f} |")
    lines+=["","## Test class results","",
        "| Class | GT | Baseline P | Augmented P | Baseline R | Augmented R | Baseline AP50 | Augmented AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["p","o","s","sc"]:
        b=old["test"]["per_class"][name];a=summary["test"]["per_class"][name]
        lines.append(f"| {name} | {a['instances']} | {b['precision']:.8f} | {a['precision']:.8f} | {b['recall']:.8f} | {a['recall']:.8f} | {b['ap50']:.8f} | {a['ap50']:.8f} |")
    lines+=["","- Test open1/short1 are individual observations; no generalization claim.",
        "- These variants do not increase independent morphology. Steps per epoch change because Train has12 additional samples.",
        "- No baseline retraining; old baseline results are read from the saved completed-run JSON.","",
        "## Reproducibility","",
        f"- Script: {SCRIPT}.",f"- Script SHA256: {summary['script_sha256']}.",
        f"- Dataset: {DATA}.",f"- Run: {RUN}.",f"- Best checkpoint: {RUN/'weights/best.pt'}.",
        f"- Test evaluation calls: {summary['test_evaluation_calls']}.",
        "- Dataset QA PASS; missing image/label0, invalid coordinate0, polygon rows0, group leakage0.",
        "- Source V1, fixed manifest, baseline artifacts and Git release hashes unchanged.",
        "- NaN none; loader exclusions0. Warnings: "+("; ".join(warnings) if warnings else "none")+".",""]
    (REPORT/"augmentation_results.md").write_text("\n".join(lines),encoding="utf-8")
    print("V1_AUG_COMPLETED="+json.dumps({k:summary[k] for k in [
        "training_success","source_counts","effective_train_samples","actual_epochs","best_epoch",
        "test","test_delta_vs_baseline","rare_class_observations","best_pt_path","warnings"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        old=json.loads((BASE_REPORT/"baseline_summary.json").read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and not (REPORT/"training.log").exists()
        assert summary["script_sha256"]==baseline.digest(SCRIPT)
        verify_dataset(summary)
    else:
        summary,old=prepare()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8",buffering=1) as log:
        with contextlib.redirect_stdout(baseline.Tee(sys.stdout,log)),contextlib.redirect_stderr(baseline.Tee(sys.stderr,log)):
            try:
                train(summary,old)
            except BaseException as error:
                summary.update(status="FAILED",training_success=False,error=f"{type(error).__name__}: {error}",
                               ended_utc=datetime.now(timezone.utc).isoformat())
                baseline.write_json(SUMMARY,summary)
                traceback.print_exc()
                raise


if __name__=="__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()


