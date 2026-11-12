"""Fixed v1 rare-class image repetitions only; no model/data parameter search."""
from __future__ import annotations

import argparse
import contextlib
import csv
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import yaml
import train_yolov8_detection_baseline_v1 as baseline

ROOT=Path(__file__).resolve().parents[1]
SOURCE=baseline.DATA
DATA=ROOT/"data/derived/yolov8_detection_oversample_v1"
REPORT=ROOT/"data/derived/yolov8_oversampling_v1"
RUN=ROOT/"runs/oversampling/yolov8n_oversample_v1"
MANIFEST=ROOT/"data/derived/fixed_split_v1/split_manifest.csv"
BASE_REPORT=ROOT/"data/derived/yolov8_baseline_v1"
NAMES=baseline.NAMES
SUMMARY=REPORT/"oversampling_summary.json"


def repeat_factor(classes):
    return max([1]+[{"o":10,"s":5,"sc":2}.get(c,1) for c in classes])


def snapshot():
    protected=baseline.immutable_dataset()
    roots=(baseline.RUN,BASE_REPORT)
    for root in roots:
        protected.update({p.relative_to(ROOT).as_posix():baseline.digest(p)
                          for p in root.rglob("*") if p.is_file()})
    for p in (MANIFEST,baseline.WEIGHTS,Path(baseline.__file__)):
        protected[p.relative_to(ROOT).as_posix()]=baseline.digest(p)
    return protected


def unchanged(protected):
    return all((ROOT/p).is_file() and baseline.digest(ROOT/p)==h for p,h in protected.items())


def read_labels(path):
    classes=Counter()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        tokens=line.split()
        if not tokens:
            continue
        assert len(tokens)==5, f"Not a detection row: {path}"
        cid=int(tokens[0])
        assert str(cid)==tokens[0] and cid in NAMES
        x,y,w,h=map(float,tokens[1:])
        assert all(math.isfinite(v) for v in (x,y,w,h))
        assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
        classes[NAMES[cid]]+=1
    return classes


def prepare():
    for path in (DATA,REPORT,RUN):
        if path.exists():
            raise FileExistsError(f"Never overwrite an experiment: {path}")
    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/convert_coco_detection_fixed_v1.py"),
                    "--verify-only"],cwd=ROOT,check=True)
    old=json.loads((BASE_REPORT/"baseline_summary.json").read_text(encoding="utf-8"))
    assert old["training_success"] and old["status"]=="COMPLETED"
    assert old["environment"]["ultralytics"]==importlib.metadata.version("ultralytics")
    assert old["environment"]["torch"]==importlib.metadata.version("torch")
    assert baseline.digest(baseline.WEIGHTS)==old["pretrained_weights_sha256"]
    with MANIFEST.open(encoding="utf-8-sig") as f:
        assignments=list(csv.DictReader(f))
    assert len(assignments)==629 and len({r["image_id"] for r in assignments})==629
    assert len({r["filename"] for r in assignments})==629
    groups={}
    by_split=Counter()
    source_counts={split:Counter() for split in baseline.COUNTS}
    rows=[]
    all_hashes=set()
    for a in assignments:
        split,filename=a["split"],a["filename"]
        assert split in baseline.COUNTS
        group=a["group_key"]
        if group:
            assert group not in groups or groups[group]==split
            groups[group]=split
        image=SOURCE/split/"images"/filename
        label=SOURCE/split/"labels"/(Path(filename).stem+".txt")
        assert image.is_file() and label.is_file()
        h=baseline.digest(image)
        assert h==a["sha256"] and h not in all_hashes
        all_hashes.add(h)
        cc=read_labels(label)
        assert all(cc[name]==int(a[name+"_count"]) for name in NAMES.values())
        source_counts[split].update(cc)
        by_split[split]+=1
        if split=="train":
            factor=repeat_factor(cc)
            for index in range(1,factor+1):
                target=f"{Path(filename).stem}__os{index:02d}{Path(filename).suffix}"
                rows.append({
                    "oversampled_filename":target,"source_filename":filename,
                    "repeat_index":index,"repeat_factor":factor,
                    "source_classes":";".join(n for n in NAMES.values() if cc[n]),
                    "image_id":a["image_id"],"source_group_key":group,
                    "source_sha256":h,"source_label_sha256":baseline.digest(label),
                })
    assert dict(by_split)==baseline.COUNTS
    assert all(dict(source_counts[s])=={k:v for k,v in baseline.CLASS_COUNTS[s].items() if v}
               for s in by_split)
    assert repeat_factor({"p":1,"o":1,"s":1,"sc":1})==10
    assert repeat_factor({"p":1,"s":1,"sc":1})==5 and repeat_factor({})==1
    protected=snapshot()
    REPORT.mkdir(parents=True)
    DATA.mkdir(parents=True)
    RUN.mkdir(parents=True)
    effective=Counter()
    for split in baseline.COUNTS:
        for kind in ("images","labels"):
            (DATA/split/kind).mkdir(parents=True)
        if split=="train":
            for r in rows:
                for kind,sname,dname in (
                    ("images",r["source_filename"],r["oversampled_filename"]),
                    ("labels",Path(r["source_filename"]).stem+".txt",
                     Path(r["oversampled_filename"]).stem+".txt")):
                    src=SOURCE/split/kind/sname
                    dst=DATA/split/kind/dname
                    shutil.copy2(src,dst)
                    assert baseline.digest(src)==baseline.digest(dst)
                effective.update(read_labels(DATA/split/"labels"/(Path(r["oversampled_filename"]).stem+".txt")))
        else:
            for a in assignments:
                if a["split"]!=split:
                    continue
                for kind,name in (("images",a["filename"]),
                                  ("labels",Path(a["filename"]).stem+".txt")):
                    src=SOURCE/split/kind/name
                    dst=DATA/split/kind/name
                    shutil.copy2(src,dst)
                    assert baseline.digest(src)==baseline.digest(dst)
    for split,expected in (("train",len(rows)),("val",62),("test",62)):
        ims=list((DATA/split/"images").iterdir())
        labs=list((DATA/split/"labels").glob("*.txt"))
        assert len(ims)==expected and len(labs)==expected
        assert {p.stem for p in ims}=={p.stem for p in labs}
    config={"path":DATA.as_posix(),"train":"train/images","val":"val/images",
            "test":"test/images","nc":4,"names":NAMES}
    (DATA/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    with (REPORT/"oversampling_manifest.csv").open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (REPORT/"oversampling_manifest.csv").open(encoding="utf-8-sig") as f:
        saved=list(csv.DictReader(f))
    assert len(saved)==len(rows) and len({r["oversampled_filename"] for r in saved})==len(rows)
    per_source=Counter(r["source_filename"] for r in rows)
    assert len(per_source)==505
    assert all(per_source[r["source_filename"]]==r["repeat_factor"] for r in rows)
    factors=Counter(next(r["repeat_factor"] for r in rows if r["source_filename"]==filename)
                    for filename in per_source)
    rare_images={n:sum(int(a[n+"_count"])>0 for a in assignments if a["split"]=="train")
                 for n in ("o","s","sc")}
    summary={
        "model":"yolov8n.pt","status":"PREPARED","training_success":False,"error":None,
        "started_utc":datetime.now(timezone.utc).isoformat(),"requested_epochs":100,
        "actual_epochs":0,"best_epoch":None,"imgsz":640,"batch":8,"seed":42,
        "source_dataset":SOURCE.as_posix(),"dataset":DATA.as_posix(),
        "fixed_split_manifest":MANIFEST.as_posix(),"source_hashes":protected,
        "original_image_counts":baseline.COUNTS,"effective_train_samples":len(rows),
        "original_unique_train_images":505,"repeat_factors":{"o":10,"s":5,"sc":2,"other":1},
        "repeat_policy":"Maximum present factor, never product; total copies include original exposure.",
        "source_image_counts_by_repeat_factor":dict(sorted(factors.items())),
        "original_train_rare_class_image_counts":rare_images,
        "original_train_class_counts":baseline.CLASS_COUNTS["train"],
        "effective_train_class_counts":{n:effective[n] for n in NAMES.values()},
        "val_test_byte_identical":True,"image_and_label_copy_method":"copy2; no resampling or label rewriting",
        "unique_morphology_increase":False,"source_group_leakage":0,
        "manifest_rows":len(rows),"manifest_unique_sources":505,
        "baseline_summary_sha256":baseline.digest(BASE_REPORT/"baseline_summary.json"),
        "pretrained_weights_sha256":baseline.digest(baseline.WEIGHTS),
        "loader_validation":{},
        "intended_experimental_difference":"Train image repetition only; same validation/test, architecture, initialization and training/evaluation settings.",
        "comparison_caveat":"Oversampling changes steps per epoch and early stopping exposure; optimizer/scheduler policies are not tuned.",
    }
    baseline.write_json(SUMMARY,summary)
    (DATA/"README.md").write_text(
        "# YOLOv8 detection oversampling v1\n\n"
        f"Original Train: 505 unique; effective Train: {len(rows)} samples; Val/Test: 62/62.\n\n"
        "Fixed factors o=10, s=5, sc=2, otherwise=1; take maximum, not product. "
        "Factors count total copies, including first exposure. No new morphology is created.\n\n"
        "Images and labels copied byte-for-byte using copy2; train filenames have __osNN suffixes. "
        "Val/Test are byte-identical to the fixed dataset, with unchanged filenames. "
        "No image/bbox/class/annotation/split editing or augmentation.\n\n"
        "Source mapping: ../yolov8_oversampling_v1/oversampling_manifest.csv\n",
        encoding="utf-8")
    assert unchanged(protected)
    print("OVERSAMPLING_PREFLIGHT="+json.dumps({k:summary[k] for k in (
        "effective_train_samples","original_unique_train_images","source_image_counts_by_repeat_factor",
        "original_train_rare_class_image_counts","effective_train_class_counts","val_test_byte_identical")}),flush=True)
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
        print("OVERSAMPLING_PROGRESS="+json.dumps({"epoch":summary["actual_epochs"],
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
        metrics=best_model.val(data=str(DATA/"data.yaml"),split=split,imgsz=640,batch=8,
                              device=settings["device"],workers=0,plots=True,augment=False,
                              project=str(RUN),name=split+"_best",exist_ok=False,verbose=True)
        summary[split]=baseline.evaluation_result(metrics,split)
        baseline.write_json(SUMMARY,summary)
    assert unchanged(summary["source_hashes"]), "Protected source/baseline/split changed"
    # Recheck every train copy and unchanged eval bytes after loader/training completed.
    with (REPORT/"oversampling_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            assert baseline.digest(DATA/"train/images"/r["oversampled_filename"])==r["source_sha256"]
            assert baseline.digest(DATA/"train/labels"/(Path(r["oversampled_filename"]).stem+".txt"))==r["source_label_sha256"]
    for split in ("val","test"):
        for kind in ("images","labels"):
            for src in (SOURCE/split/kind).iterdir():
                assert baseline.digest(src)==baseline.digest(DATA/split/kind/src.name)
    summary["test_delta_vs_baseline"]={
        key:summary["test"]["overall"][key]-old["test"]["overall"][key]
        for key in ("precision","recall","map50","map50_95")}
    summary["test_class_ap50_delta_vs_baseline"]={
        n:summary["test"]["per_class"][n]["ap50"]-old["test"]["per_class"][n]["ap50"]
        for n in ("p","o","s","sc")}
    log=(REPORT/"training.log").read_text(encoding="utf-8",errors="replace")
    warnings=sorted(set(l.strip() for l in log.splitlines()
                        if "WARNING" in l or "warning:" in l.lower()))
    summary.update(status="COMPLETED",training_success=True,error=None,warnings=warnings,
                   nan_detected=False,loader_exclusions=0,
                   source_dataset_and_baseline_unchanged=True,val_test_byte_identical=True,
                   training_copies_byte_identical=True,test_evaluation_completed=True,
                   ended_utc=datetime.now(timezone.utc).isoformat())
    summary["artifacts"]=sorted(p.relative_to(ROOT).as_posix() for p in RUN.rglob("*") if p.is_file())
    baseline.write_json(SUMMARY,summary)
    lines=[
        "# YOLOv8n rare-class oversampling v1","",
        f"Original Train 505 unique; effective Train {summary['effective_train_samples']}; Val/Test 62/62.",
        "Total repeat factors: o x10, s x5, sc x2, other x1. Maximum, not product.",
        "Repetition creates no new unique morphology. No additional augmentation or other experiment.",
        f"Epochs {summary['actual_epochs']}/100; best epoch {summary['best_epoch']}.",
        "Baseline settings unchanged except data path and run output paths; resolved optimizer checked.",
        "imgsz 640; batch 8; seed 42; patience 20; deterministic True; workers 0.",
        "Device: "+summary["device"]+". All baseline augmentations remain explicitly OFF.",
        "Copies made with copy2, no re-encoding or label changes. Val/Test are byte-identical.",
        "Training steps per epoch change as a direct consequence of oversampling.","",
        "| Split | Precision | Recall | mAP50 | mAP50-95 |","|---|---:|---:|---:|---:|",
    ]
    for split in ("val","test"):
        values=summary[split]["overall"]
        lines.append("| "+split+" | "+" | ".join(f"{values[k]:.6f}" for k in ("precision","recall","map50","map50_95"))+" |")
    lines+=["","| Test class | GT | Precision | Recall | AP50 | AP50-95 | Delta AP50 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for n in ("p","o","s","sc"):
        c=summary["test"]["per_class"][n]
        lines.append(f"| {n} | {c['instances']} | "+" | ".join(f"{c[k]:.6f}" for k in ("precision","recall","ap50","ap50_95"))+f" | {summary['test_class_ap50_delta_vs_baseline'][n]:+.6f} |")
    lines+=["","| Test metric | Baseline | Oversampling | Delta |","|---|---:|---:|---:|"]
    for k,v in summary["test_delta_vs_baseline"].items():
        lines.append(f"| {k} | {old['test']['overall'][k]:.6f} | {summary['test']['overall'][k]:.6f} | {v:+.6f} |")
    lines+=["","Val has no open GT. Test open and short have only one GT each: numeric results only.",
            f"Best checkpoint: {best}",f"Run: {RUN}",
            "Errors: none. NaN: none. Loader exclusions: 0.",
            "Warnings: "+("; ".join(warnings) or "none"),
            "Execution: .venv/Scripts/python.exe -X utf8 -B -u scripts/train_yolov8_oversampling_v1.py",""]
    (REPORT/"oversampling_results.md").write_text("\n".join(lines),encoding="utf-8")
    print("OVERSAMPLING_COMPLETE="+json.dumps({k:summary[k] for k in (
        "training_success","effective_train_samples","actual_epochs","best_epoch","val","test",
        "test_delta_vs_baseline","test_class_ap50_delta_vs_baseline","best_pt_path","warnings")}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--train-prepared",action="store_true")
    args=parser.parse_args()
    assert not (args.prepare_only and args.train_prepared)
    if args.train_prepared:
        summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
        old=json.loads((BASE_REPORT/"baseline_summary.json").read_text(encoding="utf-8"))
        assert summary["status"]=="PREPARED" and unchanged(summary["source_hashes"])
        assert not (REPORT/"training.log").exists()
    else:
        summary,old=prepare()
    if args.prepare_only:
        return
    with (REPORT/"training.log").open("x",encoding="utf-8") as log:
        with contextlib.redirect_stdout(baseline.Tee(sys.stdout,log)),contextlib.redirect_stderr(baseline.Tee(sys.stderr,log)):
            try:
                train(summary,old)
            except BaseException as e:
                summary.update(status="FAILED",training_success=False,error=f"{type(e).__name__}: {e}",
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
