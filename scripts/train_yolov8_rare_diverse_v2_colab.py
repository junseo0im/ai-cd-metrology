"""E6 diverse augmentation on read-only E1 originals; Colab CUDA training only.

No weighted sampler, class weights, loss/model modifications or new online
augmentation. --qa-only does local dataset integrity checks without a model.
Only Val67 is used for metrics; Test66/external20 are never sent to a model.
All custom E6 changes are OFFLINE pixels/bboxes recorded in the dataset manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import yaml
from PIL import Image

DRIVE = Path("/content/drive/MyDrive/ai-cd-metrology")
DATA = DRIVE / "datasets/yolov8_detection_rare_diverse_v2"
WEIGHTS = DRIVE / "models/yolov8n.pt"
RUN = DRIVE / "runs/v2_rare_diverse/yolov8n_rare_diverse_v2"
COUNTS = {"train":744,"val":67,"test":66}
BASE_COUNTS = {"train":604,"val":67,"test":66}
NAMES = {0: "o", 1: "p", 2: "s", 3: "sc"}
CLASS_COUNTS = {
    "train":{"o":245,"p":6094,"s":280,"sc":679},
    "val": {"o": 4, "p": 593, "s": 4, "sc": 54},
    "test": {"o": 6, "p": 580, "s": 5, "sc": 53},
}
E1_MANIFEST_SHA256 = "b0ce5cd816ef2dd3a0a8d8652785844d009a66dcf06505cb03485e56ad9f1b11"
E1_PARENT_SHA256 = "2f537e1db78385cb0671a3965f97a36e27982a524adad0e520dd34f035b9e4c7"
MANIFEST_SHA256 = "d73d197d8bd195f722baf4080a6945e77433e3d5b3f211f4741d594bdddceb91"
PREPARATION_SHA256 = "654afa69a56db289715d4ac3bd2bb9c90f9d2ddc43d2629dc8f3ac854f42b83a"
REFERENCE_SHA256 = "9ceb08d53f9be2b89fffa8954c91f1821e50fef0514eac8eb5408d63f17eacdf"
PRETRAINED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
ZERO_AUGMENTATIONS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "erasing", "bgr", "cutmix", "multi_scale",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_info(info):
    (RUN/"experiment_info.json").write_text(
        json.dumps(info, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def gpu_preflight(torch):
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    name = torch.cuda.get_device_name(0) if available and count > 0 else None
    result = {"cuda_available": available, "cuda_device_count": count, "gpu_model": name}
    print("GPU_PREFLIGHT=" + json.dumps(result), flush=True)
    require(available and count > 0, "CUDA GPU unavailable. Select a Colab GPU runtime; CPU fallback is forbidden.")
    return result


def verify_dataset(folder):
    require(digest(folder/"preparation_qa.json") == PREPARATION_SHA256,"Frozen E6 QA metadata changed")
    qa = json.loads((folder/"preparation_qa.json").read_text(encoding="utf-8"))
    require(qa["dataset_qa"] == qa["status"] == "PASS" and qa["conversion_success"] is True and
            qa["experiment"] == "E6" and qa["source_manifest_sha256"] == E1_MANIFEST_SHA256,
            "E6 preparation/provenance mismatch")
    require(digest(folder/"image_manifest.csv") == qa["image_manifest_sha256"] == MANIFEST_SHA256,
            "Frozen E6 manifest changed")
    require(digest(folder/"baseline_reference.json") == qa["baseline_reference_sha256"] == REFERENCE_SHA256,
            "E1 training reference changed")
    expected_metadata = {
        "augmentation_manifest.json":"augmentation_manifest_sha256",
        "augmentation_manifest.csv":"augmentation_csv_sha256","parent_manifest.csv":"parent_manifest_sha256",
        "rejected_transforms.json":"rejected_transforms_sha256",
    }
    for name,key in expected_metadata.items():
        require(digest(folder/name) == qa[key],"E6 transform metadata changed: "+name)
    cfg = yaml.safe_load((folder/"data_colab.yaml").read_text(encoding="utf-8"))
    require(cfg["path"] == DATA.as_posix() and cfg["nc"] == 4 and cfg["names"] == NAMES and
            all(cfg[s] == s+"/images" for s in COUNTS),"Frozen Colab paths/classes changed")
    require(qa["evaluation_policy"] == {"selection_split":"val","test_evaluation_allowed":False,
                                      "test_used_for_selection":False,"external_20_used":False},
            "Val-only/external20 policy changed")
    require(digest(folder/"E1_image_manifest.csv") == E1_MANIFEST_SHA256 and
            digest(folder/"E1_augmentation_manifest.json") == E1_PARENT_SHA256,"Frozen E1 identity changed")
    with (folder/"image_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    with (folder/"E1_image_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        base = list(csv.DictReader(f))
    current = {r["image_id"]:r for r in rows}
    require(len(rows) == len(current) == len({r["filename"] for r in rows}) == 877 and
            dict(Counter(r["split"] for r in rows)) == COUNTS,"E6 image identity/count mismatch")
    require(len(base) == 737 and dict(Counter(r["split"] for r in base)) == BASE_COUNTS,"E1 base mismatch")
    for old in base:
        require(old["image_id"] in current and all(current[old["image_id"]][k] == v for k,v in old.items()) and
                current[old["image_id"]]["e6_augmentation"] == "False","An E1 original/augmented row changed")
    old_aug = json.loads((folder/"E1_augmentation_manifest.json").read_text(encoding="utf-8-sig"))
    parents = {str(r["parent_image_id"]) for r in old_aug["records"]}
    with (folder/"parent_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        parent_rows = list(csv.DictReader(f))
    require(len(parents) == len(parent_rows) == 35 and
            {r["image_id"] for r in parent_rows} == parents,"Not the same E1 35 parents")
    for r in parent_rows:
        require(all(current[r["image_id"]][k] == v for k,v in r.items()) and
                r["split"] == "train" and int(r["o_count"])+int(r["s_count"]) > 0,
                "Parent metadata/identity differs")
    augmentation = json.loads((folder/"augmentation_manifest.json").read_text(encoding="utf-8"))
    records = augmentation["records"]
    new = [r for r in rows if r["e6_augmentation"] == "True"]
    require(augmentation["seed"] == 42 and len(records) == len(new) == 140 and
            Counter(r["variant"] for r in records) == {"A":35,"B":35,"C":35,"D":35} and
            {str(r["parent_image_id"]) for r in records} == parents and
            all(sum(str(r["parent_image_id"]) == p for r in records) == 4 for p in parents),
            "E6 augmentation family/budget/parent set differs")
    require({r["augmented_filename"] for r in records} == {r["filename"] for r in new} and
            all(r["split"] == "train" for r in new),"E6 augmentation leaked or missing")
    groups,hashes,counts,zeros = defaultdict(set),set(),{},{}
    for split in COUNTS:
        chosen = [r for r in rows if r["split"] == split]
        expected = {r["filename"] for r in chosen}
        require({p.name for p in (folder/split/"images").iterdir()} == expected and
                {p.name for p in (folder/split/"labels").iterdir()} ==
                {Path(n).stem+".txt" for n in expected},"Missing/extra image-label pair")
        total,zero = Counter(),0
        for row in chosen:
            image = folder/split/"images"/row["filename"]
            label = folder/split/"labels"/(image.stem+".txt")
            require(digest(image) == row["image_sha256"] and digest(label) == row["label_sha256"],
                    "Image/label bytes differ")
            require(row["image_sha256"] not in hashes,"Exact duplicate image")
            hashes.add(row["image_sha256"])
            groups[row["group_key"]].add(split)
            if split == "test":
                total.update({n:int(row[n+"_count"]) for n in NAMES.values()})
                zero += row["zero_annotation"] == "True"
                continue  # Test file identity only; no image decode or label parsing.
            with Image.open(image) as im:
                require(im.size == (int(row["width"]),int(row["height"])),"Image size changed")
                im.verify()
            if image.suffix.lower() in {".jpg",".jpeg"}:
                with image.open("rb") as f:
                    f.seek(-2,2)
                    require(f.read() == bytes([255,217]),"Refuse possible automatic JPEG repair")
            parsed = label.read_text(encoding="ascii").splitlines()
            per_image = Counter()
            for line in parsed:
                t = line.split()
                require(len(t) == 5,"Only Detection bbox labels")
                cid = int(t[0])
                x,y,w,h = map(float,t[1:])
                require(cid in NAMES and str(cid) == t[0] and all(math.isfinite(v) for v in [x,y,w,h]) and
                        0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                        x-w/2 >= -1e-12 and y-h/2 >= -1e-12 and x+w/2 <= 1+1e-12 and y+h/2 <= 1+1e-12,
                        "Invalid Detection bbox")
                per_image[NAMES[cid]] += 1
            require(len(parsed) == int(row["annotation_count"]) and
                    all(per_image[n] == int(row[n+"_count"]) for n in NAMES.values()),"Annotation count changed")
            total.update(per_image)
            zero += not parsed
        counts[split] = {n:total[n] for n in NAMES.values()}
        zeros[split] = zero
    require(counts == CLASS_COUNTS and zeros == {"train":5,"val":0,"test":1},"Class/zero-label counts changed")
    require(all(len(v) == 1 for v in groups.values()),"Group leakage")
    require(qa["val_identical"] and qa["test_identical"] and qa["external_20_used"] is False and
            qa["removed_annotations"] == 0,"Preservation policy mismatch")
    return {"status":"PASS","image_counts":COUNTS,"class_counts":counts,"zero_labels":zeros,
            "manifest_sha256":MANIFEST_SHA256,"reference_sha256":REFERENCE_SHA256,
            "parent_count":35,"new_augmentation_count":140,"all_E1_preserved":True,
            "val_identical":True,"test_identical":True,"declared_group_leakage":0,
            "exact_duplicates":0,"test_evaluated":False,"external_20_used":False,
            "physical_source_identity_verified":False}


def training_settings(reference):
    settings = reference["requested_train_args"].copy()
    settings.update(data=str(DATA/"data_colab.yaml"), device=0,
                    project=str(RUN.parent), name=RUN.name)
    settings.setdefault("optimizer", "auto")
    require(settings["epochs"] == 100 and settings["imgsz"] == 640 and settings["batch"] == 8 and
            settings["seed"] == 42 and settings["patience"] == 20 and settings["workers"] == 0 and
            settings["deterministic"] and settings["pretrained"], "Baseline training parameters changed")
    require(settings["amp"] is False and settings["auto_augment"] is None and
            settings["augment"] is False and settings["close_mosaic"] == 0 and
            all(settings[k] == 0 for k in ZERO_AUGMENTATIONS), "Baseline augmentation/AMP condition changed")
    require(settings["optimizer"] == "auto" and settings.get("split", "val") == "val" and
            not settings.get("resume", False), "E6 requires fresh auto-optimizer training and Val only")
    return settings

def loader_check(dataset, split):
    require(len(dataset) == len(dataset.labels) == COUNTS[split], "Loader excluded an image: "+split)
    counts = Counter()
    for label in dataset.labels:
        for cid in label["cls"].reshape(-1):
            counts[NAMES[int(cid)]] += 1
    require({n:counts[n] for n in NAMES.values()} == CLASS_COUNTS[split],
            "Loader dropped/changed annotation: "+split)
    return {"image_count":len(dataset), "class_counts":dict(counts), "excluded":0}

def run_training(info, reference, torch, YOLO):
    settings = training_settings(reference)
    model = YOLO(str(WEIGHTS), task="detect")
    require(model.task == "detect", "Only YOLOv8n Detection is permitted")
    info["requested_training_args"] = settings
    save_info(info)

    def on_start(trainer):
        from torch.utils.data import RandomSampler
        from ultralytics.models.yolo.detect import DetectionTrainer
        require(type(trainer) is DetectionTrainer,"E6 must use the native detection trainer")
        require(type(trainer.train_loader.sampler) is RandomSampler and
                not trainer.train_loader.sampler.replacement and len(trainer.train_loader.sampler) == 744,
                "E1 unweighted sampling policy changed")
        require(trainer.args.cls_pw == 0.0,"Class weighting must stay OFF")
        info["sampling_policy"] = {"type":"RandomSampler","replacement":False,"samples_per_epoch":744,
                                   "weighted_sampling":False}

        require(trainer.device.type == "cuda" and trainer.batch_size == 8, "GPU/batch changed; no fallback or auto-tuning")
        require(Path(trainer.save_dir).resolve() == RUN.resolve(), "Unexpected run path")
        actual = vars(trainer.args).copy()
        old = reference["effective_training_args"]
        allowed = {"model", "data", "device", "project", "name", "save_dir"}
        differences = {k:{"baseline":old.get(k), "colab":actual.get(k)}
                       for k in old.keys() | actual.keys() if old.get(k) != actual.get(k)}
        require(set(differences) <= allowed, "Unexpected hyperparameter drift: "+str(differences))
        require(type(trainer.optimizer).__name__ == "AdamW", "Baseline optimizer changed")
        groups = [{k:v for k,v in g.items() if k != "params"} for g in trainer.optimizer.param_groups]
        require(all(math.isclose(float(g["lr"]), .00125, abs_tol=1e-12) and
                    tuple(g["betas"]) == (.9, .999) for g in groups), "Actual optimizer/lr/betas changed")
        info.update(status="TRAINING", effective_training_args=actual,
                    argument_differences_from_baseline=differences,
                    optimizer={"name":type(trainer.optimizer).__name__,
                               "defaults":trainer.optimizer.defaults, "parameter_groups":groups},
                    scheduler={"name":type(trainer.scheduler).__name__,
                               "cos_lr":actual["cos_lr"], "lrf":actual["lrf"],
                               "warmup_epochs":actual["warmup_epochs"],
                               "warmup_momentum":actual["warmup_momentum"],
                               "warmup_bias_lr":actual["warmup_bias_lr"]},
                    loader_qa={s:loader_check(d,s) for s,d in [
                        ("train",trainer.train_loader.dataset), ("val",trainer.test_loader.dataset)]},
                    deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                    channels_last_model=any(p.ndim == 4 and p.is_contiguous(memory_format=torch.channels_last)
                                            for p in trainer.model.parameters()))
        (RUN/"training_config.yaml").write_text(yaml.safe_dump(actual, sort_keys=False), encoding="utf-8")
        save_info(info)

    def on_epoch_start(trainer):
        info["actual_completed_epochs"] = int(trainer.epoch)
        info["current_epoch"] = int(trainer.epoch)+1

    def on_batch_start(trainer):
        require(trainer.batch_size == 8 and trainer.args.batch == 8, "Automatic batch reduction is not permitted")

    def on_save(trainer):
        if trainer.best_fitness == trainer.fitness:
            info["best_epoch"] = int(trainer.epoch)+1
            save_info(info)

    def on_batch_end(trainer):
        if not info.get("native_unweighted_loss_verified",False):
            from ultralytics.utils.loss import v8DetectionLoss
            from ultralytics.utils.torch_utils import unwrap_model
            criterion = unwrap_model(trainer.model).criterion
            require(type(criterion) is v8DetectionLoss and criterion.class_weights is None and
                    criterion.bce.weight is None and criterion.bce.pos_weight is None,
                    "E6 requires the unchanged native unweighted v8DetectionLoss")
            info["native_unweighted_loss_verified"] = True
            save_info(info)

    def on_epoch_end(trainer):
        if int(trainer.epoch)+1 != info.get("current_epoch"):
            return  # Ignore the extra callback used by Ultralytics final best validation.
        metrics = {k:float(v) for k,v in trainer.metrics.items()}
        require(all(math.isfinite(v) for v in metrics.values()), "Non-finite training/validation metric")
        info.update(actual_completed_epochs=int(trainer.epoch)+1, latest_validation_metrics=metrics)
        save_info(info)
        print("E6_RARE_DIVERSE_PROGRESS="+json.dumps({
            "epoch":info["actual_completed_epochs"], "best_epoch":info["best_epoch"]}), flush=True)

    for event, callback in [
        ("on_train_start",on_start), ("on_train_epoch_start",on_epoch_start),
        ("on_train_batch_start",on_batch_start), ("on_train_batch_end",on_batch_end), ("on_model_save",on_save),
        ("on_fit_epoch_end",on_epoch_end)]:
        model.add_callback(event,callback)
    started = time.perf_counter()
    model.train(**settings)
    with (RUN/"results.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == info["actual_completed_epochs"] and info["best_epoch"] is not None, "Epoch bookkeeping mismatch")
    require(all(math.isfinite(float(v)) for row in rows for v in row.values()), "Non-finite results.csv value")
    for name in ["args.yaml","results.csv","weights/best.pt","weights/last.pt"]:
        require((RUN/name).is_file() and (RUN/name).stat().st_size > 0, "Missing training artifact: "+name)
    require(verify_dataset(DATA) == info["dataset_qa"] and digest(WEIGHTS) == PRETRAINED_SHA256,
            "Dataset or initial weights changed")
    require(digest(Path(__file__)) == info["script_sha256"] == digest(RUN/Path(__file__).name), "Executing script changed")
    fit_seconds = time.perf_counter()-started
    require(info.get("native_unweighted_loss_verified",False),"Loss verification did not run")
    info["early_stopping"] = {"triggered":info["actual_completed_epochs"] < settings["epochs"],
                              "patience":20,"stopper_best_epoch":int(model.trainer.stopper.best_epoch)}
    evaluate_best_validation(info,YOLO)
    info.update(status="COMPLETED", training_success=True, error=None,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                training_seconds=fit_seconds,
                best_pt_path=str(RUN/"weights/best.pt"), last_pt_path=str(RUN/"weights/last.pt"),
                test_evaluated=False,validation_evaluated=True)
    save_info(info)
    (RUN/"experiment_summary.md").write_text("\n".join([
        "# E6 rare diverse / Colab GPU", "",
        "- Status: COMPLETED; Train744 / Val67 / Test66, fixed general_v2 split.",
        "- Requested epochs100; completed "+str(info["actual_completed_epochs"])+"; best "+str(info["best_epoch"])+".",
        "- imgsz640 / batch8 / seed42 / patience20 / device0 / workers0 / deterministic=True.",
        "- Baseline optimizer=auto policy retained; actual AdamW, lr0.00125, betas(0.9,0.999) verified.",
        "- AMP=False; all baseline online augmentation disabled. Offline E6 diverse augmentation only; no weighted sampling/class weights.",
        "- Dataset hash QA and loader image/annotation counts passed.",
        "- Val67 alone selects/compares checkpoints. Test66 identity/hash only; no inference or external20 usage.",
        "- GPU, Python/Torch/CUDA and Ultralytics must match the completed corrected baseline.",
        "- Full environment, optimizer groups, hashes and actual args: experiment_info.json.",
        "- Best: "+info["best_pt_path"], "",
    ]), encoding="utf-8")
    print("E6_RARE_DIVERSE_COMPLETED="+json.dumps({
        k:info[k] for k in ["training_success","actual_completed_epochs","best_epoch","best_pt_path"]}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true",help="Colab GPU/E1/data preflight only; no training/inference")
    parser.add_argument("--qa-only",action="store_true",help="CPU-safe local dataset integrity only")
    parser.add_argument("--dataset-root",type=Path,help="Local E6 dataset; --qa-only only")
    parser.add_argument("--qa-output",type=Path,help="New preparation report; --qa-only only")
    args = parser.parse_args()
    require(not(args.check_only and args.qa_only),"Select one check mode")
    require(args.qa_only or(args.dataset_root is None and args.qa_output is None),
            "Training paths are fixed")
    if args.qa_only:
        report = {"dataset_qa":verify_dataset(args.dataset_root or DATA),
                  "script_sha256":digest(Path(__file__)),"training_executed":False,"inference_executed":False,
                  "test_evaluated":False,"external_20_used":False,
                  "colab_gpu_and_actual_E1_run_preflight":"PENDING_ON_COLAB"}
        if args.qa_output:
            require(not args.qa_output.exists(),"Never overwrite an existing QA report")
            args.qa_output.parent.mkdir(parents=True,exist_ok=True)
            args.qa_output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    gpu = gpu_preflight(torch)
    require(int(os.environ.get("WORLD_SIZE","1")) == 1,"Match E1 single GPU policy")
    prepared = json.loads((DATA/"baseline_reference.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("ultralytics")
    require(version == prepared["ultralytics_version"] == "8.4.174","Ultralytics version drift")
    require(importlib.util.find_spec("albumentations") is None,"Albumentations must be absent to match E1")
    qa = verify_dataset(DATA)
    corrected = corrected_run_reference(torch,gpu,prepared)
    reference = general_run_reference(torch,gpu,prepared,corrected)
    require(digest(WEIGHTS) == PRETRAINED_SHA256 == reference["pretrained_weights_sha256"],
            "Official pretrained weights differ")
    settings = training_settings(reference)
    require(reference["effective_training_args"]["cls_pw"] == 0.0,"Class weighting policy changed")
    print("DATASET_QA="+json.dumps(qa),flush=True)
    print("TRAINING_ARGS="+json.dumps(settings),flush=True)
    if args.check_only:
        print("READY: E6 Colab GPU/E1/data preflight PASS; no training/inference.",flush=True)
        return
    require(not RUN.exists(),"Run already exists; no overwrite/resume")
    RUN.mkdir(parents=True,exist_ok=False)
    script = Path(__file__).resolve()
    shutil.copy2(script,RUN/script.name)
    shutil.copy2(DATA/"preparation_qa.json",RUN/"dataset_preparation_qa.json")
    runtime = RUN/"runtime_config"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    info = {
        "status":"PREPARING","training_success":False,"error":None,
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"platform":platform.platform(),
        "torch":str(torch.__version__),"torchvision":importlib.metadata.version("torchvision"),
        "ultralytics":version,"cuda_version":torch.version.cuda,
        "cudnn_version":torch.backends.cudnn.version(),**gpu,"device":0,
        "experiment":"E6","dataset_path":str(DATA),"data_yaml":str(DATA/"data_colab.yaml"),"dataset_qa":qa,
        "train_val_test_image_counts":COUNTS,"model":"yolov8n.pt",
        "pretrained_weights_path":str(WEIGHTS),"pretrained_weights_sha256":PRETRAINED_SHA256,
        "epochs_requested":100,"actual_completed_epochs":0,"best_epoch":None,
        "imgsz":640,"batch":8,"seed":42,"script_path":str(script),"script_sha256":digest(script),
        "test_evaluated":False,"test_evaluation_calls":0,"validation_evaluation_calls":0,
        "selection_split":"val","validation_only":True,"external_20_used":False,
        "comparison_baselines":reference["verified_run"],
        "weighted_sampling":False,"class_weights":None,"custom_loss":False,"custom_trainer":False,
    }
    save_info(info)
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in [
            "sync","wandb","mlflow","comet","clearml","dvc","neptune","raytune","tensorboard","hub"] if k in SETTINGS})
        from unittest.mock import patch
        with patch("ultralytics.data.dataset.save_dataset_cache_file",side_effect=lambda *a,**k: None):
            run_training(info,reference,torch,YOLO)
    except BaseException as error:
        info.update(status="FAILED",training_success=False,error=f"{type(error).__name__}: {error}",
                    ended_utc=datetime.now(timezone.utc).isoformat())
        save_info(info)
        traceback.print_exc()
        raise


CORRECTED_RUN = DRIVE/"runs/v2_baseline/yolov8n_corrected_v2"
CORRECTED_MANIFEST_SHA256 = "f6f884d2079e9154e164da7a4548f08dcca23f86fab5cfe8268a8b08808fde9f"
GENERAL_RUN = DRIVE/"runs/v2_augmented/yolov8n_general_v2"
GENERAL_SCRIPT_SHA256 = "4c3e06c6b2631cee44834ce633361f35d72fc7488ead5bcae56a7403ec677eab"

def corrected_run_reference(torch, gpu, prepared):
    needed = ["experiment_info.json","args.yaml","results.csv","weights/best.pt",
              "train_yolov8_corrected_v2_colab.py"]
    require(all((CORRECTED_RUN/n).is_file() for n in needed),
            "Completed corrected GPU run artifacts missing on Drive: "+str(CORRECTED_RUN))
    info = json.loads((CORRECTED_RUN/"experiment_info.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((CORRECTED_RUN/"args.yaml").read_text(encoding="utf-8"))
    require(info.get("status") == "COMPLETED" and info.get("training_success") is True and
            info.get("train_val_test_image_counts") == {"train":534,"val":67,"test":66} and
            info.get("actual_completed_epochs") == prepared["expected_reported_actual_epochs"] and
            info.get("best_epoch") == prepared["expected_reported_best_epoch"], "Unexpected/incomplete corrected GPU run")
    require(info.get("model") == "yolov8n.pt" and info.get("pretrained_weights_sha256") == PRETRAINED_SHA256 and
            info.get("dataset_qa",{}).get("manifest_sha256") == CORRECTED_MANIFEST_SHA256,
            "Corrected model or evaluated dataset identity differs")
    require(info.get("script_sha256") == prepared["corrected_script_sha256"] ==
            digest(CORRECTED_RUN/"train_yolov8_corrected_v2_colab.py"),
            "Archived corrected script differs; baseline preflight/augmentation policy is unverified")
    environment = {
        "python":platform.python_version(), "torch":str(torch.__version__),
        "torchvision":importlib.metadata.version("torchvision"),
        "ultralytics":importlib.metadata.version("ultralytics"),
        "cuda_version":torch.version.cuda, "cudnn_version":torch.backends.cudnn.version(),
        **gpu,
    }
    mismatch = {k:{"corrected":info.get(k),"current":v} for k,v in environment.items() if info.get(k) != v}
    require(not mismatch, "Corrected GPU environment mismatch; no training: "+json.dumps(mismatch))
    require(environment["ultralytics"] == "8.4.174" and
            environment["gpu_model"] == prepared["expected_reported_gpu"], "Expected corrected GPU/version differs")
    requested = info.get("requested_training_args")
    effective = info.get("effective_training_args")
    require(isinstance(requested,dict) and isinstance(effective,dict), "Actual corrected train args missing")
    reference = {
        "requested_train_args":requested, "effective_training_args":effective,
        "pretrained_weights_sha256":info["pretrained_weights_sha256"],
        "ultralytics_version":info["ultralytics"],
    }
    training_settings(reference)  # Validates the full required no-augmentation protocol.
    portable_paths = {"model","data","device","project","name","save_dir"}
    template = prepared["effective_training_args"]
    differences = {k:[template.get(k),effective.get(k)] for k in template.keys()|effective.keys()
                   if template.get(k) != effective.get(k) and k not in portable_paths}
    require(not differences, "Corrected actual settings drift from the saved protocol: "+str(differences))
    resolved = {"optimizer","lr0","warmup_bias_lr"}
    arg_diffs = {k:[saved.get(k),effective.get(k)] for k in saved.keys()|effective.keys()
                 if saved.get(k) != effective.get(k) and k not in resolved and
                 not (k == "device" and str(saved.get(k)) == str(effective.get(k)))}
    require(not arg_diffs and saved.get("optimizer") == "auto" and requested.get("optimizer","auto") == "auto",
            "Corrected saved args and run metadata disagree: "+str(arg_diffs))
    require(effective["optimizer"] == "AdamW" and math.isclose(effective["lr0"],.00125,abs_tol=1e-12),
            "Corrected effective optimizer/lr differs")
    with (CORRECTED_RUN/"results.csv").open(encoding="utf-8-sig") as f:
        history = list(csv.DictReader(f))
    require(len(history) == info["actual_completed_epochs"] and
            all(math.isfinite(float(v)) for r in history for v in r.values()), "Corrected history corrupt/incomplete")
    reference["verified_run"] = {
        "path":str(CORRECTED_RUN), "environment":environment,
        "actual_epochs":info["actual_completed_epochs"],"best_epoch":info["best_epoch"],
        "artifact_sha256":{n:digest(CORRECTED_RUN/n) for n in needed if n != "weights/best.pt"},
        "evaluation_policy":"Val only; historical Test metrics are not consumed",
    }
    return reference

def general_run_reference(torch, gpu, prepared, corrected_reference):
    needed = ["experiment_info.json","args.yaml","results.csv","weights/best.pt",
              "train_yolov8_general_v2_colab.py"]
    require(all((GENERAL_RUN/n).is_file() for n in needed),
            "Completed E1 run missing on Drive; E6 training is blocked: "+str(GENERAL_RUN))
    info = json.loads((GENERAL_RUN/"experiment_info.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((GENERAL_RUN/"args.yaml").read_text(encoding="utf-8"))
    require(info.get("status") == "COMPLETED" and info.get("training_success") is True and
            info.get("train_val_test_image_counts") == BASE_COUNTS and info.get("model") == "yolov8n.pt" and
            info.get("pretrained_weights_sha256") == PRETRAINED_SHA256 and
            info.get("dataset_qa",{}).get("manifest_sha256") == E1_MANIFEST_SHA256,
            "Wrong/incomplete E1 model or dataset")
    require(info.get("script_sha256") == GENERAL_SCRIPT_SHA256 ==
            digest(GENERAL_RUN/"train_yolov8_general_v2_colab.py"),"Actual E1 script changed")
    current = {
        "python":platform.python_version(),"torch":str(torch.__version__),
        "torchvision":importlib.metadata.version("torchvision"),
        "ultralytics":importlib.metadata.version("ultralytics"),
        "cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),**gpu,
    }
    mismatches = {k:[info.get(k),v] for k,v in current.items() if info.get(k) != v}
    require(not mismatches,"E1 environment differs; no E6 training: "+json.dumps(mismatches))
    requested,effective = info.get("requested_training_args"),info.get("effective_training_args")
    require(isinstance(requested,dict) and isinstance(effective,dict),"E1 actual training args missing")
    reference = {"requested_train_args":requested,"effective_training_args":effective,
                 "pretrained_weights_sha256":info["pretrained_weights_sha256"],
                 "ultralytics_version":info["ultralytics"]}
    training_settings(reference)
    require(requested.get("optimizer","auto") == saved.get("optimizer") == "auto","E1 optimizer policy changed")
    ignore = {"model","data","device","project","name","save_dir"}
    old = corrected_reference["effective_training_args"]
    drift = {k:[old.get(k),effective.get(k)] for k in old.keys()|effective.keys()
             if old.get(k) != effective.get(k) and k not in ignore}
    require(not drift,"E0/E1 hyperparameters differ: "+str(drift))
    resolved = {"optimizer","lr0","warmup_bias_lr"}
    saved_diff = {k:[saved.get(k),effective.get(k)] for k in saved.keys()|effective.keys()
                  if saved.get(k) != effective.get(k) and k not in resolved and
                  not (k == "device" and str(saved.get(k)) == str(effective.get(k)))}
    require(not saved_diff,"E1 saved args and actual run metadata disagree: "+str(saved_diff))
    require(effective["optimizer"] == "AdamW" and math.isclose(effective["lr0"],.00125,abs_tol=1e-12),
            "E1 effective optimizer/lr changed")
    with (GENERAL_RUN/"results.csv").open(encoding="utf-8-sig") as f:
        history = list(csv.DictReader(f))
    require(len(history) == info["actual_completed_epochs"] and
            1 <= info["best_epoch"] <= info["actual_completed_epochs"] <= 100 and
            all(math.isfinite(float(v)) for row in history for v in row.values()),
            "E1 completed-epoch/checkpoint history mismatch")
    reference["verified_run"] = {
        "E0":corrected_reference["verified_run"],
        "E1":{"path":str(GENERAL_RUN),"environment":current,"actual_epochs":info["actual_completed_epochs"],
              "best_epoch":info["best_epoch"],
              "artifact_sha256":{n:digest(GENERAL_RUN/n) for n in needed if n != "weights/best.pt"}},
    }
    return reference

def pack_validation_metrics(metrics):
    values = [float(v) for v in metrics.box.mean_results()]
    require(len(values) == 4 and all(math.isfinite(v) for v in values),"Non-finite validation metric")
    indices = {int(c):i for i,c in enumerate(metrics.box.ap_class_index)}
    per_class = {}
    for cid,name in NAMES.items():
        require(cid in indices,"Missing validation class metric: "+name)
        p,r,ap50,ap = [float(v) for v in metrics.box.class_result(indices[cid])]
        require(all(math.isfinite(v) for v in [p,r,ap50,ap]),"Non-finite validation class metric")
        per_class[name] = {"instances":CLASS_COUNTS["val"][name],
                           "precision":p,"recall":r,"ap50":ap50,"ap50_95":ap}
    seen = getattr(metrics,"nt_per_class",None)
    if seen is not None:
        require(list(map(int,seen)) == [CLASS_COUNTS["val"][n] for n in NAMES.values()],
                "Validation GT count changed")
    return {"split":"val","images":67,"overall":dict(zip(["precision","recall","map50","map50_95"],values)),
            "per_class":per_class,"test_evaluated":False,"selection_split":"val",
            "corrected_manifest_sha256":CORRECTED_MANIFEST_SHA256}


def evaluate_best_validation(info, YOLO):
    require(info.get("test_evaluation_calls",0) == 0 and not info.get("test_evaluated",False),
            "E6 forbids Test performance evaluation")
    require(info.get("validation_evaluation_calls",0) == 0,"Standalone best validation already executed")
    require(verify_dataset(DATA) == info["dataset_qa"],"Dataset changed before validation")
    best = YOLO(str(RUN/"weights/best.pt"),task="detect")
    require(getattr(best.model,"class_weights",None) is None,"E6 checkpoint has unexpected class weights")
    evaluated_settings = {}
    def on_val_start(validator):
        require(validator.args.split == "val","Only Val67 is permitted for E6 metrics")
        info.setdefault("loader_qa",{})["val_best"] = loader_check(validator.dataloader.dataset,"val")
        evaluated_settings.update(vars(validator.args))
    best.add_callback("on_val_start",on_val_start)
    settings = dict(data=str(DATA/"data_colab.yaml"),split="val",imgsz=640,batch=8,device=0,
                    workers=0,plots=True,augment=False,project=str(RUN),name="val_best",exist_ok=False,verbose=True)
    info["validation_evaluation_calls"] = 1
    save_info(info)
    result = pack_validation_metrics(best.val(**settings))
    result["evaluation_settings"] = evaluated_settings
    (RUN/"validation_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    info.update(validation=result,validation_evaluated=True,test_evaluated=False,
                selection_split="val",test_evaluation_calls=0)
    lines = ["# E6 best checkpoint / Validation67 only","",
             "Test66 is preserved for dataset integrity only and is not evaluated.","",
             "| Metric | Validation |","|---|---:|"]
    lines += [f"| {k} | {v:.8f} |" for k,v in result["overall"].items()]
    lines += ["","| Class | GT | Precision | Recall | AP50 | AP50-95 |","|---|---:|---:|---:|---:|---:|"]
    for name,m in result["per_class"].items():
        lines.append("| "+name+" | "+str(m["instances"])+" | "+
                     " | ".join(f"{m[k]:.6f}" for k in ["precision","recall","ap50","ap50_95"])+" |")
    (RUN/"validation_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_info(info)


if __name__ == "__main__":
    main()
