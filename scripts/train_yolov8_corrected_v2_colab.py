"""Colab CUDA-only corrected_v2 YOLOv8n training. Importing this file never trains.

Mount Drive, install the dataset's requirements_colab.txt, then execute this saved
script. --check-only performs GPU/environment/dataset checks without training.
No CPU fallback, dataset edits, additional augmentation, or standalone Test run.
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
DATA = DRIVE / "datasets/yolov8_detection_corrected_v2"
WEIGHTS = DRIVE / "models/yolov8n.pt"
RUN = DRIVE / "runs/v2_baseline/yolov8n_corrected_v2"
COUNTS = {"train": 534, "val": 67, "test": 66}
NAMES = {0: "o", 1: "p", 2: "s", 3: "sc"}
CLASS_COUNTS = {
    "train": {"o": 35, "p": 4534, "s": 40, "sc": 427},
    "val": {"o": 4, "p": 593, "s": 4, "sc": 54},
    "test": {"o": 6, "p": 580, "s": 5, "sc": 53},
}
SOURCE_SHA256 = "890c9df87d04c952066fd4e9587a4ed93bdd8ccd591d7100d733ac7ff6ba0314"
MANIFEST_SHA256 = "f6f884d2079e9154e164da7a4548f08dcca23f86fab5cfe8268a8b08808fde9f"
REFERENCE_SHA256 = "97a24f4adac56ca07cd51be26570bb3a92a9ca2dc44f575d6aea1d7fd7edb3ab"
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
    qa = json.loads((folder/"conversion_summary.json").read_text(encoding="utf-8"))
    require(qa["conversion_success"] and qa["dataset_qa"] == "PASS" and qa["source_sha256"] == SOURCE_SHA256,
            "Source conversion QA/provenance mismatch")
    require(digest(folder/"image_manifest.csv") == qa["image_manifest_sha256"] == MANIFEST_SHA256,
            "Dataset manifest changed")
    require(digest(folder/"baseline_reference.json") == qa["baseline_reference_sha256"] == REFERENCE_SHA256,
            "Baseline reference changed")
    cfg = yaml.safe_load((folder/"data_colab.yaml").read_text(encoding="utf-8"))
    require(cfg["path"] == DATA.as_posix() and cfg["nc"] == 4 and cfg["names"] == NAMES,
            "Colab data.yaml path/class mapping mismatch")
    require(all(cfg[s] == s+"/images" for s in COUNTS), "Colab split path mismatch")
    with (folder/"image_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 667 and len({r["image_id"] for r in rows}) == 667, "Manifest count/ID mismatch")
    require(dict(Counter(r["split"] for r in rows)) == COUNTS, "Split image counts changed")
    groups, hashes, counts, zeros = defaultdict(set), defaultdict(set), {}, {}
    for split in COUNTS:
        selected = [r for r in rows if r["split"] == split]
        expected_images = {r["filename"] for r in selected}
        require({p.name for p in (folder/split/"images").iterdir()} == expected_images, "Missing/extra image")
        require({p.name for p in (folder/split/"labels").iterdir()} ==
                {Path(n).stem+".txt" for n in expected_images}, "Missing/extra label")
        total = Counter()
        zero = 0
        for row in selected:
            image = folder/split/"images"/row["filename"]
            label = folder/split/"labels"/(image.stem+".txt")
            require(digest(image) == row["image_sha256"] and digest(label) == row["label_sha256"],
                    "Image/label bytes differ from QA")
            with Image.open(image) as im:
                require(im.size == (int(row["width"]), int(row["height"])), "Image dimensions changed")
            lines = label.read_text(encoding="ascii").splitlines()
            require(len(lines) == int(row["annotation_count"]), "Annotation count changed")
            per_image = Counter()
            for line in lines:
                tokens = line.split()
                require(len(tokens) == 5, "Only five-token Detection bbox rows are permitted")
                cid = int(tokens[0])
                x,y,w,h = map(float, tokens[1:])
                require(cid in NAMES and str(cid) == tokens[0] and
                        all(math.isfinite(v) for v in (x,y,w,h)) and
                        0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                        x-w/2 >= -1e-12 and y-h/2 >= -1e-12 and x+w/2 <= 1+1e-12 and y+h/2 <= 1+1e-12,
                        "Invalid Detection coordinates; do not clip or fix")
                per_image[NAMES[cid]] += 1
            require(all(per_image[c] == int(row[c+"_count"]) for c in NAMES.values()),
                    "Per-image class count changed")
            total.update(per_image)
            zero += not lines
            groups[row["group_key"]].add(split)
            hashes[row["image_sha256"]].add(split)
        counts[split] = {c:total[c] for c in NAMES.values()}
        require(counts[split] == CLASS_COUNTS[split], "Split class counts changed")
        zeros[split] = zero
    require(all(len(v) == 1 for v in groups.values()) and len(hashes) == 667 and
            all(len(v) == 1 for v in hashes.values()), "Duplicate/group split leakage")
    require(zeros == {"train":5, "val":0, "test":1}, "Zero-label membership changed")
    return {"status":"PASS", "image_counts":COUNTS, "class_counts":counts,
            "zero_labels":zeros, "declared_group_leakage":0, "exact_duplicates":0,
            "manifest_sha256":MANIFEST_SHA256, "reference_sha256":REFERENCE_SHA256,
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

    def on_epoch_end(trainer):
        if int(trainer.epoch)+1 != info.get("current_epoch"):
            return  # Ignore the extra callback used by Ultralytics final best validation.
        metrics = {k:float(v) for k,v in trainer.metrics.items()}
        require(all(math.isfinite(v) for v in metrics.values()), "Non-finite training/validation metric")
        info.update(actual_completed_epochs=int(trainer.epoch)+1, latest_validation_metrics=metrics)
        save_info(info)
        print("CORRECTED_V2_PROGRESS="+json.dumps({
            "epoch":info["actual_completed_epochs"], "best_epoch":info["best_epoch"]}), flush=True)

    for event, callback in [
        ("on_train_start",on_start), ("on_train_epoch_start",on_epoch_start),
        ("on_train_batch_start",on_batch_start), ("on_model_save",on_save),
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
    info.update(status="COMPLETED", training_success=True, error=None,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                training_seconds=time.perf_counter()-started,
                best_pt_path=str(RUN/"weights/best.pt"), last_pt_path=str(RUN/"weights/last.pt"),
                test_evaluated=False)
    save_info(info)
    (RUN/"experiment_summary.md").write_text("\n".join([
        "# corrected_v2 YOLOv8n / Colab GPU", "",
        "- Status: COMPLETED; Train534 / Val67 / Test66, fixed corrected_v2 split.",
        "- Requested epochs100; completed "+str(info["actual_completed_epochs"])+"; best "+str(info["best_epoch"])+".",
        "- imgsz640 / batch8 / seed42 / patience20 / device0 / workers0 / deterministic=True.",
        "- Baseline optimizer=auto policy retained; actual AdamW, lr0.00125, betas(0.9,0.999) verified.",
        "- AMP=False; all baseline online augmentation disabled. No new augmentation/oversampling.",
        "- Dataset hash QA and loader image/annotation counts passed.",
        "- Val is used for checkpoint selection; no standalone Test evaluation was run.",
        "- CPU/CUDA, fused optimizer kernels and runtime versions may cause numerical differences.",
        "- Full environment, optimizer groups, hashes and actual args: experiment_info.json.",
        "- Best: "+info["best_pt_path"], "",
    ]), encoding="utf-8")
    print("CORRECTED_V2_COMPLETED="+json.dumps({
        k:info[k] for k in ["training_success","actual_completed_epochs","best_epoch","best_pt_path"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="GPU/environment/data QA only; no training or run output")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    gpu = gpu_preflight(torch)  # Must fail before importing/constructing a model when CUDA is unavailable.
    reference = json.loads((DATA/"baseline_reference.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("ultralytics")
    require(version == reference["ultralytics_version"], "Install the pinned requirements_colab.txt; version drift is forbidden")
    require(importlib.util.find_spec("albumentations") is None,
            "Albumentations is installed. Remove it before this baseline (pip uninstall -y albumentations).")
    qa = verify_dataset(DATA)
    require(digest(WEIGHTS) == PRETRAINED_SHA256 == reference["pretrained_weights_sha256"],
            "Copy the unchanged official baseline yolov8n.pt to the expected Drive models path")
    settings = training_settings(reference)
    print("DATASET_QA="+json.dumps(qa), flush=True)
    print("TRAINING_ARGS="+json.dumps(settings), flush=True)
    if args.check_only:
        print("READY: CUDA/environment/dataset QA PASS. No training executed.", flush=True)
        return
    require(not RUN.exists(), "Run already exists; never overwrite or silently resume an experiment")
    RUN.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    shutil.copy2(script, RUN/script.name)
    runtime = RUN/"runtime_config"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    info = {
        "status":"PREPARING", "training_success":False, "error":None,
        "started_utc":datetime.now(timezone.utc).isoformat(), "python":platform.python_version(),
        "platform":platform.platform(), "torch":torch.__version__, "ultralytics":version,
        "torchvision":importlib.metadata.version("torchvision"), "cuda_version":torch.version.cuda,
        "cudnn_version":torch.backends.cudnn.version(), **gpu, "device":0,
        "dataset_path":str(DATA), "data_yaml":str(DATA/"data_colab.yaml"), "dataset_qa":qa,
        "train_val_test_image_counts":COUNTS, "model":"yolov8n.pt",
        "pretrained_weights_path":str(WEIGHTS), "pretrained_weights_sha256":PRETRAINED_SHA256,
        "epochs_requested":100, "actual_completed_epochs":0, "best_epoch":None,
        "imgsz":640, "batch":8, "seed":42, "script_path":str(script), "script_sha256":digest(script),
        "test_evaluated":False,
        "environment_note":"GPU fused kernels/channels-last and PyTorch/CUDA versions are logged; no cross-hardware bitwise guarantee.",
    }
    save_info(info)
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in [
            "sync","wandb","mlflow","comet","clearml","dvc","neptune","raytune","tensorboard","hub"] if k in SETTINGS})
        run_training(info,reference,torch,YOLO)
    except BaseException as error:
        info.update(status="FAILED", training_success=False, error=f"{type(error).__name__}: {error}",
                    ended_utc=datetime.now(timezone.utc).isoformat())
        save_info(info)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
