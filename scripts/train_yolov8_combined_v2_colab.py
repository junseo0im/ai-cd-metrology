"""Colab CUDA-only combined_v2 YOLOv8n training. Importing this file never trains.

Mount Drive, install the dataset's requirements_colab.txt, then execute this saved
script. --check-only performs GPU/environment/dataset checks without training.
No CPU fallback, dataset edits or new augmentation. Best checkpoint is evaluated only on Val67; Test performance evaluation is forbidden.
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
DATA = DRIVE / "datasets/yolov8_detection_combined_v2"
WEIGHTS = DRIVE / "models/yolov8n.pt"
RUN = DRIVE / "runs/v2_combined/yolov8n_combined_v2"
COUNTS = {"train": 604, "val": 67, "test": 66}
NAMES = {0: "o", 1: "p", 2: "s", 3: "sc"}
CLASS_COUNTS = {
    "train": {"o": 105, "p": 5054, "s": 120, "sc": 511},
    "val": {"o": 4, "p": 593, "s": 4, "sc": 54},
    "test": {"o": 6, "p": 580, "s": 5, "sc": 53},
}
SOURCE_SHA256 = "f6f884d2079e9154e164da7a4548f08dcca23f86fab5cfe8268a8b08808fde9f"
MANIFEST_SHA256 = "3ea94d90462ddce066c6ed4b01d5ed956ad9771fe9eb69f3ee5dd64c449c3108"
REFERENCE_SHA256 = "45efe817857f129a39e7532a69899dd73c5aa6963c9a4146dddbee089f450c67"
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
    policy = qa["evaluation_policy"]
    require(policy["selection_split"] == "val" and policy["test_evaluation_allowed"] is False and
            policy["test_used_for_selection"] is False,"E3 must remain validation-only")
    with (folder/"image_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 737 and len({r["image_id"] for r in rows}) == 737, "Manifest count/ID mismatch")
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
    require(all(len(v) == 1 for v in groups.values()) and len(hashes) == 737 and
            all(len(v) == 1 for v in hashes.values()), "Duplicate/group split leakage")
    require(zeros == {"train":5, "val":0, "test":1}, "Zero-label membership changed")
    verify_comparison(folder,rows,qa)
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
        print("COMBINED_V2_PROGRESS="+json.dumps({
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
    fit_seconds = time.perf_counter()-started
    evaluate_best_validation(info,YOLO)
    info.update(status="COMPLETED", training_success=True, error=None,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                training_seconds=fit_seconds,
                best_pt_path=str(RUN/"weights/best.pt"), last_pt_path=str(RUN/"weights/last.pt"),
                test_evaluated=False, validation_evaluated=True)
    save_info(info)
    (RUN/"experiment_summary.md").write_text("\n".join([
        "# combined_v2 YOLOv8n / Colab GPU", "",
        "- Status: COMPLETED; Train604 / Val67 / Test66, fixed combined_v2 split.",
        "- Requested epochs100; completed "+str(info["actual_completed_epochs"])+"; best "+str(info["best_epoch"])+".",
        "- imgsz640 / batch8 / seed42 / patience20 / device0 / workers0 / deterministic=True.",
        "- Baseline optimizer=auto policy retained; actual AdamW, lr0.00125, betas(0.9,0.999) verified.",
        "- AMP=False; all baseline online augmentation disabled. No new augmentation/oversampling.",
        "- Dataset hash QA and loader image/annotation counts passed.",
        "- Val67 selects the checkpoint and is the only official comparison set. See validation_results.md.",
        "- Test66 is preserved for integrity checks only; no Test performance evaluation is run.",
        "- GPU, Python/Torch/CUDA and Ultralytics must match completed E0/E1/E2 runs.",
        "- Full environment, optimizer groups, hashes and actual args: experiment_info.json.",
        "- Best: "+info["best_pt_path"], "",
    ]), encoding="utf-8")
    print("COMBINED_V2_COMPLETED="+json.dumps({
        k:info[k] for k in ["training_success","actual_completed_epochs","best_epoch","best_pt_path"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="GPU/environment/data QA only; no training or run output")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    gpu = gpu_preflight(torch)  # Must fail before importing/constructing a model when CUDA is unavailable.
    prepared = json.loads((DATA/"baseline_reference.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("ultralytics")
    require(version == prepared["ultralytics_version"], "Install the pinned requirements_colab.txt; version drift is forbidden")
    require(importlib.util.find_spec("albumentations") is None,
            "Albumentations is installed. Remove it before this baseline (pip uninstall -y albumentations).")
    qa = verify_dataset(DATA)
    corrected_reference = corrected_run_reference(torch,gpu,prepared)
    general_reference = general_run_reference(torch,gpu,prepared,corrected_reference)
    reference = geometric_run_reference(torch,gpu,prepared,general_reference)
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
        "test_evaluated":False, "test_evaluation_calls":0, "validation_evaluation_calls":0,
        "selection_split":"val", "validation_only":True,
        "comparison_baselines":reference["verified_run"],
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

CORRECTED_RUN = DRIVE/"runs/v2_baseline/yolov8n_corrected_v2"
CORRECTED_MANIFEST_SHA256 = "f6f884d2079e9154e164da7a4548f08dcca23f86fab5cfe8268a8b08808fde9f"


def verify_comparison(folder, rows, qa):
    require(digest(folder/"corrected_image_manifest.csv") == qa["corrected_manifest_sha256"] ==
            CORRECTED_MANIFEST_SHA256, "Corrected reference manifest changed")
    require(digest(folder/"augmentation_manifest.json") == qa["augmentation_manifest_sha256"],
            "Existing augmentation manifest changed")
    with (folder/"corrected_image_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        originals = list(csv.DictReader(f))
    current = {r["image_id"]:r for r in rows}
    original_by_id = {r["image_id"]:r for r in originals}
    require(len(original_by_id) == len(originals) == 667, "Corrected reference count mismatch")
    same_fields = ["filename","split","group_key","width","height","image_sha256","label_sha256",
                   "annotation_count","o_count","p_count","s_count","sc_count"]
    for row in originals:
        require(row["image_id"] in current and
                all(current[row["image_id"]][k] == row[k] for k in same_fields),
                "Corrected original/image/label/split changed")
    augmented = [r for r in rows if r["image_id"] not in original_by_id]
    aug = json.loads((folder/"augmentation_manifest.json").read_text(encoding="utf-8-sig"))
    by_id = {str(r["image_id"]):r for r in aug["records"]}
    require(len(augmented) == len(by_id) == 70 and
            {r["image_id"] for r in augmented} == set(by_id), "Augmentation set changed")
    parents = set()
    for row in augmented:
        entry = by_id[row["image_id"]]
        parent = original_by_id[str(entry["parent_image_id"])]
        require(row["split"] == parent["split"] == "train" and row["is_augmentation"] == "True" and
                row["parent_image_id"] == parent["image_id"] and row["filename"] == entry["file_name"] and
                row["group_key"] == parent["group_key"] and int(parent["o_count"])+int(parent["s_count"]) > 0,
                "Augmentation parent/split provenance mismatch")
        parents.add(parent["image_id"])
    require(len(parents) == 35, "Existing augmentation parent count changed")
    verify_combined_provenance(folder,rows,qa,parents)


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
        "baseline_test_metrics":"Use saved Test metrics only; never rerun corrected inference/training",
    }
    return reference


GENERAL_RUN = DRIVE/"runs/v2_augmented/yolov8n_general_v2"
E1_PARENT_SHA256 = "2f537e1db78385cb0671a3965f97a36e27982a524adad0e520dd34f035b9e4c7"
COMBINED_CSV_SHA256 = "0ac8ea9fb85e17350067f5d65fe9be559b3066fc748594665e80f8620ba09f2e"
REJECTED_SHA256 = "a33937777807f012f29e81e08b9d26bfc338d0ec2f33c5710a0632ece64ea8aa"


def verify_combined_provenance(folder, rows, qa, parents):
    require(digest(folder/"E1_parent_manifest.json") == E1_PARENT_SHA256 == qa["E1_parent_manifest_sha256"],
            "E1 parent source changed")
    e1 = json.loads((folder/"E1_parent_manifest.json").read_text(encoding="utf-8-sig"))
    require(parents == {str(r["parent_image_id"]) for r in e1["records"]},"E3 parent set differs from E1")
    require(digest(folder/"E2_parent_manifest.json") == E2_PARENT_SHA256 == qa["E2_parent_manifest_sha256"],
            "E2 parent reference changed")
    e2 = json.loads((folder/"E2_parent_manifest.json").read_text(encoding="utf-8"))
    require(parents == {str(r["parent_image_id"]) for r in e2["records"]},"E3 parent set differs from E2")
    require(digest(folder/"augmentation_manifest.csv") == COMBINED_CSV_SHA256 == qa["augmentation_csv_sha256"] and
            digest(folder/"rejected_transforms.csv") == REJECTED_SHA256 == qa["rejected_transforms_sha256"],
            "Accepted/rejected transform records changed")
    with (folder/"augmentation_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        accepted = list(csv.DictReader(f))
    with (folder/"rejected_transforms.csv").open(encoding="utf-8-sig",newline="") as f:
        rejected = list(csv.DictReader(f))
    all_images = {r["filename"]:r for r in rows}
    by_variant = {(r["parent_image_id"],r["variant"]):r for r in accepted}
    require(len(accepted) == len(by_variant) == 70 and
            set(by_variant) == {(i,v) for i in parents for v in ["A","B"]},"Variant budget changed")
    def validate_parameters(row):
        parent = all_images[row["parent_filename"]]
        angle,tx,ty = (float(row[k]) for k in ["angle_deg","translate_x_px","translate_y_px"])
        require(int(row["seed"]) == 42 and -5 <= angle <= 5 and angle != 0 and
                all(math.isfinite(v) for v in [angle,tx,ty]) and
                abs(tx) <= .03*int(parent["width"]) and abs(ty) <= .03*int(parent["height"]),
                "Geometric parameter range/seed changed")
        require(row["variant"] in {"A","B"} and (row["variant"] != "A" or tx == ty == 0),
                "Variant A may rotate only")
        brightness,contrast = float(row["brightness_factor"]),float(row["contrast_factor"])
        require(.93 <= brightness <= 1.07 and .95 <= contrast <= 1.05 and
                row["flip_type"] == ("horizontal" if row["variant"] == "A" else "vertical"),
                "Combined flip/photometric parameters differ")
        if "augmented_filename" in row:
            require(row["transform_order"] == "flip -> affine -> brightness -> contrast" and
                    row["o_present"] == str(int(parent["o_count"]) > 0) and
                    row["s_present"] == str(int(parent["s_count"]) > 0) and
                    int(row["annotation_count"]) == int(parent["annotation_count"]),
                    "Combined transform order or source annotation trace changed")
        return parent
    for row in rejected:
        validate_parameters(row)
        require(row["failure_reason"],"Missing rejected-transform reason")
    for row in accepted:
        parent = validate_parameters(row)
        image = all_images[row["augmented_filename"]]
        retries = int(row["retry_count"])
        attempts = [int(r["attempt"]) for r in rejected if
                    (r["parent_image_id"],r["variant"]) == (row["parent_image_id"],row["variant"])]
        require(0 <= retries < 100 and retries == int(row["rejected_transform_count"]) and
                attempts == list(range(1,retries+1)),"Retry record/count mismatch")
        require(int(image["annotation_count"]) == int(parent["annotation_count"]) and
                all(image[c+"_count"] == parent[c+"_count"] for c in NAMES.values()),
                "A source annotation/class was removed or changed")
    require(len(rejected) == qa["rejected_transform_attempts"] == sum(int(r["retry_count"]) for r in accepted) and
            max(int(r["retry_count"]) for r in accepted) == qa["max_retry_count_used"] and
            qa["max_transform_attempts"] == 100,"Rejection-sampling trace changed")


def general_run_reference(torch, gpu, prepared, corrected_reference):
    needed = ["experiment_info.json","args.yaml","results.csv","weights/best.pt",
              "train_yolov8_general_v2_colab.py"]
    require(all((GENERAL_RUN/n).is_file() for n in needed),
            "Completed E1 run missing on Drive; E2 training is blocked: "+str(GENERAL_RUN))
    info = json.loads((GENERAL_RUN/"experiment_info.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((GENERAL_RUN/"args.yaml").read_text(encoding="utf-8"))
    require(info.get("status") == "COMPLETED" and info.get("training_success") is True and
            info.get("train_val_test_image_counts") == COUNTS and info.get("model") == "yolov8n.pt" and
            info.get("pretrained_weights_sha256") == PRETRAINED_SHA256 and
            info.get("dataset_qa",{}).get("manifest_sha256") == prepared["general_manifest_sha256"],
            "Wrong/incomplete E1 model or dataset")
    require(info.get("script_sha256") == prepared["general_script_sha256"] ==
            digest(GENERAL_RUN/"train_yolov8_general_v2_colab.py"),"Actual E1 script changed")
    current = {
        "python":platform.python_version(),"torch":str(torch.__version__),
        "torchvision":importlib.metadata.version("torchvision"),
        "ultralytics":importlib.metadata.version("ultralytics"),
        "cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),**gpu,
    }
    mismatches = {k:[info.get(k),v] for k,v in current.items() if info.get(k) != v}
    require(not mismatches,"E1 environment differs; no E2 training: "+json.dumps(mismatches))
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


GEOMETRIC_RUN = DRIVE/"runs/v2_geometric/yolov8n_geometric_v2"
E2_PARENT_SHA256 = "4650f2fd909672b2ef3252c786610160e4e8b9ff3f000d565a837cb9420306f0"

def geometric_run_reference(torch, gpu, prepared, previous_reference):
    needed = ["experiment_info.json","args.yaml","results.csv","weights/best.pt",
              "train_yolov8_geometric_v2_colab.py"]
    require(all((GEOMETRIC_RUN/n).is_file() for n in needed),
            "Completed E2 run missing on Drive; E2 training is blocked: "+str(GEOMETRIC_RUN))
    info = json.loads((GEOMETRIC_RUN/"experiment_info.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((GEOMETRIC_RUN/"args.yaml").read_text(encoding="utf-8"))
    require(info.get("status") == "COMPLETED" and info.get("training_success") is True and
            info.get("train_val_test_image_counts") == COUNTS and info.get("model") == "yolov8n.pt" and
            info.get("pretrained_weights_sha256") == PRETRAINED_SHA256 and
            info.get("dataset_qa",{}).get("manifest_sha256") == prepared["geometric_manifest_sha256"],
            "Wrong/incomplete E2 model or dataset")
    require(info.get("script_sha256") == prepared["geometric_script_sha256"] ==
            digest(GEOMETRIC_RUN/"train_yolov8_geometric_v2_colab.py"),"Actual E2 script changed")
    current = {
        "python":platform.python_version(),"torch":str(torch.__version__),
        "torchvision":importlib.metadata.version("torchvision"),
        "ultralytics":importlib.metadata.version("ultralytics"),
        "cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),**gpu,
    }
    mismatches = {k:[info.get(k),v] for k,v in current.items() if info.get(k) != v}
    require(not mismatches,"E2 environment differs; no E2 training: "+json.dumps(mismatches))
    requested,effective = info.get("requested_training_args"),info.get("effective_training_args")
    require(isinstance(requested,dict) and isinstance(effective,dict),"E2 actual training args missing")
    reference = {"requested_train_args":requested,"effective_training_args":effective,
                 "pretrained_weights_sha256":info["pretrained_weights_sha256"],
                 "ultralytics_version":info["ultralytics"]}
    training_settings(reference)
    require(requested.get("optimizer","auto") == saved.get("optimizer") == "auto","E2 optimizer policy changed")
    ignore = {"model","data","device","project","name","save_dir"}
    old = previous_reference["effective_training_args"]
    drift = {k:[old.get(k),effective.get(k)] for k in old.keys()|effective.keys()
             if old.get(k) != effective.get(k) and k not in ignore}
    require(not drift,"Previous/E2 hyperparameters differ: "+str(drift))
    resolved = {"optimizer","lr0","warmup_bias_lr"}
    saved_diff = {k:[saved.get(k),effective.get(k)] for k in saved.keys()|effective.keys()
                  if saved.get(k) != effective.get(k) and k not in resolved and
                  not (k == "device" and str(saved.get(k)) == str(effective.get(k)))}
    require(not saved_diff,"E2 saved args and actual run metadata disagree: "+str(saved_diff))
    require(effective["optimizer"] == "AdamW" and math.isclose(effective["lr0"],.00125,abs_tol=1e-12),
            "E2 effective optimizer/lr changed")
    with (GEOMETRIC_RUN/"results.csv").open(encoding="utf-8-sig") as f:
        history = list(csv.DictReader(f))
    require(len(history) == info["actual_completed_epochs"] and
            1 <= info["best_epoch"] <= info["actual_completed_epochs"] <= 100 and
            all(math.isfinite(float(v)) for row in history for v in row.values()),
            "E2 completed-epoch/checkpoint history mismatch")
    reference["verified_run"] = {
        **previous_reference["verified_run"],
        "E2":{"path":str(GEOMETRIC_RUN),"environment":current,"actual_epochs":info["actual_completed_epochs"],
              "best_epoch":info["best_epoch"],
              "artifact_sha256":{n:digest(GEOMETRIC_RUN/n) for n in needed if n != "weights/best.pt"}},
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
            "E3 forbids Test performance evaluation")
    require(info.get("validation_evaluation_calls",0) == 0,"Standalone best validation already executed")
    require(verify_dataset(DATA) == info["dataset_qa"],"Dataset changed before validation")
    best = YOLO(str(RUN/"weights/best.pt"),task="detect")
    evaluated_settings = {}
    def on_val_start(validator):
        require(validator.args.split == "val","Only Val67 is permitted for E3 metrics")
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
    lines = ["# E3 best checkpoint / Validation67 only","",
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





