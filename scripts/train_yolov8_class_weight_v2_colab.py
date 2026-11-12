"""E5-A/B fixed class-wise BCE weights on the read-only E1 general_v2 dataset.

Preparation/check modes never fit a model or perform inference. Training is
CUDA-only and explicitly selected with --variant E5-A or --variant E5-B.
The native Ultralytics 8.4.174 v8DetectionLoss already multiplies unreduced
classification BCE by model.class_weights once. Only DetectionTrainer's
set_class_weights hook is replaced, with no mean normalization or cls_pw
frequency weighting. Both positive and negative BCE terms are weighted.
Box/DFL, assignment, cls gain, model architecture and E1 sampler stay native.
Only Val67 is evaluated. Test66 is opaque hash/count/identity/leakage QA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
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
DATA = DRIVE / "datasets/yolov8_detection_general_v2"
WEIGHTS = DRIVE / "models/yolov8n.pt"
RUN = None  # Selected explicitly by --variant; never defaults to another experiment.
COUNTS = {"train": 604, "val": 67, "test": 66}
NAMES = {0: "o", 1: "p", 2: "s", 3: "sc"}
CLASS_COUNTS = {
    "train": {"o": 105, "p": 5054, "s": 120, "sc": 511},
    "val": {"o": 4, "p": 593, "s": 4, "sc": 54},
    "test": {"o": 6, "p": 580, "s": 5, "sc": 53},
}
SOURCE_SHA256 = "9c9d5b31af2923b92a42a8f3e39caf5f6dbc90f00a2ca0c1b005ebd5e270a28d"
MANIFEST_SHA256 = "b0ce5cd816ef2dd3a0a8d8652785844d009a66dcf06505cb03485e56ad9f1b11"
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


EXPERIMENTS = {
    "E5-A":{"weights":[1.5,1.0,1.5,1.0],"run_name":"e5a_weight_1p5"},
    "E5-B":{"weights":[2.0,1.0,2.0,1.0],"run_name":"e5b_weight_2p0"},
}
LOSS_SOURCE_SHA256 = "91146bb2ebb791f350b2c3cc0cdafc361ddbdb615276d65738dad7d7b1a6baff"


def verify_dataset(folder):
    qa = json.loads((folder/"conversion_summary.json").read_text(encoding="utf-8"))
    require(qa["conversion_success"] and qa["dataset_qa"] == "PASS" and qa["source_sha256"] == SOURCE_SHA256,
            "Source conversion QA/provenance mismatch")
    require(digest(folder/"image_manifest.csv") == qa["image_manifest_sha256"] == MANIFEST_SHA256,
            "Dataset manifest changed")
    require(digest(folder/"baseline_reference.json") == qa["baseline_reference_sha256"] == REFERENCE_SHA256,
            "Baseline reference changed")
    cfg = yaml.safe_load((folder/"data_colab.yaml").read_text(encoding="utf-8"))
    require(cfg["path"] == (DRIVE / "datasets/yolov8_detection_general_v2").as_posix() and cfg["nc"] == 4 and cfg["names"] == NAMES,
            "Colab data.yaml path/class mapping mismatch")
    require(all(cfg[s] == s+"/images" for s in COUNTS), "Colab split path mismatch")
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
                im.verify()
            if split != "test" and image.suffix.lower() in {".jpg",".jpeg"}:
                with image.open("rb") as f:
                    f.seek(-2,2)
                    require(f.read() == bytes([255,217]), "Refuse JPEG that could trigger Ultralytics auto-repair")
            if split == "test":
                # Test content stays opaque: fixed hashes prove identity; no Test loader/inference.
                for name in NAMES.values():
                    total[name] += int(row[name+"_count"])
                zero += row["zero_annotation"] == "True"
                groups[row["group_key"]].add(split)
                hashes[row["image_sha256"]].add(split)
                continue
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
    require(settings["optimizer"] == "auto" and settings.get("split", "val") == "val" and
            not settings.get("resume", False), "E5 requires fresh auto-optimizer training and Val only")
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

def native_loss_structure():
    """Verify the exact locally audited loss, not an assumed upstream API."""
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.models.yolo.detect import DetectionTrainer
    require(importlib.metadata.version("ultralytics") == "8.4.174","E5 pins Ultralytics 8.4.174")
    source = inspect.getsource(v8DetectionLoss).replace("\r\n","\n")
    sha = hashlib.sha256(source.encode()).hexdigest()
    require(sha == LOSS_SOURCE_SHA256,"Detection loss source differs from the audited 8.4.174 implementation")
    # The one native multiply is deliberately retained, never wrapped by another weighted BCE.
    require(source.count("bce_loss *= self.class_weights") == 1,
            "Expected exactly one native classification class-weight application")
    return {
        "ultralytics":"8.4.174","criterion":"ultralytics.utils.loss.v8DetectionLoss",
        "source_path":inspect.getsourcefile(v8DetectionLoss),
        "source_start_line":inspect.getsourcelines(v8DetectionLoss)[1],
        "source_sha256":sha,
        "application":"unreduced BCE[batch,anchor,class] * class_weights[1,1,4], once, before sum",
        "normalization":"original unweighted target_scores_sum; existing hyp.cls gain unchanged",
        "scope":"Both positive and negative BCE terms for each class, not pos_weight",
        "box_and_dfl":"Original bbox_loss inputs and hyp.box/hyp.dfl gains unchanged",
        "trainer_override":"set_class_weights only; native DataLoader, sampler, model and loss methods",
        "default_frequency_weighting":"Not used: cls_pw stays 0; no weight mean normalization",
        "native_trainer_weight_source_sha256":hashlib.sha256(
            inspect.getsource(DetectionTrainer.set_class_weights).replace("\r\n","\n").encode()).hexdigest(),
    }


def make_class_weight_trainer(variant):
    """Assign the exact fixed vector; the native v8DetectionLoss applies it once.

    This is an assignment, not multiplication of an existing model weight. No
    custom BCE, loss copy, architecture, sampler or inverse-frequency weighting.
    """
    require(variant in EXPERIMENTS,"Only E5-A and E5-B are allowed")
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    weights = tuple(EXPERIMENTS[variant]["weights"])

    class FixedClassWeightTrainer(DetectionTrainer):
        def set_class_weights(self):
            require(self.args.cls_pw == 0.0,"Automatic frequency weighting must remain OFF")
            require(self.data["nc"] == 4 and self.data["names"] == NAMES,"Unexpected class order")
            model = unwrap_model(self.model)
            require(not hasattr(model,"student_model"),"Distillation is outside E5")
            require(getattr(model,"criterion",None) is None,"Fixed weights must be assigned before lazy loss initialization")
            model.class_weights = torch.tensor(weights,dtype=torch.float32,
                                               device=next(model.parameters()).device)
            require(tuple(model.class_weights.tolist()) == weights,"Fixed vector changed")
    return FixedClassWeightTrainer


def actual_weight_state(model,variant,require_criterion=False):
    """Observe actual model/native-criterion weights; never modifies a loss tensor."""
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import unwrap_model
    model = unwrap_model(model)
    expected = torch.tensor(EXPERIMENTS[variant]["weights"],device=next(model.parameters()).device)
    require(torch.equal(model.class_weights,expected),"Actual model class weights differ")
    criterion = getattr(model,"criterion",None)
    result = {"requested":EXPERIMENTS[variant]["weights"],
              "effective_model":model.class_weights.detach().cpu().tolist(),
              "criterion_initialized":criterion is not None,
              "weight_application_count_in_native_loss":1}
    if require_criterion or criterion is not None:
        require(type(criterion) is v8DetectionLoss,"Only native YOLOv8 detection loss is permitted")
        require(torch.equal(criterion.class_weights.reshape(-1),expected),"Actual loss vector differs")
        require(type(criterion.bce) is torch.nn.BCEWithLogitsLoss and
                criterion.bce.reduction == "none" and criterion.bce.weight is None and
                criterion.bce.pos_weight is None,"BCE may not add a second class-weight operation")
        result["effective_criterion"] = criterion.class_weights.detach().cpu().reshape(-1).tolist()
    return result


def loss_sanity_check():
    """CPU tensor-only verification of the REAL native loss; not training/inference.

    Uses foreground targets for all four classes so box/DFL comparisons cannot
    pass vacuously at zero. No dataset images or model forward passes are used.
    """
    import torch
    from types import SimpleNamespace
    from ultralytics.utils.loss import v8DetectionLoss
    native_loss_structure()
    generator = torch.Generator().manual_seed(1241)
    scores = torch.randn((1,4,64),generator=generator)
    boxes = torch.randn((1,64,64),generator=generator)
    batch = {
        "batch_idx":torch.zeros(4),"cls":torch.arange(4).reshape(-1,1),
        "bboxes":torch.tensor([[.25,.25,.5,.5],[.75,.25,.5,.5],
                              [.25,.75,.5,.5],[.75,.75,.5,.5]]),
    }

    class LossOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device_parameter = torch.nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
            self.model = [SimpleNamespace(nc=4,reg_max=16,stride=torch.tensor([8.]))]

    records,details = {},{}
    for variant in ["CONTROL",*EXPERIMENTS]:
        model = LossOnlyModel()
        original_parameters = [(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()]
        if variant != "CONTROL":
            cls = make_class_weight_trainer(variant)
            trainer = object.__new__(cls)
            trainer.model,trainer.args,trainer.data = model,SimpleNamespace(cls_pw=0.0),{"nc":4,"names":NAMES}
            # Two calls must overwrite with the same vector, not square/multiply it.
            trainer.set_class_weights()
            trainer.set_class_weights()
        require(original_parameters == [(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()],
                "Class-weight assignment changed model parameters")
        criterion = model.criterion = v8DetectionLoss(model)
        raw = {}
        def capture_bce(module,inputs,output):
            raw["bce"] = output.detach().clone()  # native loss multiplies its output in-place afterwards
        handle = criterion.bce.register_forward_hook(capture_bce)
        def capture_targets(module,inputs,output):
            raw["target_scores_sum"] = output[2].sum().clamp(min=1).detach().clone()
        assignment_handle = criterion.assigner.register_forward_hook(capture_targets)
        pred_scores = scores.clone().requires_grad_()
        pred_boxes = boxes.clone().requires_grad_()
        preds = {"scores":pred_scores,"boxes":pred_boxes,"feats":[torch.zeros(1,68,8,8)]}
        assignment,loss,_ = criterion.get_assigned_targets_and_loss(preds,batch)
        handle.remove()
        assignment_handle.remove()
        require(assignment[0].any().item() and loss[0].item() > 0 and loss[2].item() > 0,
                "Sanity fixture must exercise real foreground box/DFL losses")
        grad_cls = torch.autograd.grad(loss[1],pred_scores,retain_graph=True)[0]
        grad_box,grad_dfl = [torch.autograd.grad(loss[i],pred_boxes,retain_graph=True)[0] for i in (0,2)]
        weight = torch.ones(4) if variant == "CONTROL" else model.class_weights.detach()
        weighted = raw["bce"]*weight.view(1,1,4)
        expected_cls = weighted.sum()/raw["target_scores_sum"]*model.args.cls
        require(torch.allclose(loss[1],expected_cls,rtol=1e-6,atol=1e-6),
                "Weighted classification scalar/normalizer/gain differs")
        records[variant] = {
            "loss":loss.detach().clone(),"raw_bce":raw["bce"],"weighted_bce":weighted,
            "cls_grad":grad_cls.detach(),"box_grad":grad_box.detach(),"dfl_grad":grad_dfl.detach(),
            "foreground":assignment[0].clone(),
        }
        if variant != "CONTROL":
            details[variant] = actual_weight_state(model,variant,require_criterion=True)
    control = records["CONTROL"]
    for variant in EXPERIMENTS:
        record = records[variant]
        weights = torch.tensor(EXPERIMENTS[variant]["weights"])
        require(torch.equal(record["raw_bce"],control["raw_bce"]),"Predictions/targets/assignment changed")
        require(torch.equal(record["foreground"],control["foreground"]),"Assignment changed")
        ratios = (record["weighted_bce"].sum((0,1))/control["weighted_bce"].sum((0,1)))
        require(torch.allclose(ratios,weights,rtol=1e-6,atol=1e-6),"Per-class BCE contribution differs")
        require(torch.allclose(record["cls_grad"],control["cls_grad"]*weights.view(1,4,1),
                               rtol=1e-5,atol=1e-6),"Class gradients were weighted twice or on the wrong axis")
        for index,key in [(0,"box_grad"),(2,"dfl_grad")]:
            require(torch.equal(record["loss"][index],control["loss"][index]) and
                    torch.equal(record[key],control[key]),"Box/DFL value or gradient changed")
        details[variant].update(
            per_class_bce_ratio=ratios.tolist(),box_loss_unchanged=True,dfl_loss_unchanged=True,
            box_gradient_unchanged=True,dfl_gradient_unchanged=True,
            assignment_unchanged=True,no_double_weighting=True)
    return {"status":"PASS","kind":"native-loss tensor-only unit check; no training/inference",
            "class_order":list(NAMES.values()),"variants":details}


def run_training(info, reference, variant, torch, YOLO):
    settings = training_settings(reference)
    CustomTrainer = make_class_weight_trainer(variant)
    model = YOLO(str(WEIGHTS), task="detect")
    require(model.task == "detect", "Only YOLOv8n Detection is permitted")
    info["requested_training_args"] = settings
    save_info(info)

    def on_start(trainer):
        from torch.utils.data import RandomSampler
        require(type(trainer.train_loader.sampler) is RandomSampler and
                trainer.train_loader.sampler.replacement is False and
                len(trainer.train_loader.sampler) == 604,
                "E5 requires the unchanged E1 unweighted random Train sampler")
        require(trainer.args.cls_pw == 0.0,"Frequency weighting must stay disabled")
        model_weight = actual_weight_state(trainer.model,variant)
        ema_weight = actual_weight_state(trainer.ema.ema,variant)
        info["class_weights"]["effective_model"] = model_weight
        info["class_weights"]["effective_ema"] = ema_weight
        info["sampler"] = {"type":"RandomSampler","replacement":False,"samples_per_epoch":604,
                           "weighted_sampling":False}
        (RUN/"class_weight_config.json").write_text(json.dumps(info["class_weights"],indent=2)+"\n",encoding="utf-8")
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
        if not info["class_weights"].get("native_criterion_verified",False):
            info["class_weights"]["actual_loss"] = actual_weight_state(
                trainer.model,variant,require_criterion=True)
            info["class_weights"]["native_criterion_verified"] = True
            (RUN/"class_weight_config.json").write_text(
                json.dumps(info["class_weights"],indent=2)+"\n",encoding="utf-8")
            save_info(info)

    def on_epoch_end(trainer):
        if int(trainer.epoch)+1 != info.get("current_epoch"):
            return  # Ignore the extra callback used by Ultralytics final best validation.
        metrics = {k:float(v) for k,v in trainer.metrics.items()}
        require(all(math.isfinite(v) for v in metrics.values()), "Non-finite training/validation metric")
        info.update(actual_completed_epochs=int(trainer.epoch)+1, latest_validation_metrics=metrics)
        save_info(info)
        print("E5_CLASS_WEIGHT_PROGRESS="+json.dumps({
            "epoch":info["actual_completed_epochs"], "best_epoch":info["best_epoch"]}), flush=True)

    for event, callback in [
        ("on_train_start",on_start), ("on_train_epoch_start",on_epoch_start),
        ("on_train_batch_start",on_batch_start), ("on_train_batch_end",on_batch_end), ("on_model_save",on_save),
        ("on_fit_epoch_end",on_epoch_end)]:
        model.add_callback(event,callback)
    started = time.perf_counter()
    model.train(trainer=CustomTrainer,**settings)
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
    require(info["class_weights"].get("native_criterion_verified",False),"Actual classification weight was not verified")
    info["early_stopping"] = {
        "triggered":info["actual_completed_epochs"] < settings["epochs"],
        "patience":settings["patience"],"criterion":"unchanged Ultralytics Val fitness",
        "stopper_best_epoch":int(model.trainer.stopper.best_epoch),
        "requested_epochs":settings["epochs"],"actual_completed_epochs":info["actual_completed_epochs"],
    }
    evaluate_best_validation(info,YOLO)
    info.update(status="COMPLETED", training_success=True, error=None,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                training_seconds=fit_seconds,
                best_pt_path=str(RUN/"weights/best.pt"), last_pt_path=str(RUN/"weights/last.pt"),
                test_evaluated=False,validation_evaluated=True)
    save_info(info)
    (RUN/"experiment_summary.md").write_text("\n".join([
        "# E5 class-wise BCE weight / Colab GPU", "",
        "- Status: COMPLETED; Train604 / Val67 / Test66, fixed general_v2 split.",
        "- Requested epochs100; completed "+str(info["actual_completed_epochs"])+"; best "+str(info["best_epoch"])+".",
        "- imgsz640 / batch8 / seed42 / patience20 / device0 / workers0 / deterministic=True.",
        "- Baseline optimizer=auto policy retained; actual AdamW, lr0.00125, betas(0.9,0.999) verified.",
        "- AMP=False; all baseline online augmentation disabled. Fixed native BCE class vector only; no new augmentation or sampling changes.",
        "- Dataset hash QA and loader image/annotation counts passed.",
        "- Val67 alone selects/compares checkpoints. Test66 identity/hash/count/leakage only; no inference.",
        "- GPU, Python/Torch/CUDA and Ultralytics must match completed E1; no package source edits.",
        "- Full environment, optimizer groups, hashes and actual args: experiment_info.json.",
        "- Best: "+info["best_pt_path"], "",
    ]), encoding="utf-8")
    print("E5_CLASS_WEIGHT_COMPLETED="+json.dumps({
        k:info[k] for k in ["training_success","actual_completed_epochs","best_epoch","best_pt_path"]}), flush=True)

def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",required=True,choices=list(EXPERIMENTS))
    parser.add_argument("--check-only",action="store_true",
                        help="Colab GPU/E1 environment and local loss/data checks only; no training/inference")
    parser.add_argument("--qa-only",action="store_true",
                        help="CPU-safe data/native-loss checks only; no training/model inference")
    parser.add_argument("--dataset-root",type=Path,help="Read-only local E1 path; --qa-only only")
    parser.add_argument("--qa-output",type=Path,help="New QA JSON; --qa-only only")
    args = parser.parse_args()
    require(not (args.check_only and args.qa_only),"Select one check mode")
    require(args.qa_only or (args.dataset_root is None and args.qa_output is None),
            "Training must use the frozen Drive E1 dataset path")
    RUN = DRIVE/"runs/v2_class_weight"/EXPERIMENTS[args.variant]["run_name"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    if args.qa_only:
        folder = args.dataset_root or DATA
        report = {
            "variant":args.variant,"class_weights":EXPERIMENTS[args.variant]["weights"],
            "loss_structure":native_loss_structure(),"loss_unit_checks":loss_sanity_check(),
            "dataset_qa":verify_dataset(folder),
            "training_executed":False,"inference_executed":False,
            "test_evaluated":False,"test_evaluation_calls":0,
            "script_sha256":digest(Path(__file__)),
            "colab_gpu_and_actual_E1_run_preflight":"PENDING_ON_COLAB",
        }
        if args.qa_output:
            require(not args.qa_output.exists(),"Never overwrite an existing QA report")
            args.qa_output.parent.mkdir(parents=True,exist_ok=True)
            args.qa_output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
        return
    gpu = gpu_preflight(torch)
    require(int(os.environ.get("WORLD_SIZE","1")) == 1,"E5 is single GPU to match E1")
    prepared = json.loads((DATA/"baseline_reference.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("ultralytics")
    require(version == prepared["ultralytics_version"] == "8.4.174","Ultralytics version drift")
    require(importlib.util.find_spec("albumentations") is None,
            "Albumentations must be absent to preserve E1 augmentation policy")
    structure = native_loss_structure()
    unit_checks = loss_sanity_check()
    qa = verify_dataset(DATA)
    corrected = corrected_run_reference(torch,gpu,prepared)
    reference = general_run_reference(torch,gpu,prepared,corrected)
    require(digest(WEIGHTS) == PRETRAINED_SHA256 == reference["pretrained_weights_sha256"],
            "Official pretrained weights differ")
    settings = training_settings(reference)
    require(reference["effective_training_args"]["cls_pw"] == 0.0,
            "E1 must have no native frequency weighting")
    print("DATASET_QA="+json.dumps(qa),flush=True)
    print("LOSS_UNIT_CHECKS="+json.dumps(unit_checks),flush=True)
    print("CLASS_WEIGHTS="+json.dumps(EXPERIMENTS[args.variant]),flush=True)
    print("TRAINING_ARGS="+json.dumps(settings),flush=True)
    if args.check_only:
        print("READY: E5 GPU/E1/loss/dataset QA passed; no training or inference.",flush=True)
        return
    require(not RUN.exists(),"Run exists; never overwrite or silently resume")
    RUN.mkdir(parents=True,exist_ok=False)
    script = Path(__file__).resolve()
    shutil.copy2(script,RUN/script.name)  # Includes the entire custom trainer; no hidden helper.
    from ultralytics.utils.loss import v8DetectionLoss
    (RUN/"native_loss_source.py.txt").write_text(
        inspect.getsource(v8DetectionLoss),encoding="utf-8")
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
        "variant":args.variant,"class_order":list(NAMES.values()),
        "class_weights":{"requested":EXPERIMENTS[args.variant]["weights"],
                         "normalization":None,"automatic_frequency_weighting":False,
                         "native_criterion_verified":False,
                         "scope":"entire per-class positive/negative BCE, once"},
        "loss_structure":structure,"loss_preflight":unit_checks,
        "dataset_path":str(DATA),"data_yaml":str(DATA/"data_colab.yaml"),"dataset_qa":qa,
        "train_val_test_image_counts":COUNTS,"model":"yolov8n.pt",
        "pretrained_weights_path":str(WEIGHTS),"pretrained_weights_sha256":PRETRAINED_SHA256,
        "epochs_requested":100,"actual_completed_epochs":0,"best_epoch":None,
        "imgsz":640,"batch":8,"seed":42,"script_path":str(script),"script_sha256":digest(script),
        "test_evaluated":False,"test_evaluation_calls":0,"validation_evaluation_calls":0,
        "selection_split":"val","validation_only":True,
        "comparison_baselines":reference["verified_run"],
        "sampler":{"weighted_sampling":False,"policy":"unchanged E1 native RandomSampler"},
        "cache_policy":"Suppress disposable label-cache writes; do not modify the E1 dataset",
    }
    (RUN/"class_weight_config.json").write_text(json.dumps(info["class_weights"],indent=2)+"\n",encoding="utf-8")
    save_info(info)
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in [
            "sync","wandb","mlflow","comet","clearml","dvc","neptune","raytune","tensorboard","hub"] if k in SETTINGS})
        from unittest.mock import patch
        with patch("ultralytics.data.dataset.save_dataset_cache_file",side_effect=lambda *a,**k: None):
            run_training(info,reference,args.variant,torch,YOLO)
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
        require(row["split"] == parent["split"] == "train" and row["existing_augmentation"] == "True" and
                row["parent_image_id"] == parent["image_id"] and row["filename"] == entry["file_name"] and
                row["group_key"] == parent["group_key"] and int(parent["o_count"])+int(parent["s_count"]) > 0,
                "Augmentation parent/split provenance mismatch")
        parents.add(parent["image_id"])
    require(len(parents) == 35, "Existing augmentation parent count changed")

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
            "Completed E1 run missing on Drive; E5 training is blocked: "+str(GENERAL_RUN))
    info = json.loads((GENERAL_RUN/"experiment_info.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((GENERAL_RUN/"args.yaml").read_text(encoding="utf-8"))
    require(info.get("status") == "COMPLETED" and info.get("training_success") is True and
            info.get("train_val_test_image_counts") == COUNTS and info.get("model") == "yolov8n.pt" and
            info.get("pretrained_weights_sha256") == PRETRAINED_SHA256 and
            info.get("dataset_qa",{}).get("manifest_sha256") == MANIFEST_SHA256,
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
    require(not mismatches,"E1 environment differs; no E5 training: "+json.dumps(mismatches))
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

def evaluate_best_validation(info,YOLO):
    """One standalone best.pt Val67 evaluation; never a Test loader/inference."""
    require(not info.get("test_evaluated") and info.get("test_evaluation_calls",0) == 0,
            "Test performance evaluation is forbidden")
    require(info.get("validation_evaluation_calls",0) == 0,"Best validation already executed")
    require(verify_dataset(DATA) == info["dataset_qa"],"Frozen E1 dataset changed")
    best = YOLO(str(RUN/"weights/best.pt"),task="detect")
    info["checkpoint_class_weights"] = actual_weight_state(best.model,info["variant"])
    effective = {}
    def on_val_start(validator):
        require(validator.args.split == "val","Only Val67 metrics are permitted")
        info.setdefault("loader_qa",{})["val_best"] = loader_check(validator.dataloader.dataset,"val")
        effective.update(vars(validator.args))
    best.add_callback("on_val_start",on_val_start)
    settings = dict(data=str(DATA/"data_colab.yaml"),split="val",imgsz=640,batch=8,device=0,
                    workers=0,plots=True,augment=False,project=str(RUN),name="val_best",exist_ok=False,verbose=True)
    info["validation_evaluation_calls"] = 1
    save_info(info)
    result = pack_validation_metrics(best.val(**settings))
    result.update(variant=info["variant"],class_weights=EXPERIMENTS[info["variant"]]["weights"],
                  evaluation_settings=effective,checkpoint=str(RUN/"weights/best.pt"))
    (RUN/"validation_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    info.update(validation=result,validation_evaluated=True,test_evaluated=False,test_evaluation_calls=0)
    lines = ["# "+info["variant"]+" / Validation67 only","",
             "Class order o/p/s/sc; weights "+str(EXPERIMENTS[info["variant"]]["weights"])+".",
             "Test66 integrity only; no Test performance or qualitative prediction inspection.","",
             "| Metric | Value |","|---|---:|"]
    lines += [f"| {k} | {v:.8f} |" for k,v in result["overall"].items()]
    lines += ["","| Class | GT | Precision | Recall | AP50 | AP50-95 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,m in result["per_class"].items():
        lines.append("| "+name+" | "+str(m["instances"])+" | "+
                     " | ".join(f"{m[k]:.8f}" for k in ["precision","recall","ap50","ap50_95"])+" |")
    (RUN/"validation_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    require(verify_dataset(DATA) == info["dataset_qa"],"Dataset changed during validation")
    save_info(info)



if __name__ == "__main__":
    main()
