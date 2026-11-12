"""E4 CUDA-only YOLOv8n image-level weighted sampling; E1 general_v2 is read-only.

Use --sampler-check-only --dataset-root PATH for CPU-safe sampler/data diagnostics
(no model, optimization or inference). Normal execution requires the verified E1
Colab GPU environment. Only Val67 may be used for model selection/comparison.
Test66 is checked for identity/hash/count/leakage only, never sent to a model.
Train draws 604 samples WITH replacement per epoch; rare image weight is 2,
otherwise 1 (o+s is still 2). This is a per-image relative exposure ratio, not
a guarantee of two draws for each rare image per epoch. No class-weighted loss.
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
DATA = DRIVE / "datasets/yolov8_detection_general_v2"
WEIGHTS = DRIVE / "models/yolov8n.pt"
RUN = DRIVE / "runs/v2_weighted_sampling/yolov8n_weighted_sampling_v2"
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
            not settings.get("resume", False), "E4 requires fresh auto-optimizer training and Val only")
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


def image_sampling_weight(classes):
    """One union weight: o+s is 2, never 4; p/sc/empty are 1."""
    return 2.0 if set(classes) & {0,2} else 1.0


def read_sampling_population(folder):
    """Read TRAIN labels only; image-level union of o/s, never annotation frequency."""
    with (folder/"image_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = sorted((r for r in csv.DictReader(f) if r["split"] == "train"),
                      key=lambda r: r["filename"])
    population = []
    for row in rows:
        label = folder/"train/labels"/(Path(row["filename"]).stem+".txt")
        classes = {int(line.split()[0]) for line in label.read_text(encoding="ascii").splitlines()}
        rare = bool(classes & {0, 2})
        population.append({"filename":row["filename"], "classes":sorted(classes),
                           "rare_positive":rare, "weight":image_sampling_weight(classes)})
    require(len(population) == 604 and len({r["filename"] for r in population}) == 604,
            "Train sampling population mismatch")
    return population


def sampling_configuration(population):
    rare = sum(r["rare_positive"] for r in population)
    normal = len(population)-rare
    require(rare > 0 and normal > 0, "Both sampling strata must be present")
    total_weight = 2*rare+normal
    return {
        "sampler":"torch.utils.data.WeightedRandomSampler",
        "unit":"train image (including existing E1 augmented images)",
        "replacement":True, "num_samples_per_epoch":len(population), "seed":42,
        "epoch_seed_rule":"42 + zero_based_epoch; separate CPU torch.Generator",
        "rare_definition":"at least one class 0=o or 2=s annotation",
        "rare_weight":2.0, "normal_weight":1.0, "both_o_and_s_weight":2.0,
        "rare_positive_images":rare, "normal_images":normal,
        "o_positive_images":sum(0 in r["classes"] for r in population),
        "s_positive_images":sum(2 in r["classes"] for r in population),
        "both_positive_images":sum(0 in r["classes"] and 2 in r["classes"] for r in population),
        "expected_rare_draws_per_epoch":len(population)*2*rare/total_weight,
        "expected_normal_draws_per_epoch":len(population)*normal/total_weight,
        "expected_per_rare_image_draws_per_epoch":len(population)*2/total_weight,
        "expected_per_normal_image_draws_per_epoch":len(population)/total_weight,
        "expected_per_image_rare_to_normal_ratio":2.0,
        "note":"Weights double rare vs normal per-image probability, not absolute baseline exposure. "
               "Finite-epoch counts fluctuate; the dataset and epoch length do not increase.",
    }


def make_sampler(weights):
    """Seeded WeightedRandomSampler with an explicit epoch and an auditable draw list."""
    import torch
    class EpochWeightedSampler(torch.utils.data.WeightedRandomSampler):
        def __init__(self):
            super().__init__(weights, num_samples=len(weights), replacement=True,
                             generator=torch.Generator(device="cpu").manual_seed(42))
            self.drawn_indices = []
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = int(epoch)
            self.generator.manual_seed(42+self.epoch)
            self.drawn_indices = []

        def __iter__(self):
            self.drawn_indices = list(super().__iter__())
            return iter(self.drawn_indices)
    return EpochWeightedSampler()


def exposure_statistics(population, counts):
    require(len(counts) == len(population) and all(v >= 0 for v in counts), "Invalid draw counters")
    nr = sum(r["rare_positive"] for r in population)
    nn = len(population)-nr
    rare_draws = sum(n for r,n in zip(population,counts) if r["rare_positive"])
    total = sum(counts)
    normal_draws = total-rare_draws
    rare_per_image, normal_per_image = rare_draws/nr, normal_draws/nn
    return {
        "sample_count":total, "rare_draws":rare_draws, "normal_draws":normal_draws,
        "unique_sampled_images":sum(n > 0 for n in counts),
        "rare_draw_fraction":rare_draws/total if total else None,
        "per_rare_image_exposure":rare_per_image, "per_normal_image_exposure":normal_per_image,
        "per_image_rare_to_normal_exposure_ratio":
            rare_per_image/normal_per_image if normal_per_image else None,
        "o_positive_draws":sum(n for r,n in zip(population,counts) if 0 in r["classes"]),
        "s_positive_draws":sum(n for r,n in zip(population,counts) if 2 in r["classes"]),
        "both_positive_draws":sum(n for r,n in zip(population,counts)
                                  if 0 in r["classes"] and 2 in r["classes"]),
    }


def sampler_diagnostic(population, epochs=100):
    """Index-only rehearsal, NOT training exposure, predictions or model evaluation."""
    sampler = make_sampler([r["weight"] for r in population])
    totals = [0]*len(population)
    records = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        counts = Counter(iter(sampler))
        require(sum(counts.values()) == len(population), "Sampler epoch length changed")
        for index,count in counts.items():
            totals[index] += count
        records.append({"epoch":epoch+1, "seed":42+epoch,
                        **exposure_statistics(population,[counts[i] for i in range(len(population))])})
    return {"kind":"INDEX_ONLY_REHEARSAL_NOT_ACTUAL_TRAINING",
            "epochs":epochs, "epoch_statistics":records,
            "overall":exposure_statistics(population,totals)}


def make_weighted_trainer(population):
    """Override Train loader only; model/loss/validation remain DetectionTrainer defaults.

    preprocess_batch only observes file identities, then delegates unchanged.
    workers=0 + reset at epoch start prevents InfiniteDataLoader prefetch/epoch bleed.
    """
    import torch
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer
    by_filename = {r["filename"]:r for r in population}

    class WeightedSamplingTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            require(mode in {"train","val"}, "E4 forbids a Test loader")
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "val":
                return loader
            require(rank == -1 and self.world_size == 1 and self.args.workers == 0 and
                    not self.args.rect and not self.args.compile and not loader.drop_last,
                    "E4 supports single-GPU, workers0, non-rect, non-compiled training only")
            loader_check(loader.dataset,"train")
            ordered = []
            for path,label in zip(loader.dataset.im_files, loader.dataset.labels):
                row = by_filename[Path(path).name]
                classes = {int(v) for v in label["cls"].reshape(-1)}
                require(classes == set(row["classes"]), "Loaded rare membership differs")
                ordered.append(row)
            require(len(ordered) == len(by_filename) == len(loader.dataset.im_files) and
                    {r["filename"] for r in ordered} == set(by_filename), "Train loader identity differs")
            self.sampling_population = ordered
            self.sampling_index = {r["filename"]:i for i,r in enumerate(ordered)}
            self.sampling_counts = Counter()
            self.sampling_totals = [0]*len(ordered)
            self.recorded_sampling_epochs = set()
            sampler = make_sampler([r["weight"] for r in ordered])
            weighted = InfiniteDataLoader(
                dataset=loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                shuffle=False, num_workers=0, pin_memory=loader.pin_memory,
                collate_fn=loader.collate_fn, worker_init_fn=loader.worker_init_fn,
                generator=loader.generator, drop_last=False)
            require(len(weighted) == len(loader) == 76, "E1 steps per epoch changed")
            loader.close()
            return weighted

        def preprocess_batch(self, batch):
            # Counter only: never alters pixels/labels, loss, model or preprocessing.
            for path in batch["im_file"]:
                require(Path(path).name in self.sampling_index, "Non-Train image in weighted batch")
                self.sampling_counts[self.sampling_index[Path(path).name]] += 1
            return super().preprocess_batch(batch)

    return WeightedSamplingTrainer


def record_sampling_epoch(trainer, info):
    epoch = int(trainer.epoch)+1
    require(epoch not in trainer.recorded_sampling_epochs, "Duplicate epoch statistics")
    population = trainer.sampling_population
    counts = [trainer.sampling_counts[i] for i in range(len(population))]
    planned = Counter(trainer.train_loader.sampler.drawn_indices)
    require(sum(counts) == 604 and trainer.sampling_counts == planned,
            "Consumed Train batches differ from sampler draws or OOM replay occurred")
    row = {"epoch":epoch, "seed":42+epoch-1, **exposure_statistics(population,counts)}
    path = RUN/"sampling_epoch_counts.csv"
    with path.open("a",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    with (RUN/"sampling_image_counts.csv").open("a",encoding="utf-8",newline="") as f:
        fields = ["epoch","filename","rare_positive","weight","observed_count"]
        writer = csv.DictWriter(f,fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows({"epoch":epoch,"filename":r["filename"],"rare_positive":r["rare_positive"],
                          "weight":r["weight"],"observed_count":n} for r,n in zip(population,counts))
    trainer.sampling_totals = [a+b for a,b in zip(trainer.sampling_totals,counts)]
    trainer.recorded_sampling_epochs.add(epoch)
    overall = exposure_statistics(population,trainer.sampling_totals)
    info["sampling"] = {"configuration":sampling_configuration(population),
                        "completed_epochs":len(trainer.recorded_sampling_epochs),
                        "latest_epoch":row, "overall":overall,
                        "counts_source":"actual training preprocess_batch im_file, verified against sampler indices"}
    (RUN/"sampling_summary.json").write_text(json.dumps(info["sampling"],indent=2)+"\n",encoding="utf-8")
    print("E4_SAMPLING="+json.dumps(row),flush=True)


def run_training(info, reference, population, torch, YOLO):
    settings = training_settings(reference)
    WeightedTrainer = make_weighted_trainer(population)
    model = YOLO(str(WEIGHTS), task="detect")
    require(model.task == "detect", "Only YOLOv8n Detection is permitted")
    info["sampler_configuration"] = sampling_configuration(population)
    (RUN/"sampler_config.json").write_text(json.dumps(info["sampler_configuration"],indent=2)+"\n",encoding="utf-8")
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
        trainer.train_loader.sampler.set_epoch(int(trainer.epoch))
        trainer.train_loader.reset()
        trainer.sampling_counts.clear()
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
        record_sampling_epoch(trainer,info)
        metrics = {k:float(v) for k,v in trainer.metrics.items()}
        require(all(math.isfinite(v) for v in metrics.values()), "Non-finite training/validation metric")
        info.update(actual_completed_epochs=int(trainer.epoch)+1, latest_validation_metrics=metrics)
        save_info(info)
        print("WEIGHTED_SAMPLING_V2_PROGRESS="+json.dumps({
            "epoch":info["actual_completed_epochs"], "best_epoch":info["best_epoch"]}), flush=True)

    for event, callback in [
        ("on_train_start",on_start), ("on_train_epoch_start",on_epoch_start),
        ("on_train_batch_start",on_batch_start), ("on_model_save",on_save),
        ("on_fit_epoch_end",on_epoch_end)]:
        model.add_callback(event,callback)
    started = time.perf_counter()
    model.train(trainer=WeightedTrainer, **settings)
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
        "# E4 weighted sampling YOLOv8n / Colab GPU", "",
        "- Status: COMPLETED; Train604 / Val67 / Test66, fixed E4 weighted sampling split.",
        "- Requested epochs100; completed "+str(info["actual_completed_epochs"])+"; best "+str(info["best_epoch"])+".",
        "- imgsz640 / batch8 / seed42 / patience20 / device0 / workers0 / deterministic=True.",
        "- Baseline optimizer=auto policy retained; actual AdamW, lr0.00125, betas(0.9,0.999) verified.",
        "- AMP=False; all baseline online augmentation disabled. No new augmentation; weighted image sampling is the only intended change.",
        "- Dataset hash QA and loader image/annotation counts passed.",
        "- Val67 selects the checkpoint and is the only official comparison set. See validation_results.md.",
        "- Test66 is preserved for integrity checks only; no Test performance evaluation is run.",
        "- GPU, Python/Torch/CUDA and Ultralytics must match completed E0/E1 runs.",
        "- Full environment, optimizer groups, hashes and actual args: experiment_info.json.",
        "- Best: "+info["best_pt_path"], "",
    ]), encoding="utf-8")
    print("WEIGHTED_SAMPLING_V2_COMPLETED="+json.dumps({
        k:info[k] for k in ["training_success","actual_completed_epochs","best_epoch","best_pt_path"]}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify Colab GPU/E1 environment/data/sampler; do not train")
    parser.add_argument("--sampler-check-only",action="store_true",
                        help="CPU-safe index-only rehearsal; no model or inference")
    parser.add_argument("--dataset-root",type=Path,
                        help="Local E1 path, permitted only with --sampler-check-only")
    parser.add_argument("--audit-output",type=Path,
                        help="New diagnostic JSON, permitted only with --sampler-check-only")
    args = parser.parse_args()
    require(not (args.check_only and args.sampler_check_only), "Select only one check mode")
    require(args.sampler_check_only or (args.dataset_root is None and args.audit_output is None),
            "Training paths are frozen; local paths are diagnostic-only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if args.sampler_check_only:
        folder = args.dataset_root or DATA
        qa = verify_dataset(folder)
        population = read_sampling_population(folder)
        report = {"dataset_qa":qa, "sampler_configuration":sampling_configuration(population),
                  "sampling_diagnostic":sampler_diagnostic(population),
                  "training_executed":False, "inference_executed":False,
                  "test_evaluated":False,"test_evaluation_calls":0,
                  "torch":str(torch.__version__),
                  "script_sha256":digest(Path(__file__))}
        if args.audit_output:
            require(not args.audit_output.exists(),"Never overwrite a diagnostic")
            args.audit_output.parent.mkdir(parents=True,exist_ok=True)
            args.audit_output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k != "sampling_diagnostic"},indent=2))
        print("INDEX_ONLY_SAMPLING="+json.dumps(report["sampling_diagnostic"]["overall"]),flush=True)
        return
    gpu = gpu_preflight(torch)  # CPU fallback is forbidden, before model construction.
    require(int(os.environ.get("WORLD_SIZE","1")) == 1,"E4 supports single-GPU only")
    prepared = json.loads((DATA/"baseline_reference.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("ultralytics")
    require(version == prepared["ultralytics_version"] == "8.4.174","Ultralytics version drift")
    require(importlib.util.find_spec("albumentations") is None,
            "Albumentations must be absent to match E1; no implicit augmentation")
    qa = verify_dataset(DATA)
    population = read_sampling_population(DATA)
    corrected = corrected_run_reference(torch,gpu,prepared)
    reference = general_run_reference(torch,gpu,prepared,corrected)
    require(digest(WEIGHTS) == PRETRAINED_SHA256 == reference["pretrained_weights_sha256"],
            "Official pretrained weights differ")
    settings = training_settings(reference)
    rehearsal = sampler_diagnostic(population)
    print("DATASET_QA="+json.dumps(qa),flush=True)
    print("SAMPLER_CONFIG="+json.dumps(sampling_configuration(population)),flush=True)
    print("INDEX_ONLY_REHEARSAL="+json.dumps(rehearsal["overall"]),flush=True)
    print("TRAINING_ARGS="+json.dumps(settings),flush=True)
    if args.check_only:
        print("READY: Colab GPU/E1/dataset/sampler preflight PASS; no training or inference.",flush=True)
        return
    require(not RUN.exists(),"Run already exists; never overwrite or silently resume")
    RUN.mkdir(parents=True,exist_ok=False)
    script = Path(__file__).resolve()
    shutil.copy2(script,RUN/script.name)
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
        "dataset_path":str(DATA),"data_yaml":str(DATA/"data_colab.yaml"),"dataset_qa":qa,
        "train_val_test_image_counts":COUNTS,"model":"yolov8n.pt",
        "pretrained_weights_path":str(WEIGHTS),"pretrained_weights_sha256":PRETRAINED_SHA256,
        "epochs_requested":100,"actual_completed_epochs":0,"best_epoch":None,
        "imgsz":640,"batch":8,"seed":42,"script_path":str(script),"script_sha256":digest(script),
        "test_evaluated":False,"test_evaluation_calls":0,"validation_evaluation_calls":0,
        "selection_split":"val","validation_only":True,
        "comparison_baselines":reference["verified_run"],
        "sampler_configuration":sampling_configuration(population),
        "sampler_rehearsal":rehearsal,
        "cache_policy":"Disposable Ultralytics label-cache writes suppressed; E1 images/labels/metadata unchanged",
        "environment_note":"Strict E1 environment verification; hardware/version bitwise identity is not guaranteed",
    }
    save_info(info)
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in [
            "sync","wandb","mlflow","comet","clearml","dvc","neptune","raytune","tensorboard","hub"] if k in SETTINGS})
        # This affects temporary cache persistence only, never image/label contents or scanning logic.
        from unittest.mock import patch
        with patch("ultralytics.data.dataset.save_dataset_cache_file", side_effect=lambda *a,**k: None):
            run_training(info,reference,population,torch,YOLO)
    except BaseException as error:
        info.update(status="FAILED",training_success=False,
                    error=f"{type(error).__name__}: {error}",
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
            "Completed E1 run missing on Drive; E4 training is blocked: "+str(GENERAL_RUN))
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
    require(not mismatches,"E1 environment differs; no E4 training: "+json.dumps(mismatches))
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
    """Compare E4 and frozen E1 best.pt on Val67 only; save everything under E4."""
    require(not info.get("test_evaluated") and info.get("test_evaluation_calls",0) == 0,
            "Test performance evaluation is forbidden")
    require(info.get("validation_evaluation_calls",0) == 0,"Validation comparison already executed")
    require(verify_dataset(DATA) == info["dataset_qa"],"E1 dataset identity changed")
    baseline_checkpoint = GENERAL_RUN/"weights/best.pt"
    baseline_hash = digest(baseline_checkpoint)
    results = {}
    for experiment,checkpoint,subfolder in [
        ("E4",RUN/"weights/best.pt","val_best"),
        ("E1",baseline_checkpoint,"E1_val_reference")]:
        model = YOLO(str(checkpoint),task="detect")
        effective = {}
        def on_val_start(validator):
            require(validator.args.split == "val","E4 allows only Val67 metrics")
            info.setdefault("loader_qa",{})[experiment+"_val"] = loader_check(
                validator.dataloader.dataset,"val")
            effective.update(vars(validator.args))
        model.add_callback("on_val_start",on_val_start)
        settings = dict(data=str(DATA/"data_colab.yaml"),split="val",imgsz=640,batch=8,
                        device=0,workers=0,plots=True,augment=False,
                        project=str(RUN),name=subfolder,exist_ok=False,verbose=True)
        info["validation_evaluation_calls"] += 1
        save_info(info)
        result = pack_validation_metrics(model.val(**settings))
        result.update(experiment=experiment,checkpoint=str(checkpoint),
                      checkpoint_sha256=digest(checkpoint),evaluation_settings=effective)
        results[experiment] = result
        (RUN/("validation_metrics.json" if experiment == "E4" else "E1_validation_metrics.json")).write_text(
            json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        del model
    require(digest(baseline_checkpoint) == baseline_hash,"Frozen E1 checkpoint changed")
    protocol = ["data","split","imgsz","batch","device","workers","augment","conf","iou",
                "max_det","agnostic_nms","half","rect"]
    require(all(k in results["E1"]["evaluation_settings"] and k in results["E4"]["evaluation_settings"] and
                results["E1"]["evaluation_settings"][k] == results["E4"]["evaluation_settings"][k]
                for k in protocol),"E1/E4 Val evaluation protocol differs")
    comparison = []
    for metric in ["precision","recall","map50","map50_95"]:
        a,b = results["E1"]["overall"][metric],results["E4"]["overall"][metric]
        comparison.append({"class":"overall","metric":metric,"E1":a,"E4":b,"change":b-a})
    for name in NAMES.values():
        for metric in ["precision","recall","ap50","ap50_95"]:
            a,b = results["E1"]["per_class"][name][metric],results["E4"]["per_class"][name][metric]
            comparison.append({"class":name,"metric":metric,"E1":a,"E4":b,"change":b-a})
    with (RUN/"comparison_to_E1_val.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    info.update(validation=results["E4"],E1_validation=results["E1"],
                validation_comparison=comparison,validation_evaluated=True,
                test_evaluated=False,test_evaluation_calls=0)
    lines = ["# E4 vs E1 / Validation67 only","",
             "E1 best.pt is read-only; its Val metrics are computed with exactly the E4 Val protocol.",
             "Test66 is used only for identity/hash/count/leakage checks; never inference.",
             "Val results are experiment comparison/model-selection diagnostics, not untouched Test performance.","",
             "| Class | Metric | E1 | E4 | Change |","|---|---|---:|---:|---:|"]
    lines += [f"| {r['class']} | {r['metric']} | {r['E1']:.8f} | {r['E4']:.8f} | {r['change']:+.8f} |"
              for r in comparison]
    (RUN/"validation_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    require(verify_dataset(DATA) == info["dataset_qa"],"Dataset changed during evaluation")
    save_info(info)



if __name__ == "__main__":
    main()
