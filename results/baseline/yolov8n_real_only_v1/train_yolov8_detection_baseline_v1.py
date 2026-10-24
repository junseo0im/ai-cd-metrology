"""One real-only YOLOv8n detection baseline, fixed dataset, no parameter search."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/yolov8_detection_fixed_v1"
RUN = ROOT / "runs/baseline/yolov8n_real_only_v1"
REPORT = ROOT / "data/derived/yolov8_baseline_v1"
WEIGHTS = ROOT / "outputs/pretrained/yolov8n.pt"
NAMES = {0: "o", 1: "p", 2: "s", 3: "sc"}
COUNTS = {"train": 505, "val": 62, "test": 62}
CLASS_COUNTS = {
    "train": {"o": 2, "p": 4352, "s": 5, "sc": 399},
    "val": {"o": 0, "p": 546, "s": 1, "sc": 50},
    "test": {"o": 1, "p": 546, "s": 1, "sc": 50},
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def immutable_dataset():
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for p in DATA.rglob("*") if p.is_file() and
            (p.suffix.lower() in {".jpg", ".jpeg", ".png", ".txt", ".yaml", ".json", ".md"})}


def check_loader(dataset, split):
    if len(dataset) != COUNTS[split]:
        raise RuntimeError(f"{split}: loader excluded images ({len(dataset)} != {COUNTS[split]})")
    counts = {name: 0 for name in NAMES.values()}
    if len(dataset.labels) != COUNTS[split]:
        raise RuntimeError(f"{split}: loader label list changed")
    for item in dataset.labels:
        for value in item["cls"].reshape(-1):
            counts[NAMES[int(value)]] += 1
    if counts != CLASS_COUNTS[split]:
        raise RuntimeError(f"{split}: loader changed annotations: {counts}")
    return {"images": len(dataset), "class_counts": counts, "excluded_images": 0}


def evaluation_result(metrics, split):
    values = [float(x) for x in metrics.box.mean_results()]
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError(f"{split}: non-finite overall metrics")
    per_class = {}
    indices = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    for cls, name in NAMES.items():
        instances = CLASS_COUNTS[split][name]
        if instances == 0:
            p = r = ap50 = ap = None
            note = "No GT instances in this split; not evaluable."
        elif cls not in indices:
            raise RuntimeError(f"{split}: missing metrics for GT class {name}")
        else:
            p, r, ap50, ap = (float(x) for x in metrics.box.class_result(indices[cls]))
            if not all(math.isfinite(x) for x in (p, r, ap50, ap)):
                raise RuntimeError(f"{split}/{name}: non-finite class metrics")
            note = "Very small GT sample; numeric result only." if name in {"o", "s"} else ""
        per_class[name] = {
            "instances": instances, "precision": p, "recall": r,
            "ap50": ap50, "ap50_95": ap, "note": note,
        }
    seen = getattr(metrics, "nt_per_class", None)
    if seen is not None and list(map(int, seen)) != [CLASS_COUNTS[split][n] for n in NAMES.values()]:
        raise RuntimeError(f"{split}: evaluated GT counts differ: {seen}")
    return {
        "images": COUNTS[split],
        "overall": dict(zip(("precision", "recall", "map50", "map50_95"), values)),
        "per_class": per_class,
        "output_dir": str(metrics.save_dir),
        "note": "Open has zero GT instances in Val." if split == "val" else
                "Open and short have one GT instance each; no generalization claim.",
    }


def run_baseline(summary):
    # Existing immutable source verification; does not rewrite any dataset file.
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/convert_coco_detection_fixed_v1.py"),
                    "--verify-only"], cwd=ROOT, check=True)
    before = immutable_dataset()
    manifest = ROOT / "data/derived/fixed_split_v1/split_manifest.csv"
    manifest_hash = digest(manifest)
    summary.update(dataset_preflight="PASSED", dataset_files_sha256=before,
                   fixed_split_manifest_sha256=manifest_hash)
    write_json(REPORT / "baseline_summary.json", summary)

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR

    # No external tracking or optional package installation.
    SETTINGS.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml",
                                       "dvc", "neptune", "raytune", "tensorboard", "hub")
                     if k in SETTINGS})
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
    if font.is_file():
        destination = USER_CONFIG_DIR / "Arial.ttf"
        if not destination.exists():
            shutil.copy2(font, destination)

    device = 0 if torch.cuda.is_available() else "cpu"
    requested_zero = (
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
        "erasing", "bgr", "cutmix", "multi_scale",
    )
    augmentation_off = {key: 0 for key in requested_zero if key in DEFAULT_CFG_DICT}
    settings = {
        "data": str(DATA / "data.yaml"), "epochs": 100, "imgsz": 640,
        "seed": 42, "pretrained": True, "patience": 20, "device": device,
        "batch": 8, "workers": 0, "deterministic": True,
        "project": str(RUN.parent), "name": RUN.name, "exist_ok": True,
        "save": True, "plots": True, "val": True, "cache": False,
        "augment": False, "close_mosaic": 0, **augmentation_off,
    }
    if "auto_augment" in DEFAULT_CFG_DICT:
        settings["auto_augment"] = None
    # CPU AMP is disabled; avoid unrelated AMP-check model downloads.
    if device == "cpu":
        settings["amp"] = False
    model = YOLO(str(WEIGHTS), task="detect")
    if model.task != "detect":
        raise RuntimeError("Only the requested detection model is permitted")
    summary.update(
        status="TRAINING", device=str(device), batch=8, workers=0,
        environment={"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available(),
                     "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
        pretrained_weights_path=str(WEIGHTS), pretrained_weights_sha256=digest(WEIGHTS),
        augmentation_disabled=augmentation_off,
        unsupported_augmentation_options=[k for k in requested_zero if k not in DEFAULT_CFG_DICT],
        auto_augment=None, albumentations_installed=False,
        optimizer_policy="Installed Ultralytics defaults; no optimizer/lr/weight-decay overrides.",
        requested_train_args=settings, loader_validation={},
    )
    write_json(REPORT / "baseline_summary.json", summary)
    (REPORT / "training_config.yaml").write_text(
        yaml.safe_dump({"model": str(WEIGHTS), **settings}, sort_keys=False), encoding="utf-8")

    def on_start(trainer):
        summary["loader_validation"]["train"] = check_loader(trainer.train_loader.dataset, "train")
        summary["loader_validation"]["val"] = check_loader(trainer.test_loader.dataset, "val")
        summary["effective_training_args"] = vars(trainer.args).copy()
        (REPORT / "training_config.yaml").write_text(
            yaml.safe_dump(vars(trainer.args), sort_keys=False), encoding="utf-8")
        write_json(REPORT / "baseline_summary.json", summary)

    def on_epoch_start(trainer):
        summary["actual_epochs"] = int(trainer.epoch) + 1

    def on_fit_epoch_end(trainer):
        if int(trainer.epoch) + 1 != summary.get("actual_epochs"):
            return  # Ultralytics final best-model validation uses an extra callback step.
        numeric_metrics = {k: float(v) for k, v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in numeric_metrics.values()):
            raise RuntimeError("Non-finite training/validation metrics")
        summary["latest_epoch_metrics"] = numeric_metrics
        summary["elapsed_training_seconds"] = time.perf_counter() - started
        write_json(REPORT / "baseline_summary.json", summary)
        print("BASELINE_PROGRESS=" + json.dumps({
            "epoch": summary["actual_epochs"], "best_epoch": summary.get("best_epoch"),
            "metrics": numeric_metrics,
        }), flush=True)

    def on_model_save(trainer):
        if trainer.best_fitness == trainer.fitness:
            summary["best_epoch"] = int(trainer.epoch) + 1

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_model_save", on_model_save)
    print("BASELINE_TRAIN_ARGS=" + json.dumps(settings), flush=True)
    started = time.perf_counter()
    model.train(**settings)
    summary["training_time_seconds"] = time.perf_counter() - started
    if Path(model.trainer.save_dir).resolve() != RUN.resolve():
        raise RuntimeError("Unexpected training run path")
    best, last = RUN / "weights/best.pt", RUN / "weights/last.pt"
    for path in (best, last, RUN / "results.csv", RUN / "args.yaml"):
        if not path.is_file():
            raise RuntimeError(f"Missing training output: {path}")
    if not summary.get("best_epoch"):
        raise RuntimeError("Best checkpoint epoch not recorded")
    summary.update(status="EVALUATING", best_pt_path=str(best), last_pt_path=str(last))
    write_json(REPORT / "baseline_summary.json", summary)

    best_model = YOLO(str(best), task="detect")
    # Standalone evaluation only; no threshold selection or test-driven training.
    for split in ("val", "test"):
        def on_val_start(validator, split=split):
            summary["loader_validation"][split + "_best"] = check_loader(validator.dataloader.dataset, split)
        best_model.reset_callbacks()
        best_model.add_callback("on_val_start", on_val_start)
        metrics = best_model.val(
            data=str(DATA / "data.yaml"), split=split, imgsz=640,
            batch=8, device=device, workers=0, plots=True,
            augment=False, project=str(RUN), name=split + "_best",
            exist_ok=False, verbose=True,
        )
        summary[split] = evaluation_result(metrics, split)
        write_json(REPORT / "baseline_summary.json", summary)

    if immutable_dataset() != before or digest(manifest) != manifest_hash:
        raise RuntimeError("Dataset image/label content or fixed assignment changed")
    summary.update(
        status="COMPLETED", training_success=True, error=None,
        dataset_images_labels_unchanged=True, fixed_split_unchanged=True,
        test_evaluation_completed=True, nan_detected=False, loader_exclusions=0,
        ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    summary["artifacts"] = sorted(str(p.relative_to(ROOT)) for p in RUN.rglob("*") if p.is_file())
    log = (REPORT / "training.log").read_text(encoding="utf-8", errors="replace")
    summary["warnings"] = sorted(set(line.strip() for line in log.splitlines()
                                    if "WARNING" in line or "warning:" in line.lower()))
    write_json(REPORT / "baseline_summary.json", summary)
    lines = [
        "# YOLOv8n real-only fixed baseline", "",
        f"Model: yolov8n.pt. Epochs: {summary['actual_epochs']}/100. Best epoch: {summary['best_epoch']}.",
        f"imgsz=640, batch=8, device={device}, workers=0, seed=42, deterministic=True, patience=20.",
        "Pretrained=True. Installed Ultralytics optimizer/lr/weight-decay defaults unchanged.",
        "Train/Val/Test: 505/62/62. Source image/label bytes and split assignments unchanged.",
        "Explicitly disabled augmentations: " + ", ".join(augmentation_off),
        "auto_augment=None; augment=False; no custom augmentation/oversampling/synthetic/class weighting.",
        "Albumentations is not installed. Standard input resizing/letterboxing is not data augmentation.", "",
        "| Split | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        v = summary[split]["overall"]
        lines.append("| " + split + " | " + " | ".join(f"{v[k]:.6f}"
                     for k in ("precision", "recall", "map50", "map50_95")) + " |")
    lines += ["", "| Test class | GT instances | Precision | Recall | AP50 | AP50-95 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ("p", "o", "s", "sc"):
        v = summary["test"]["per_class"][name]
        lines.append(f"| {name} | {v['instances']} | " + " | ".join(
            "N/A" if v[k] is None else f"{v[k]:.6f}"
            for k in ("precision", "recall", "ap50", "ap50_95")) + " |")
    lines += ["", "Val open: no GT, not evaluable. Test open/short: one instance each; numeric results only.",
              f"Best checkpoint: {best}",
              f"Training time: {summary['training_time_seconds'] / 60:.2f} minutes.",
              "Training errors: none. NaN: none. Loader exclusions: 0.",
              "Warnings: " + ("; ".join(summary["warnings"]) or "none"),
              "Execution: .venv/Scripts/python.exe -B -u scripts/train_yolov8_detection_baseline_v1.py", ""]
    (REPORT / "baseline_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("BASELINE_COMPLETE=" + json.dumps({
        k: summary[k] for k in ("training_success", "actual_epochs", "best_epoch", "val", "test",
                               "best_pt_path", "warnings")
    }), flush=True)


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Existing baseline run/report must never be overwritten")
    if not WEIGHTS.is_file():
        raise FileNotFoundError("Provide the official local yolov8n.pt weights")
    RUN.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    runtime = RUN / "runtime_config"
    runtime.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    summary = {
        "model": "yolov8n.pt", "requested_epochs": 100, "actual_epochs": 0,
        "best_epoch": None, "imgsz": 640, "seed": 42, "status": "PREPARING",
        "image_counts": COUNTS, "training_success": False, "error": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (REPORT / "training.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run_baseline(summary)
            except BaseException as error:
                summary.update(status="FAILED", training_success=False,
                               error=f"{type(error).__name__}: {error}",
                               ended_utc=datetime.now(timezone.utc).isoformat())
                write_json(REPORT / "baseline_summary.json", summary)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()

