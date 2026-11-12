"""Canonical, isolated YOLO segmentation training; never overwrite an existing run."""

from __future__ import annotations

import argparse
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
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["open", "short", "pattern_narrowing", "pattern_thickening", "deformation", "particle", "scratch"]


def resolve(value: str, parent: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def stage_dataset(source: Path, destination: Path) -> tuple[Path, list[dict]]:
    """Copy unchanged TRAIN/VAL bytes so Ultralytics caches cannot alter inputs.

    Relative dataset paths use data.yaml's directory (or its explicit path).
    TEST is never resolved, copied or evaluated. Empty labels must be explicit;
    their existence is not a human NORMAL judgment or a HOLD handling policy.
    """
    from PIL import Image
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths

    spec = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict):
        raise ValueError("data.yaml must contain a mapping")
    if spec.get("names") not in (CLASSES, dict(enumerate(CLASSES))) or spec.get("nc", 7) != 7:
        raise ValueError("data.yaml must use the canonical seven classes in ID order 0..6")
    base = resolve(spec.get("path", str(source.parent)), source.parent)
    manifest, seen_hashes = [], set()
    for split in ("train", "val"):
        entries = spec.get(split)
        if not entries:
            raise ValueError(f"data.yaml is missing {split}")
        images = []
        for entry in entries if isinstance(entries, list) else [entries]:
            item = resolve(entry, base)
            if item.is_dir():
                images.extend(p.resolve() for p in item.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
            elif item.is_file() and item.suffix.lower() == ".txt":
                images.extend(resolve(line.strip(), item.parent) for line in item.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#"))
            else:
                raise ValueError(f"Expected an image directory or image-list TXT: {item}")
        if not images:
            raise ValueError(f"No images in {split}")
        filenames = set()
        for image in sorted(images):
            label = Path(img2label_paths([str(image)])[0])
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing image/label pair: {image} / {label}; no implicit empty labels")
            key = image.stem.casefold()
            image_hash = sha256(image)
            if key in filenames or image_hash in seen_hashes:
                raise ValueError(f"Filename collision or duplicate/train-val hash overlap: {image}")
            filenames.add(key)
            seen_hashes.add(image_hash)
            with Image.open(image) as opened:
                opened.verify()
            for row, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 7 or (len(parts) - 1) % 2 or parts[0] not in {str(i) for i in range(7)}:
                    raise ValueError(f"Malformed segmentation label: {label}:{row}")
                if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in parts[1:]):
                    raise ValueError(f"Invalid normalized coordinates: {label}:{row}")
            record = {"split": split, "image": str(image), "label": str(label), "image_sha256": image_hash, "label_sha256": sha256(label)}
            for kind, original in (("images", image), ("labels", label)):
                copied = destination / kind / split / original.name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
                if sha256(copied) != record["image_sha256" if kind == "images" else "label_sha256"]:
                    raise OSError(f"Dataset copy hash mismatch: {copied}")
            manifest.append(record)
    staged_yaml = destination / "data.yaml"
    staged_yaml.write_text(yaml.safe_dump({"path": str(destination), "train": "images/train", "val": "images/val", "nc": 7, "names": dict(enumerate(CLASSES))}, sort_keys=False), encoding="utf-8")
    return staged_yaml, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yolov8_seg_baseline.yaml")
    parser.add_argument("--data", help="Dataset YAML; overrides config data")
    parser.add_argument("--project", required=True, help="Output parent directory")
    parser.add_argument("--name", required=True, help="New run directory name (must not exist)")
    parser.add_argument("--epochs", type=int, help="Explicit epoch override, e.g. 1 for smoke")
    cli = parser.parse_args()
    config_path = resolve(cli.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Training config must contain a mapping")
    expected_environment = config.pop("baseline_environment", {})
    data_value = cli.data or config.pop("data", None)
    config.pop("data", None)
    if not data_value:
        raise ValueError("Provide --data (or data in config)")
    if config.get("task") != "segment":
        raise ValueError("task must be segment")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("A fixed non-negative integer seed is required")
    if any(key in config for key in ("project", "name", "exist_ok", "save_dir", "resume")):
        raise ValueError("Use --project/--name; existing-run reuse and resume are not supported")
    if cli.epochs is not None:
        config["epochs"] = cli.epochs
    if type(config.get("epochs")) is not int or config["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    model_path = resolve(config.pop("model"))
    data_path = resolve(data_value)
    if not model_path.is_file() or model_path.suffix.lower() != ".pt" or not data_path.is_file():
        raise FileNotFoundError("Provide an existing local .pt model and dataset YAML; no implicit downloads")
    if not cli.name or cli.name in (".", "..") or "/" in cli.name or "\\" in cli.name or ":" in cli.name:
        raise ValueError("--name must be one directory name")
    run = resolve(cli.project) / cli.name
    dataset_spec = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    if not isinstance(dataset_spec, dict):
        raise ValueError("data.yaml must contain a mapping")
    source_root = resolve(dataset_spec.get("path", str(data_path.parent)), data_path.parent)
    if run.is_relative_to(ROOT / "data/raw") or run.is_relative_to(source_root):
        raise ValueError("Output must be outside raw data and the source dataset root")
    run.mkdir(parents=True, exist_ok=False)  # Exclusive ownership; never reuse any existing run.
    metadata = {"status": "PREPARING", "started_utc": datetime.now(timezone.utc).isoformat(), "argv": [sys.executable, *sys.argv], "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "cwd": str(Path.cwd()), "config": str(config_path), "config_sha256": sha256(config_path), "source_data_yaml": str(data_path), "source_data_sha256": sha256(data_path), "model": str(model_path), "model_sha256": sha256(model_path), "baseline_environment": expected_environment, "validation_notice": "PROVISIONAL / NOT FINAL VALIDATION. A val entry does not establish official VAL status. Original B01 used Pilot36 from original TRAIN / guideline development."}
    model = None
    try:
        shutil.copy2(config_path, run / "config_snapshot.yaml")
        runtime = run / "runtime_config"
        runtime.mkdir()
        os.environ["YOLO_CONFIG_DIR"] = str(runtime)
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import USER_CONFIG_DIR

        # Reuse the installed Windows font locally, avoiding a font download.
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
        if font.is_file() and not (USER_CONFIG_DIR / "Arial.ttf").exists():
            shutil.copy2(font, USER_CONFIG_DIR / "Arial.ttf")
        metadata["environment"] = {"python": platform.python_version(), "executable": sys.executable, "ultralytics": ultralytics.__version__, "torch": torch.__version__, "requested_device": config["device"], "cuda_available": torch.cuda.is_available()}
        metadata["environment_differences"] = {key: {"baseline": value, "current": metadata["environment"].get(key)} for key, value in expected_environment.items() if metadata["environment"].get(key) != value}
        if metadata["environment_differences"]:
            print("WARNING: environment differs from B01:", metadata["environment_differences"], flush=True)
        staged, manifest = stage_dataset(data_path, run / "dataset")
        save_json(run / "dataset_manifest.json", manifest)
        metadata["dataset_counts"] = {split: sum(r["split"] == split for r in manifest) for split in ("train", "val")}
        arguments = dict(config, data=str(staged), project=str(run.parent), name=run.name, exist_ok=True)
        # exist_ok=True applies only to our exclusively-created NEW folder,
        # allowing metadata/dataset staging before the trainer creates its files.
        save_json(run / "requested_train_args.json", dict(arguments, model=str(model_path)))
        print("TRAINING_ARGUMENTS=" + json.dumps(dict(arguments, model=str(model_path))), flush=True)
        print("RUN_METADATA=" + json.dumps(metadata), flush=True)
        model = YOLO(str(model_path), task="segment")
        if model.task != "segment":
            raise ValueError(f"The loaded model is {model.task}, not segment")
        metadata["status"] = "TRAINING"
        save_json(run / "run_metadata.json", metadata)
        started = perf_counter()
        model.train(**arguments)
        metadata["training_elapsed_seconds"] = perf_counter() - started
        metadata["actual_device"] = str(model.trainer.device)
        if Path(model.trainer.save_dir).resolve() != run.resolve():
            raise RuntimeError("Trainer output directory differs from the reserved run")
        for record in manifest:
            if sha256(Path(record["image"])) != record["image_sha256"] or sha256(Path(record["label"])) != record["label_sha256"]:
                raise RuntimeError("Source dataset changed during training")
        if sha256(data_path) != metadata["source_data_sha256"]:
            raise RuntimeError("Source data.yaml changed during training")
        outputs = {"best.pt": run / "weights/best.pt", "last.pt": run / "weights/last.pt", "results.csv": run / "results.csv"}
        if not all(path.is_file() for path in outputs.values()):
            raise RuntimeError("Training returned without all expected checkpoints/results")
        metadata.update(status="COMPLETED", source_dataset_unchanged=True, outputs={key: str(value) for key, value in outputs.items()})
        print("TRAINING_COMPLETE=" + json.dumps(metadata["outputs"]), flush=True)
    except Exception as error:
        metadata.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if model is not None and getattr(model, "trainer", None) is not None:
            save_json(run / "effective_train_args.json", vars(model.trainer.args))
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(run / "run_metadata.json", metadata)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as error:
        print(f"TRAINING FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
