# Canonical YOLOv8-seg training

## Historical B01 result

The original B01 run used a temporary Python script, now deleted. Its exact source
and original shell command were not preserved. The source of truth is
`data/derived/yolo_quick_baseline_v1/runs/train_v2/args.yaml` and
`data/derived/yolo_quick_baseline_v1/runs/training_console.log`.
The original `weights/best.pt`, `weights/last.pt` and `results.csv` are preserved.
The new script reconstructs the recorded configuration; it is not a recovered copy
of the old source and does not claim bit-identical results.

B01 used pretrained YOLOv8n-seg, segment, 640 px, 30 epochs, batch 4, workers 0,
seed 42, deterministic=True, CPU, cache=False, and the default train augmentation.
Environment: Python 3.11.9, Ultralytics 8.4.174, torch 2.14.1+cpu.
The log shows optimizer=auto selected AdamW; do not replace auto with a tuned
optimizer. `augment=False` is the validation/inference augmentation option, not
a statement that training mosaic/HSV/flip augmentation was disabled.

## Future experiments

Use `scripts/train_yolov8_seg.py` and `configs/yolov8_seg_baseline.yaml` as the
canonical entry point. It uses the existing project .venv and installs nothing.
Only existing local `.pt` weights are accepted. Change `model` in the config when
the local pretrained file has a different location. The default is the original
pretrained checkpoint, NOT B01 best.pt or last.pt.

PowerShell, from the repository root (example only, not an instruction to run now):

```powershell
& .\.venv\Scripts\python.exe scripts/train_yolov8_seg.py --config configs/yolov8_seg_baseline.yaml --data data/derived/yolo_quick_baseline_v1/data.yaml --project data/derived/yolo_experiments --name b01_reproduction
```

One-epoch smoke command:

```powershell
& .\.venv\Scripts\python.exe scripts/train_yolov8_seg.py --config configs/yolov8_seg_baseline.yaml --data data/derived/yolo_quick_baseline_v1/data.yaml --project data/derived/yolo_experiments --name smoke --epochs 1
```

Never reuse a run name: an existing output directory causes an error, without
deleting, overwriting or silently incrementing it. Config/model/CLI paths are
repository-relative unless absolute. Dataset `path` is relative to data.yaml;
train/val entries are relative to that dataset root. Image-list TXT entries are
relative to the TXT's directory. Both image directories and TXT lists are accepted.
Canonical class order must remain IDs 0..6: open, short, pattern_narrowing,
pattern_thickening, deformation, particle, scratch.

TRAIN and validation image/label files are copied byte-for-byte into the new run.
Image hashes must be unique within/across these two inputs. Labels must exist,
including explicit empty TXT files where appropriate; no missing label is
silently converted to NORMAL. Basic image/label/config checks do not replace
annotation geometry, completeness, human HOLD, or split-group QA.
Ultralytics caches are written only beside the copies. TEST is never resolved,
copied or evaluated, even if the input YAML has a test entry. No original export,
raw image, split or source dataset is changed. Originals are hash-checked after
training. This isolation costs one extra dataset copy per run.

The trainer receives exist_ok=True only for the NEW directory that the script
created exclusively. This output-management difference from historical
exist_ok=False permits staging and metadata before training; existing runs still
cannot be reused. Data/project/name paths also change. Default 30-epoch training
settings are otherwise reconstructed, not tuned. Smoke overrides only epochs=1.
Version differences are recorded as warnings; the full effective arguments are
saved so defaults cannot silently go undocumented.

## Outputs and validity boundary

```text
<project>/<name>/
  config_snapshot.yaml
  requested_train_args.json
  effective_train_args.json
  run_metadata.json
  dataset_manifest.json
  dataset/{data.yaml,images/{train,val}/,labels/{train,val}/}
  runtime_config/
  args.yaml
  results.csv
  weights/{best.pt,last.pt}
  ... Ultralytics plots / batch previews
```

Metadata includes the executable, argv and command, cwd, Python/Ultralytics/torch,
device, input/config/model hashes, original-to-copy mapping, timestamps and
COMPLETED/FAILED state. Training prints the arguments and final checkpoint/CSV
paths. This is a training entry point, not a replacement for the previous
custom report/example generation or its separate best.pt evaluation.

**PROVISIONAL / NOT FINAL VALIDATION:** historical Pilot36 belongs to original
TRAIN and guideline development, not official VAL. A YAML field named `val`
does not establish official validation status. Before preparing B01+B02 or later
experiments, agree on treatment of HOLD/unannotated ambiguous regions. Keeping
HOLD in a CSV does not make ordinary YOLO training ignore those regions. This
script does not choose a HOLD/exclusion policy, change annotations or redefine
the split. No B01+B02 full retraining, TEST use or hyperparameter search is
authorized by creating this entry point.
