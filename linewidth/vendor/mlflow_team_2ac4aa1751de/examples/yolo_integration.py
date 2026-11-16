"""REFERENCE ONLY. Insert marked sections into your ORIGINAL YOLO training code.
Do not run this example as-is: dataset paths and training variables are placeholders.
"""
from ultralytics import YOLO, settings
from mlflow_config import RunInfo
from experiment_logger import tracked_run, log_file, log_checkpoint
from yolo_adapter import attach_yolo_epoch_logger

# Disable Ultralytics built-in MLflow integration to prevent duplicate Runs.
# This is a logging option, not a hyperparameter or dataset change.
settings.update({"mlflow": False})

info = RunInfo(
    owner="REPLACE_WITH_DAGSHUB_USERNAME",
    environment="kaggle",  # or local / colab
    model_name="yolov8n",
    dataset_version="REPLACE_WITH_ACTUAL_DATASET_VERSION",
    seed=42,
    task="yolo",
    val_split_id="unverified",
    test_split_id="unverified",
)

# Keep your ORIGINAL values and model train arguments unchanged.
train_kwargs = dict(data="REPLACE_DATA_YAML", epochs=35, batch=8, imgsz=640, seed=42)
model = YOLO("yolov8n.pt")
attach_yolo_epoch_logger(model)

with tracked_run(info, params={"epochs": train_kwargs["epochs"], "batch": train_kwargs["batch"], "imgsz": train_kwargs["imgsz"], "seed": train_kwargs["seed"]}):
    model.train(**train_kwargs)  # REPLACE with your original training call
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir:
        log_file(save_dir / "results.csv", artifact_path="training_results")
        # Only enable this after checking remote artifact size and access policy:
        # log_checkpoint(save_dir / "weights" / "best.pt")
    # Test metrics: never invent/evaluate automatically.
