"""Model-independent MLflow logger: calls are additive to existing training code."""
from __future__ import annotations
import hashlib
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import mlflow
if __package__:
    from .mlflow_config import RunInfo, configure_mlflow
else:
    from mlflow_config import RunInfo, configure_mlflow


def _scalar(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, int, float)):
        return value
    return None


def log_params(params: Mapping) -> None:
    values = {str(k): v for k, value in params.items() if (v := _scalar(value)) is not None}
    if values:
        mlflow.log_params(values)


def log_metrics(metrics: Mapping, epoch: int, *, prefix: str = "") -> None:
    """Log only metrics your original code actually computed."""
    clean = {}
    for name, raw in metrics.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            key = f"{prefix}/{name}" if prefix else str(name)
            clean[key] = value
    if clean:
        mlflow.log_metrics(clean, step=int(epoch))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_file(path: str | Path, *, artifact_path: str = "results", required: bool = False) -> bool:
    """Uploads exact file; never deletes local source. Raises on upload failure."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        warnings.warn(f"Skipped missing file: {path}")
        return False
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
    return True


def log_checkpoint(path: str | Path, *, artifact_path: str = "checkpoint") -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    # Artifact upload must succeed before marking the checkpoint as uploaded.
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
    mlflow.set_tags({"checkpoint_sha256": digest, "checkpoint_upload_status": "uploaded_unverified"})
    # Upload success is not the same as independently downloading and checking SHA-256.
    return digest


def log_test_metrics(metrics: Mapping, *, split_id: str) -> None:
    """Call ONLY after existing evaluation code really evaluated the test split."""
    if not split_id or split_id == "unverified":
        raise ValueError("A verified test split ID is required")
    clean = {}
    for name, raw in metrics.items():
        try:
            v = float(raw)
        except (ValueError, TypeError):
            continue
        if math.isfinite(v):
            clean[f"test/{name}"] = v
    if not clean:
        raise ValueError("No valid test metrics to log")
    mlflow.log_metrics(clean)
    mlflow.set_tags({"test_split_id": split_id, "test_evaluation_status": "logged_unverified"})


@contextmanager
def tracked_run(info: RunInfo, *, params: Mapping | None = None):
    """One run per invocation; exceptions mark MLflow run FAILED.

    This wrapper does not change training data/model/optimizer/hyperparameters.
    """
    configure_mlflow(info)
    if mlflow.active_run() is not None:
        raise RuntimeError("An MLflow run is already active: prevent nested/double runs")
    with mlflow.start_run(run_name=info.run_name, tags=info.tags) as run:
        if params:
            log_params(params)
        yield run
