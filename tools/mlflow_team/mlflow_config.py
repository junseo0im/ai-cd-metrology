"""Shared DagsHub MLflow tracking configuration. No secrets in source."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

REPO_OWNER = "mnitocea"
REPO_NAME = "IDA-Defect-AI-MLflow"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
EXPERIMENTS = {
    "yolo": "IDA_Defect_YOLO",
    "classification": "IDA_Defect_Classification",
    # Add width experiment after the team agrees on its name.
}


def _required(value: str, label: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} must not be blank")
    return value


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", _required(value, "name")).strip("-")


@dataclass(frozen=True)
class RunInfo:
    owner: str
    environment: str
    model_name: str
    dataset_version: str
    seed: int
    task: str
    val_split_id: str = "unverified"
    test_split_id: str = "unverified"
    run_suffix: str = ""

    @property
    def experiment(self) -> str:
        if self.task not in EXPERIMENTS:
            raise ValueError(f"Unknown task '{self.task}'. Allowed: {list(EXPERIMENTS)}")
        return EXPERIMENTS[self.task]

    @property
    def run_name(self) -> str:
        # MLflow adds its own unique run_id even with repeated run_name.
        pieces = [self.model_name, self.dataset_version, f"seed{self.seed}", self.owner]
        if self.run_suffix:
            pieces.append(self.run_suffix)
        return "_".join(slug(part) for part in pieces)

    @property
    def tags(self) -> dict[str, str]:
        return {
            "owner": _required(self.owner, "owner"),
            "environment": _required(self.environment, "environment"),
            "model_name": _required(self.model_name, "model_name"),
            "dataset_version": _required(self.dataset_version, "dataset_version"),
            "val_split_id": self.val_split_id,
            "test_split_id": self.test_split_id,
            "task": self.task,
            "logging_version": "v0.2",
            "test_evaluation_status": "not_logged",
        }


def configure_mlflow(info: RunInfo) -> None:
    """Connect to shared tracking server, using each member's own credentials.

    Authenticate locally with `dagshub login`, or provide a personal token through
    DagsHub's supported environment settings. Never commit credentials.
    """
    import dagshub
    import mlflow
    # Explicit URI is helpful when working alongside unrelated local experiments.
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(info.experiment)
