"""Optional Ultralytics callback: add logging without replacing model.train().

NOTE: Ultralytics own MLflow integration must be OFF, to avoid duplicate Runs.
Attach callback before calling the ORIGINAL model.train(...) with ORIGINAL args.
"""
from __future__ import annotations
import math
import warnings
from typing import Any

if __package__:
    from .experiment_logger import log_metrics
else:
    from experiment_logger import log_metrics


def attach_yolo_epoch_logger(model: Any, *, strict_logging: bool = False) -> None:
    """Do not modify training settings; only reads trainer's already-computed results."""
    seen_epochs: set[int] = set()

    def on_fit_epoch_end(trainer):
        step = int(trainer.epoch) + 1  # 1-indexed human-readable epoch
        if step in seen_epochs:
            return  # The callback may run again for final best-model validation.
        seen_epochs.add(step)
        raw = getattr(trainer, "metrics", {}) or {}
        if not isinstance(raw, dict):
            return
        # Preserve original Ultralytics metric key names. No new evaluation.
        clean = {}
        for key, value in raw.items():
            try:
                num = float(value)
                if math.isfinite(num):
                    clean[str(key)] = num
            except (TypeError, ValueError):
                continue
        if clean:
            try:
                log_metrics(clean, step, prefix="val_yolo_raw")
            except Exception as exc:
                if strict_logging:
                    raise
                warnings.warn(f"MLflow epoch logging failed at epoch {step}: {exc}")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
