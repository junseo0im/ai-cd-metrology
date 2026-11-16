"""Central project paths used by the line-width pipeline.

Set LINEWIDTH_UNET_CHECKPOINT to use another checkpoint without source edits.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
TRAINING_ROOT = ANALYSIS_ROOT / "training"
TRAINING_SOURCE_ROOT = TRAINING_ROOT / "source"
UNET_DATA_ROOT = TRAINING_ROOT / "unet_data"
UNET_DATASET_ROOT = TRAINING_ROOT / "unet_dataset"
MANUAL_REVIEW_ROOT = TRAINING_ROOT / "manual_reviews"
TESTING_ROOT = ANALYSIS_ROOT / "testing"
MODELS_ROOT = ANALYSIS_ROOT / "models"
UNET_METHOD_DIR = Path(__file__).resolve().parent / "methods" / "unet"
BASELINE_MODEL_DIR = MODELS_ROOT / "baseline"
EVALUATIONS_ROOT = ANALYSIS_ROOT / "evaluations"
OUTPUTS_ROOT = ANALYSIS_ROOT / "outputs"

_configured_checkpoint = os.environ.get("LINEWIDTH_UNET_CHECKPOINT")
ACTIVE_UNET_CHECKPOINT = (
    Path(_configured_checkpoint).expanduser()
    if _configured_checkpoint
    else UNET_METHOD_DIR / "models" / "best_unet_manual.pt"
)
