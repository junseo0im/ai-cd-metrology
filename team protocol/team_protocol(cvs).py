"""Team output protocol.

Replace this module with the team's official implementation if needed.  The
measurement program calls save_prediction_csv(predictions, output_path).
"""
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "image_name", "measurement_type", "target_id", "mean_um", "std_um",
    "valid_points", "status", "native_width", "native_height",
    "duplicate_factor_x", "duplicate_factor_y", "processing_time_s",
    "timing_included",
]


def save_prediction_csv(predictions, output_path):
    """Validate the canonical prediction table and save UTF-8 CSV."""
    frame = predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction columns missing: {missing}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
