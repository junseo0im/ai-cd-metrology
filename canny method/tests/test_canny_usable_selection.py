from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_canny_usable_selection import (  # noqa: E402
    _local_retry_candidate,
    select_track,
    summarize_tracks,
    track_sidewalls,
)


def candidate(center: float, width: float) -> dict[str, object]:
    return {
        "left_x_px": center - width / 2,
        "right_x_px": center + width / 2,
        "center_x_px": center,
        "width_px": width,
    }


def test_tracks_filter_and_select_median() -> None:
    rows = []
    for row_index in range(5):
        items = [candidate(20 + row_index * 0.1, 5), candidate(60, 6), candidate(100, 7)]
        if row_index == 2:
            items.pop(1)
        rows.append(
            {
                "row_index": row_index,
                "y_fraction": row_index / 4,
                "y_native": row_index * 10,
                "candidates": items,
                "gradient_row": np.zeros(120),
            }
        )
    tracks, _ = track_sidewalls(rows)
    summaries = summarize_tracks(tracks, expected_rows=5, min_valid_rows=4, max_deviation_um=0.1)
    selected = select_track(summaries, "2")

    assert len(summaries) == 3
    assert all(item["usable"] for item in summaries)
    assert selected is not None
    assert selected["mean_width_px_native"] == 6.0


def test_local_retry_uses_signed_edges() -> None:
    gradient = np.zeros(80, dtype=float)
    gradient[37] = -10
    gradient[43] = 9

    result = _local_retry_candidate(gradient, expected_center=40, expected_width=6)

    assert result is not None
    assert result["left_x_px"] == 37
    assert result["right_x_px"] == 43
    assert result["width_px"] == 6
