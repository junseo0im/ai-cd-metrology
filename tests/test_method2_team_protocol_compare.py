from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_method2_team_protocol_compare import (
    MISSING_SLOT_MESSAGE,
    continuity_preserving_selection,
    recover_native_pixels,
)


def test_recover_native_pixels_removes_declared_2x_repetition() -> None:
    native = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
    enlarged = np.repeat(np.repeat(native, 2, axis=0), 2, axis=1)

    recovery = recover_native_pixels(enlarged, 2)

    np.testing.assert_array_equal(recovery.image, native)
    assert recovery.input_shape == (6, 8)
    assert recovery.native_shape == (3, 4)
    assert recovery.row_pair_consistency == 1.0
    assert recovery.column_pair_consistency == 1.0


def test_recover_native_pixels_rejects_false_2x_declaration() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)

    with pytest.raises(ValueError, match="not supported"):
        recover_native_pixels(image, 2)


def _candidate(center_x: float, physical_index: int) -> dict[str, object]:
    return {
        "measurement_type": "line",
        "left_x_px": center_x - 4.0,
        "right_x_px": center_x + 4.0,
        "center_x_px": center_x,
        "width_px": 8.0,
        "brightness": 100.0,
        "status": "ok",
        "physical_index": physical_index,
    }


def test_continuity_selection_keeps_missing_physical_slot() -> None:
    reference = [_candidate(float(x), index) for index, x in enumerate(range(10, 100, 10))]
    current = [
        _candidate(float(x), index)
        for index, x in enumerate(range(10, 100, 10))
        if x != 70
    ]

    selected = continuity_preserving_selection(current, reference, "line")

    assert [item["target_id"] for item in selected] == ["L1", "L2", "L3", "L4", "L5"]
    assert [item["center_x_px"] for item in selected[:4]] == [30.0, 40.0, 50.0, 60.0]
    assert selected[4]["status"] == "error"
    assert selected[4]["center_x_px"] is None
    assert selected[4]["width_px"] is None
    assert selected[4]["selection_message"] == MISSING_SLOT_MESSAGE
    assert all(item["center_x_px"] != 80.0 for item in selected if item["center_x_px"] is not None)


def test_continuity_selection_preserves_ids_across_small_shifts() -> None:
    reference = [_candidate(float(x), index) for index, x in enumerate(range(10, 100, 10))]
    shifted = [_candidate(float(x + 1), index) for index, x in enumerate(range(10, 100, 10))]

    selected = continuity_preserving_selection(shifted, reference, "line")

    assert [item["target_id"] for item in selected] == ["L1", "L2", "L3", "L4", "L5"]
    assert [item["center_x_px"] for item in selected] == [31.0, 41.0, 51.0, 61.0, 71.0]
    assert all(item["status"] == "ok" for item in selected)
