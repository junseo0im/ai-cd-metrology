"""TASK 18-B-13A.2 / 18-B-18: shared exposure display unit (s), numeric condition order and X titles (display only)."""
import unittest

import pandas as pd

from dashboard.units import EXPOSURE_DISPLAY_UNIT, exposure_display, stored_unit_text
from dashboard.process_analysis import exposure_label, metadata_frame
from dashboard.components.sqlite_input import exposure_text, condition_accumulation, condition_batches
from dashboard.components.inspection_wafer import exposure_label as ia_label
from dashboard.components.analysis_management import condition_label, image_condition
from dashboard.components.process_analysis import _sorted_conditions, _x_title


class UnitDisplayTests(unittest.TestCase):
    def test_01_rule(self):
        # TASK18-B-18: exposure 50/60/70/80 are times in seconds; legacy 'mJ/cm²' text is the same condition.
        self.assertEqual(EXPOSURE_DISPLAY_UNIT, "s")
        self.assertEqual(exposure_display(50, None), "50 s")
        self.assertEqual(exposure_display(62.5, float("nan")), "62.5 s")
        self.assertEqual(exposure_display(50, ""), "50 s")
        for legacy in ("mJ/cm²", "mJ/cm2", "mJ"):
            self.assertEqual(exposure_display(50, legacy), "50 s")
        self.assertEqual(exposure_display(50, "min"), "50 min")                 # Other recorded unit kept as stored.
        self.assertEqual(exposure_display(None, None), "미확인")
        self.assertEqual(stored_unit_text(None), "DB 미기록 · 표시 기준 s")
        self.assertEqual(stored_unit_text("mJ/cm²"), "mJ/cm² (표시 기준 s)")
        # Every screen uses the same rule.
        for label in (exposure_label(50, None), ia_label(50, None), exposure_text(50, None)):
            self.assertEqual(label, "50 s")
        self.assertEqual(condition_label(image_condition(50, None, 220)), "노광 50 s · PEB 220 s")
        self.assertEqual(condition_label(image_condition(50, "mJ/cm²", None)), "노광 50 s · PEB 미입력")

    def test_02_legacy_unit_text_groups_with_unrecorded_and_other_units_stay_apart(self):
        rows = pd.DataFrame([
            {"batch_id": "A", "wafer_id": "W1", "exposure_value": 50.0, "exposure_unit": None, "peb_temperature_c": 220.0, "image_count": 3},
            {"batch_id": "B", "wafer_id": "W1", "exposure_value": 50.0, "exposure_unit": "mJ/cm²", "peb_temperature_c": 220.0, "image_count": 2},
            {"batch_id": "C", "wafer_id": "W1", "exposure_value": 50.0, "exposure_unit": "min", "peb_temperature_c": 220.0, "image_count": 1}])
        table = condition_accumulation(rows, {})
        self.assertEqual(sorted(table["노광"]), ["50 min", "50 s"])
        self.assertEqual(table.set_index("노광").loc["50 s", "등록 이미지 수"], 5)      # Old Batch B joins 50 s.
        self.assertEqual(sorted(condition_batches(rows, "W1", "50 s", "220")["batch_id"]), ["A", "B"])
        self.assertEqual(_x_title(rows, "exposure"), "노광 조건 (조건별 단위)")
        self.assertEqual(_x_title(rows.iloc[:2], "exposure"), "노광 조건 (s)")
        self.assertEqual(_x_title(rows, "peb"), "PEB 조건 (s)")

    def test_03_numeric_ascending_with_new_condition(self):
        images = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "exposure_value": [70.0, 62.5, 50.0, None],
                               "exposure_unit": [None] * 4, "peb_temperature_c": [220.0, None, 120.0, 220.0]})
        subset = metadata_frame(images)
        valid = pd.DataFrame({"공정 조건": list(subset["exposure_condition"])})
        self.assertEqual(list(_sorted_conditions(valid, subset, "exposure")["공정 조건"]),
                         ["50 s", "62.5 s", "70 s", "노광 미입력"])           # Unrecorded value last.
        valid = pd.DataFrame({"공정 조건": list(dict.fromkeys(subset["peb_condition"]))})
        self.assertEqual(list(_sorted_conditions(valid, subset, "peb")["공정 조건"]), ["120 s", "220 s", "PEB 미입력"])


if __name__ == "__main__":
    unittest.main()
