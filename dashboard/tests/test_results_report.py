"""I&A 결과 리포트: scope isolation (Wafer / mixed-condition Batch), signature binding and Word content.

A stub provider only; the operational SQLite is never opened.
"""
from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd

from dashboard import report_builder as rb


def image(image_id, batch, wafer, die, exposure, peb, sha=None):
    return {"image_id": image_id, "batch_id": batch, "wafer_id": wafer, "die_code": die, "position_code": "c",
            "exposure_value": exposure, "exposure_unit": None, "peb_temperature_c": peb, "magnification": 50,
            "original_filename": image_id + ".png", "upload_mode": "WAFER", "sha256": sha or image_id,
            "storage_path": None}


def result(image_id, batch, wafer, line, *, outcome="COMPLETE", defects=()):
    return {"image_id": image_id, "batch_id": batch, "wafer_id": wafer, "analysis_run_id": "run-" + image_id,
            "outcome": outcome, "is_adopted": 0, "started_at": "2026-11-20T01:00:00",
            "calibration": {"calibration_id": "cal-1", "um_per_px": .16, "magnification": 50},
            "defect_status": "SUCCESS", "engine_status": "SUCCESS", "metrology_status": "SUCCESS",
            "pd_policy_state": "NO_PD", "partial_measurement": 0, "line_widths_um": [line, line],
            "gap_widths_um": [line + 1], "line_measurement_count": 2, "gap_measurement_count": 1,
            "line_mean_um": line, "gap_mean_um": line + 1, "defect_classes": [d for d in defects],
            "defect_details": [{"class_code": d, "confidence": .9, "source": "MODEL_POSTPROCESSED"} for d in defects],
            "judgments": []}


IMAGES = pd.DataFrame([
    image("A1", "B-MIX", "W2", "11", 70.0, 220.0), image("A2", "B-MIX", "W2", "12", 70.0, 220.0),
    image("B1", "B-MIX", "W4", "11", 60.0, 220.0),
    image("C1", "B-OLD", "W2", "11", 50.0, 220.0)])  # Same Wafer W2, other Batch, other condition.
ANALYSIS = {"A1": result("A1", "B-MIX", "W2", 9.0, defects=("p",)), "A2": result("A2", "B-MIX", "W2", 11.0),
            "B1": result("B1", "B-MIX", "W4", 8.0, defects=("s", "s")), "C1": result("C1", "B-OLD", "W2", 5.0)}


class StubProvider:
    def latest_analysis(self, wafer_id=None):
        return dict(ANALYSIS)

    def wafer_sites(self, wafer_id):
        return []

    def image_record(self, image_id, *, batch_id=None):
        row = IMAGES[IMAGES["image_id"] == image_id]
        return row.iloc[0].to_dict() if len(row) else None


def collect(provider, *, batch_id=None, wafer_id=None, image_id=None):
    frame = IMAGES
    if batch_id:
        frame = frame[frame["batch_id"] == batch_id]
    if wafer_id:
        frame = frame[frame["wafer_id"] == wafer_id]
    return frame.copy()


OPTIONS = {"title": "T", "author": "홍길동", "purpose": "", "notes": "", "defects": True, "wafer_map": True,
           "images": False, "image_id": None}


@patch("dashboard.components.analysis_management.collect_images", collect)
class ResultsReportTests(unittest.TestCase):
    def setUp(self):
        self.provider = StubProvider()

    def test_01_mixed_batch_is_grouped_by_wafer_and_condition_without_other_batches(self):
        scope = rb.scope(self.provider, "Batch", "B-MIX")
        self.assertEqual(set(scope["images"]["image_id"]), {"A1", "A2", "B1"})
        self.assertEqual(set(scope["analysis"]), {"A1", "A2", "B1"})
        report = rb.build(self.provider, scope, OPTIONS)
        groups = {g["label"]: g for g in report["groups"]}
        self.assertEqual(len(groups), 2)
        w2 = next(g for g in report["groups"] if g["wafer"] == "W2")
        self.assertEqual(w2["stats"]["registered"], 2)
        self.assertAlmostEqual(w2["stats"]["line_mean"], 10.0)            # C1 (5.0, other Batch) not mixed in.
        self.assertEqual(w2["stats"]["line_count"], 4)
        self.assertEqual(w2["defects"]["counts"], {"p": 1})
        self.assertEqual(report["scope"]["registered"], 3)
        self.assertEqual([r["그룹"] for r in report["rows"]], [g["label"] for g in report["groups"]])

    def test_02_wafer_condition_filter_keeps_only_that_condition(self):
        whole = rb.scope(self.provider, "Wafer", "W2")
        self.assertEqual(len(whole["groups"]), 2)
        condition = next(c for c in set(whole["images"]["condition"]) if c.startswith("50"))
        scope = rb.scope(self.provider, "Wafer", "W2", condition)
        self.assertEqual(list(scope["images"]["image_id"]), ["C1"])
        report = rb.build(self.provider, scope, OPTIONS)
        self.assertEqual([r["Die"] for r in report["groups"][0]["rows"]], ["11"])
        self.assertAlmostEqual(report["scope"]["line_mean"], 5.0)

    def test_03_signature_changes_with_scope_inputs_and_sections(self):
        scope = rb.scope(self.provider, "Batch", "B-MIX")
        base = rb.signature(scope, OPTIONS)
        self.assertEqual(base, rb.signature(rb.scope(self.provider, "Batch", "B-MIX"), dict(OPTIONS)))
        self.assertNotEqual(base, rb.signature(rb.scope(self.provider, "Wafer", "W2"), OPTIONS))
        self.assertNotEqual(base, rb.signature(scope, {**OPTIONS, "author": "다른 사람"}))
        self.assertNotEqual(base, rb.signature(scope, {**OPTIONS, "wafer_map": False}))

    def test_04_word_has_inputs_only_when_given_checked_sections_and_pictures(self):
        from docx import Document
        report = rb.build(self.provider, rb.scope(self.provider, "Batch", "B-MIX"), {**OPTIONS, "wafer_map": False})
        doc = Document(BytesIO(rb.docx_bytes(report)))
        text = "\n".join(p.text for p in doc.paragraphs)
        cells = [c.text for t in doc.tables for row in t.rows for c in row.cells]
        self.assertIn("홍길동", cells)
        self.assertNotIn("실험 목적", cells)                              # Empty input -> omitted.
        self.assertIn("결함 검출 결과", text)
        self.assertNotIn("Wafer Map", "\n".join(p.text for p in doc.paragraphs if p.style.name.startswith("Heading")))
        self.assertEqual(len(doc.inline_shapes), 1)                       # Measurement chart embedded.
        self.assertNotIn("B-OLD", "".join(cells))
        self.assertTrue(any("10.000" in c for c in cells))

    def test_06_wafer_line_width_stats_per_group_same_in_preview_and_word(self):
        from docx import Document
        from statistics import stdev
        report = rb.build(self.provider, rb.scope(self.provider, "Batch", "B-MIX"), OPTIONS)
        rows = {r[0]: r for r in rb.wafer_stats_rows(report)}
        w2 = next(k for k in rows if k.startswith("W2")); w4 = next(k for k in rows if k.startswith("W4"))
        self.assertEqual(rows[w2][1:], ["10.000", f"{stdev([9.0, 9.0, 11.0, 11.0]):.3f}", "4"])
        self.assertEqual(rows[w4][1:], ["8.000", "0.000", "2"])
        preview = rb.preview_frames(report)["wafer_stats"].values.tolist()
        doc = Document(BytesIO(rb.docx_bytes(report)))
        word = next([[c.text for c in r.cells] for r in t.rows][1:] for t in doc.tables
                    if t.rows[0].cells[0].text == "Wafer · 조건" and t.rows[0].cells[1].text.startswith("Line Width 평균"))
        self.assertEqual(preview, word)
        single = rb.build(self.provider, rb.scope(self.provider, "Wafer", "W2",
                                                  next(c for c in set(rb.scope(self.provider, "Wafer", "W2")["images"]["condition"]) if c.startswith("50"))), OPTIONS)
        self.assertEqual(rb.wafer_stats_rows(single)[0][1:], ["5.000", "0.000", "2"])   # N=2 -> std shown.

    def test_05_navigation_marks_report_as_sqlite(self):
        from dashboard.navigation import page_data_source
        self.assertEqual(page_data_source("results_report", {}), "SQLite")


if __name__ == "__main__":
    unittest.main()
