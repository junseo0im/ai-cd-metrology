"""TASK 18-B-15: PD 사용 불가 (red) vs. plain 미평가 (grey) on the I&A Wafer Map; hover and Die summary.

Synthetic stored results only (no DB). The v0.1 score and PN/PT bounds are not changed by this rule.
"""
import unittest

from dashboard.components.ia_analysis_result import die_summary, image_status, image_summary
from dashboard.components.inspection_wafer import apply_judgment_view, die_hover_lines, die_snapshot

WAFER = "W9"


def item(image_id, *, outcome="COMPLETE", widths=(10.0, 10.0), gaps=(9.0,), defects=(), heuristic=(), pn=0,
         partial=0, metrology="SUCCESS"):
    measured = metrology == "SUCCESS" and outcome not in ("MODEL_PD_EXCLUDED", "POLICY_REVIEW_REQUIRED")
    return {"image_id": image_id, "outcome": outcome, "defect_status": "SUCCESS", "engine_status": "SUCCESS",
            "metrology_status": metrology if measured else "SKIPPED", "partial_measurement": partial,
            "line_widths_um": list(widths) if measured else [], "gap_widths_um": list(gaps) if measured else [],
            "configuration": {"design_width_um": 10.0}, "is_adopted": 0, "started_at": "2026-11-20T00:00:00",
            "judgments": [{"label": "pn"}] * pn + [{"label": "normal"}],
            "defect_classes": list(defects) + list(heuristic),
            "defect_details": [{"class_code": c, "source": "MODEL_POSTPROCESSED"} for c in defects]
                              + [{"class_code": c, "source": "HEURISTIC"} for c in heuristic],
            "line_mean_um": None, "gap_mean_um": None}


def dies(spec):
    """spec: {die_code: {position: item-kwargs or None (no image)}}."""
    images, analysis = [], {}
    for code, positions in spec.items():
        for position, kwargs in positions.items():
            image_id = f"{code}-{position}"
            images.append({"image_id": image_id, "wafer_id": WAFER, "die_code": code, "position_code": position,
                           "upload_mode": "wafer_batch", "sha256": image_id})
            if kwargs is not None:
                analysis[image_id] = item(image_id, **kwargs)
    result = die_snapshot(WAFER, [], images, analysis)
    return {d["die_code"]: d for d in apply_judgment_view(result, "AI 잠정 위험도")}


class PdUnusableTests(unittest.TestCase):
    def test_01_pd_hold_beats_other_missing_position_and_plain_incomplete_stays_grey(self):
        d = dies({"13": {"lt": {}, "c": None, "rb": {"outcome": "POLICY_REVIEW_REQUIRED", "heuristic": ("pd",)}},
                  "14": {"lt": {}, "c": None, "rb": {}}})
        self.assertEqual(d["13"]["criticality"], "UNUSABLE")          # Hold at rb + c not analysed -> red.
        self.assertEqual(d["14"]["criticality"], "NOT_EVALUATED")     # No PD evidence -> grey.
        self.assertEqual(die_hover_lines(d["13"]), ["W9 · Die 13", "사용 불가 · PD 의심으로 선폭계측 보류", "rb"])
        self.assertEqual(die_hover_lines(d["14"])[1], "미평가 · 평가 미완료")
        self.assertEqual(d["13"]["assessment"]["pd_unusable"]["pd_count"], 0)   # Hold is not a PD detection.
        summary = die_summary(d["13"])
        self.assertEqual(summary["count"], "0건")
        self.assertIn("보류 rb", summary["line"]);self.assertIn("(2개)", summary["line"])  # lt's 2 valid lines only.

    def test_02_model_pd_detection_counts_and_marks_positions(self):
        d = dies({"21": {"lt": {"outcome": "MODEL_PD_EXCLUDED", "defects": ("pd", "pd")}, "c": {}, "rb": {}}})
        self.assertEqual(d["21"]["criticality"], "UNUSABLE")
        self.assertEqual(die_hover_lines(d["21"]), ["W9 · Die 21", "사용 불가 · PD 검출", "lt (PD 2건)"])
        summary = die_summary(d["21"])
        self.assertEqual(summary["count"], "2건 (pd 2)")
        self.assertIn("제외 lt", summary["line"])

    def test_03_no_valid_measurement_shows_hold_or_exclusion(self):
        d = dies({"31": {"lt": {"outcome": "POLICY_REVIEW_REQUIRED"}, "c": {"outcome": "POLICY_REVIEW_REQUIRED"},
                         "rb": {"outcome": "POLICY_REVIEW_REQUIRED"}}})
        self.assertTrue(die_summary(d["31"])["line"].startswith("계측 보류"))
        d = dies({"32": {"lt": {"outcome": "MODEL_PD_EXCLUDED", "defects": ("pd",)}, "c": {"partial": 1},
                         "rb": {"outcome": "MODEL_PD_EXCLUDED", "defects": ("pd",)}}})
        self.assertTrue(die_summary(d["32"])["line"].startswith("계측 제외"))

    def test_04_scored_die_shows_actual_basis_and_pnpt_lines(self):
        d = dies({"11": {"lt": {"pn": 2, "widths": (7.5, 10.0)}, "c": {"defects": ("s",)}, "rb": {}}})
        lines = die_hover_lines(d["11"])
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("위험도 "))
        self.assertIn("CD 최대 오차 25.0% (lt)", lines[2]);self.assertIn("S 1건 (c)", lines[2])
        self.assertNotIn("점수", " ".join(lines))
        summary = die_summary(d["11"])
        self.assertIn("PN 2 / PT 0 Line", summary["line"]);self.assertEqual(summary["count"], "1건 (s 1)")

    def test_05_image_table_uses_one_image_and_keeps_rows(self):
        d = dies({"35": {"lt": {"widths": (9.0, 9.0)}, "c": {"widths": (11.0, 11.0), "defects": ("s",)},
                         "rb": {"outcome": "POLICY_REVIEW_REQUIRED"}}})
        die = d["35"]
        whole = die_summary(die)
        self.assertTrue(whole["line"].startswith("10.000 µm (4개)"));self.assertIn("보류 rb", whole["line"])
        self.assertEqual(whole["scope"], "집계 lt, c · 보류 rb")
        lt = image_summary(die, "lt", "35-lt")
        self.assertTrue(lt["line"].startswith("9.000 µm (2개)"));self.assertEqual(lt["count"], "0건")
        self.assertEqual(lt["error"], "-10.00% / 10.00%")
        c = image_summary(die, "c", "35-c")
        self.assertEqual((c["kinds"], c["count"]), ("Short(s)", "1건 (s 1)"))
        rb = image_summary(die, "rb", "35-rb")
        self.assertEqual(rb["line"], "계측 보류");self.assertEqual(image_status(die, "35-rb")[0], "사용 불가 · PD 의심으로 선폭계측 보류")
        self.assertEqual(die_summary(die), whole)   # Die table unchanged whatever image is selected.


if __name__ == "__main__":
    unittest.main()
