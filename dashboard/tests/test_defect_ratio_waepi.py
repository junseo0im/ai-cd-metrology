"""TASK 18-B-18: condition defect share (I&A) and the '웨피' upload/registration states. Synthetic data only."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from dashboard.process_analysis import defect_ratio_table, metadata_frame


def item(run, started, details=(), status="SUCCESS"):
    return {"analysis_run_id": run, "started_at": started, "defect_status": status,
            "defect_details": [{"class_code": c, "source": src} for c, src in details]}


class DefectRatioTests(unittest.TestCase):
    def setUp(self):
        images = pd.DataFrame([
            {"image_id": "a1", "sha256": "s1", "exposure_value": 50.0, "exposure_unit": None, "peb_temperature_c": 220.0},
            {"image_id": "a2", "sha256": "s2", "exposure_value": 50.0, "exposure_unit": None, "peb_temperature_c": 220.0},
            {"image_id": "a1copy", "sha256": "s1", "exposure_value": 50.0, "exposure_unit": None, "peb_temperature_c": 220.0},
            {"image_id": "b1", "sha256": "s3", "exposure_value": 60.0, "exposure_unit": None, "peb_temperature_c": 220.0},
            {"image_id": "c1", "sha256": "s4", "exposure_value": 70.0, "exposure_unit": None, "peb_temperature_c": 220.0}])
        self.images = metadata_frame(images)
        self.analysis = {
            "a1": item("r1", "2026-11-20", [("p", "MODEL_POSTPROCESSED"), ("s", "MODEL_POSTPROCESSED"),
                                             ("pd", "HEURISTIC")]),            # Rule-based PD suspicion: not counted.
            "a1copy": item("r0", "2026-11-19", [("p", "MODEL_POSTPROCESSED")]),  # Same capture, older: excluded.
            "a2": item("r2", "2026-11-20", [("sc", "MODEL_POSTPROCESSED")]),
            "b1": item("r3", "2026-11-20", []),                                  # Analysed, 0 defects.
        }                                                                        # c1: no result.

    def test_01_share_per_condition_with_zero_and_no_result_kept_apart(self):
        table = defect_ratio_table(self.images, "exposure", self.analysis).set_index("공정 조건")
        self.assertEqual(table.attrs["total_defects"], 3)
        self.assertEqual(table.attrs["duplicates_excluded"], 1)
        self.assertEqual((table.loc["50 s", "결함 건수"], table.loc["50 s", "결함 분석 완료 이미지 수"]), (3, 2))
        self.assertAlmostEqual(table.loc["50 s", "결함 개수 비율 (%)"], 100.0)
        self.assertAlmostEqual(table.loc["50 s", "이미지 100장당 결함 건수"], 150.0)
        self.assertEqual(table.loc["60 s", "결함 개수 비율 (%)"], 0.0)                 # Analysed, 0 defects -> 0 %.
        self.assertEqual(table.loc["60 s", "결함 분석 완료 이미지 수"], 1)
        self.assertTrue(pd.isna(table.loc["70 s", "결함 개수 비율 (%)"]))                # No result -> no bar.
        self.assertEqual(table.loc["70 s", "결함 분석 미실행·결과 없음 이미지 수"], 1)

    def test_02_whole_scope_zero_defects_has_no_ratio(self):
        analysis = {k: item(v["analysis_run_id"], v["started_at"]) for k, v in self.analysis.items()}
        table = defect_ratio_table(self.images, "peb", analysis)
        self.assertEqual(table.attrs["total_defects"], 0)
        self.assertTrue(table["결함 개수 비율 (%)"].isna().all())


class DerivedParticleCountTests(unittest.TestCase):
    """TASK18-B-22: S + P split out of it = 1 defect; independent P counted as before. Synthetic data only."""
    def test_pair_counts_once_and_independent_p_counts(self):
        from dashboard.process_analysis import count_defects, derived_pairs
        details = [{"class_code": "s", "source": "MODEL_POSTPROCESSED", "team_source": "model"},
                   {"class_code": "p", "source": "RULE", "team_source": "from_s", "derived_from_index": 0},
                   {"class_code": "p", "source": "MODEL_POSTPROCESSED", "team_source": "model"},   # overlaps, independent
                   {"class_code": "s", "source": "MODEL_POSTPROCESSED", "team_source": "model"},
                   {"class_code": "p", "source": "RULE", "team_source": "from_s", "derived_from_index": 3}]
        self.assertEqual(count_defects(details), 3)          # S+P, independent P, S+P
        self.assertEqual(derived_pairs(details), ([(0, 1), (3, 4)], []))

    def test_unresolved_link_is_reported_not_guessed(self):
        from dashboard.process_analysis import derived_pairs
        details = [{"class_code": "s", "source": "MODEL_POSTPROCESSED"},
                   {"class_code": "p", "source": "RULE", "team_source": "from_s", "derived_from_index": None}]
        self.assertEqual(derived_pairs(details), ([], [1]))

    def test_condition_table_uses_pair_rule(self):
        images = metadata_frame(pd.DataFrame([{"image_id": "x", "sha256": "h", "exposure_value": 50.0,
                                                "exposure_unit": None, "peb_temperature_c": 220.0}]))
        analysis = {"x": {"analysis_run_id": "r", "started_at": "2026-11-25", "defect_status": "SUCCESS",
                          "defect_details": [{"class_code": "s", "source": "MODEL_POSTPROCESSED", "team_source": "model"},
                                             {"class_code": "p", "source": "RULE", "team_source": "from_s",
                                              "derived_from_index": 0},
                                             {"class_code": "p", "source": "MODEL_POSTPROCESSED", "team_source": "model"}]}}
        table = defect_ratio_table(images, "exposure", analysis)
        self.assertEqual(int(table["결함 건수"].iloc[0]), 2)
        self.assertEqual(int(table["S 파생 P (S와 1건 합산)"].iloc[0]), 1)


class WaepiFlowTests(unittest.TestCase):
    def view(self, state, now):
        from dashboard.components import waepi
        fake = SimpleNamespace(session_state=state)
        with patch.object(waepi, "st", fake), patch.object(waepi.time, "time", return_value=now):
            return waepi.session_state_view()["text"]

    def test_01_upload_register_and_before_analysis(self):
        from dashboard.components.waepi import GREETING
        state = {}
        self.assertEqual(self.view(state, 100), GREETING)
        state["sqlite_uploads"] = {"inputs": [{"image_id": "x"}]}
        self.assertEqual(self.view(state, 101), "이미지 확인 중입니다")
        state["_waepi_event"] = {"kind": "register_failed", "time": 102}
        self.assertEqual(self.view(state, 103), "DB 등록 실패 · 오류 내용을 확인해 주세요")   # Never the success line.
        self.assertEqual(self.view(state, 115), "이미지 확인 중입니다")
        state["sqlite_uploads"]["registered"] = {"batch_id": "B"}
        state["_waepi_event"] = {"kind": "registered", "time": 130}
        self.assertEqual(self.view(state, 131), "이미지 DB 등록 완료")
        self.assertEqual(self.view(state, 141), "이미지 DB 등록 완료")                         # Held (no sleep).
        self.assertEqual(self.view(state, 143), "분석 실행 전")
        state["_waepi_event"] = {"kind": "analysis_started", "time": 150}
        self.assertEqual(self.view(state, 151), GREETING)    # Analysis states come from the existing job path.


if __name__ == "__main__":
    unittest.main()
