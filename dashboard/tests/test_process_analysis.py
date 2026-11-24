"""PEB, cohorts and unit boundaries. Numbers are unit-test inputs, never run results."""
from contextlib import closing
from copy import deepcopy
import math
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from dashboard.sqlite_provider import runtime_provider
from dashboard.process_analysis import image_mean_um,summarize_um,metadata_frame,condition_table,ALL,MISSING
from database.migrate_v2 import migrate_database_v2
from database.migrate_v3 import migrate_database_v3
import dashboard.tests.test_sqlite_input as upload_tests


class ProcessCalculationTests(unittest.TestCase):
    def result(self,**changes):
        result={"status":"SUCCESS","pd_detected":False,"calibration_applied":True,"unit":"um",
                "calibration":{"calibration_id":"TEST-ONLY","valid":True,"magnification":20},
                "valid_line_widths_um":[8.0,12.0]}
        result.update(changes);return result

    def test_no_results_failure_pd_and_calibration_reasons_are_distinct(self):
        self.assertEqual(image_mean_um()["status"],"NO_RESULT")
        for changes,expected in [
            ({"status":"FAILED"},"MEASUREMENT_FAILED"),
            ({"pd_detected":True},"PD_EXCLUDED"),
            ({"pd_detected":None},"DEFECT_NOT_ANALYZED"),
            ({"calibration":None},"CALIBRATION_MISSING"),
            ({"calibration_applied":False,"unit":"px"},"NEEDS_CALIBRATED_UM"),
            ({"valid_line_widths_um":[]},"NO_VALID_WIDTHS")]:
            self.assertEqual(image_mean_um(self.result(**changes),magnification=20)["status"],expected)
        self.assertEqual(image_mean_um(self.result(),magnification=None)["status"],"MAGNIFICATION_MISSING")
        self.assertEqual(image_mean_um(self.result(),magnification=40)["status"],"MAGNIFICATION_MISMATCH")

    def test_already_um_is_never_scaled_and_sample_std_is_between_image_means(self):
        result=self.result(um_per_pixel=99)
        self.assertEqual(image_mean_um(result,magnification=20)["mean_um"],10)
        summary=summarize_um([8,12])
        self.assertEqual(summary["mean_um"],10)
        self.assertAlmostEqual(summary["std_um"],math.sqrt(8))
        self.assertEqual(summary["absolute_error_pct"],0)
        self.assertEqual(summarize_um([])["std_state"],"데이터 없음")
        self.assertIsNone(summarize_um([12])["std_um"])
        self.assertEqual(summarize_um([12])["signed_error_pct"],20)
        self.assertEqual(summarize_um([8])["absolute_error_pct"],20)
        with self.assertRaises(ValueError):summarize_um([float("nan")])

    def test_metadata_groups_keep_units_wafer_identity_and_unknown_peb(self):
        frame=metadata_frame(pd.DataFrame([
            {"image_id":"A","wafer_id":"W1","batch_id":"B1","exposure_value":60,"exposure_unit":None,"peb_temperature_c":100},
            {"image_id":"B","wafer_id":"W1","batch_id":"B2","exposure_value":60,"exposure_unit":None,"peb_temperature_c":110},
            {"image_id":"C","wafer_id":"W2","batch_id":"B3","exposure_value":60,"exposure_unit":"stored-unit","peb_temperature_c":None}]))
        table=condition_table(frame,"exposure")
        self.assertEqual(len(table),2)
        self.assertEqual(table.iloc[0]["포함 Wafer 수"],1)
        self.assertEqual(table.iloc[0]["등록 이미지 수"],2)
        self.assertTrue(table["평균 선폭 (µm)"].isna().all())
        self.assertTrue(table["p 검출 건수"].isna().all())
        self.assertIn("PEB 미입력",condition_table(frame,"peb")["공정 조건"].tolist())

    def test_pd_exclusion_is_per_image_not_entire_die(self):
        frame=metadata_frame(pd.DataFrame([
            {"image_id":name,"wafer_id":"W1","batch_id":"B","exposure_value":60,"exposure_unit":None,
             "peb_temperature_c":100,"magnification":20} for name in ("A","B")]))
        table=condition_table(frame,"exposure",results={"A":self.result(pd_detected=True),"B":self.result()})
        self.assertEqual(table.iloc[0]["유효 이미지 수"],1)
        self.assertEqual(table.iloc[0]["PD 제외 이미지 수"],1)
        self.assertEqual(table.iloc[0]["평균 선폭 (µm)"],10)
        self.assertIsNone(table.iloc[0]["이미지 간 표준편차 (µm)"])


class PEBInputTests(unittest.TestCase):
    def setUp(self):
        upload_tests.SQLiteInputTests.setUp(self)
        migrate_database_v2(self.db_path);migrate_database_v3(self.db_path)

    uploaded=upload_tests.SQLiteInputTests.uploaded
    app=upload_tests.SQLiteInputTests.app
    confirm_process=upload_tests.SQLiteInputTests.confirm_process

    def test_peb_input_preview_registration_and_requery_with_batch_rename(self):
        self.uploads=[self.uploaded("w1_11_lt.png")]
        at=self.app();self.confirm_process(at)
        peb=next(row for row in at.number_input if row.key.startswith("sqlite_peb_"))
        self.assertIsNone(peb.value)
        peb.set_value(110.0).run()
        self.assertEqual(at.session_state["sqlite_preview"]["batch_metadata"]["peb_temperature_c"],110)
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run();self.assertFalse(at.exception)
        provider=runtime_provider();batch=provider.batches().iloc[0]
        self.assertEqual(batch["peb_temperature_c"],110)
        record=provider.images().iloc[0]
        self.assertEqual(record["peb_temperature_c"],110)
        self.assertEqual(record["exposure_value"],60)
        provider.rename_batch(batch["batch_id"],"QA display only")
        self.assertEqual(provider.image_record(record["image_id"])["peb_temperature_c"],110)

    def test_invalid_peb_is_rejected_before_registration(self):
        provider=runtime_provider()
        bundle=provider.stage([{"filename":"w1_11_c.png","content":self.uploaded("w1_11_c.png").getvalue()}])
        self.addCleanup(bundle["files"].cleanup)
        metadata={"batch_id":"A","wafer_id":"W1","upload_mode":"wafer_batch","dataset_role":"evaluation",
                  "batch_status":"registered","process_run_id":"P","process_run":{"exposure_value":60}}
        for value in (True,"110",float("inf"),float("nan")):
            preview=provider.preview(bundle["inputs"],dict(metadata,peb_temperature_c=value))
            self.assertFalse(preview["registration_ready"])
        self.assertTrue(provider.batches().empty)

    def test_real_scope_peb_exposure_filters_and_mock_source_isolation(self):
        provider=runtime_provider()
        for index,(wafer,temperature,exposure) in enumerate([("W1",100,60),("W1",110,60),("W5",None,70)]):
            bundle=provider.stage([{"filename":wafer.lower()+"_11_c.png","content":self.uploaded("x",40+index).getvalue()}])
            metadata={"batch_id":"B"+str(index),"wafer_id":wafer,"upload_mode":"wafer_batch",
                      "dataset_role":"evaluation","batch_status":"registered","process_run_id":"P"+wafer,
                      "process_run":{"exposure_value":exposure},"peb_temperature_c":temperature}
            preview=provider.preview(bundle["inputs"],metadata)
            provider.register(preview,approved=True);bundle["files"].cleanup()
        at=AppTest.from_file(upload_tests.ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run();self.assertFalse(at.exception)  # TASK18-B-6: SQLite is the fixed Wafer Map source.
        self.assertEqual([tab.label for tab in at.tabs],["Wafer / Image","선폭 / 오차율","노광","PEB","Repeatability","Lighting"])  # TASK18-B-8: I&A tabs + nested process tabs
        self.assertTrue(any("필터 적용 이미지 3장" in row.value for row in at.caption))
        at.selectbox(key="ia_process_peb_filter").select(100.0).run()
        self.assertTrue(any("필터 적용 이미지 1장" in row.value for row in at.caption))
        at.radio(key="ia_process_scope").set_value("특정 Wafer").run()
        at.selectbox(key="ia_process_wafer").select("W1").run()
        at.selectbox(key="ia_process_peb_filter").select(ALL).run()
        self.assertTrue(any("필터 적용 이미지 2장 · 고유 Wafer 1개" in row.value for row in at.caption))
        at.radio(key="ia_process_scope").set_value("특정 Batch").run()
        at.selectbox(key="ia_process_batch").select("B2").run()
        at.selectbox(key="ia_process_peb_filter").select(MISSING).run()
        self.assertTrue(any("필터 적용 이미지 1장" in row.value for row in at.caption))
        self.assertFalse(any("미리보기 Run" in row.value for row in at.caption))
        at.session_state["_wafer_source"]="Mock";at.run();self.assertFalse(at.exception)  # Mock only via session.
        self.assertNotIn("100 s",at.selectbox(key="ia_process_peb_filter").options)
        self.assertFalse(any("분석 결과 없음" not in row.value and "계측 결과 없음" not in row.value
                             for row in at.info if "결과" in row.value))


if __name__=="__main__":
    unittest.main()
