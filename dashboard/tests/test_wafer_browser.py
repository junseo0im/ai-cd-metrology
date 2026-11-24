"""TASK 8: physical Wafer navigation in isolated v2 DBs. No operational writes."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from dashboard.sqlite_provider import runtime_provider
from dashboard.components.inspection_wafer import die_snapshot,build_inspection_wafer,select_die,DIE_CODES
from dashboard.navigation import page_data_source
import dashboard.tests.test_sqlite_input as fixtures
from database.tests.test_image_registry import png_bytes

ROOT=Path(__file__).resolve().parents[2]


class WaferProviderTests(unittest.TestCase):
    setUp=fixtures.SQLiteBatchDetailTests.setUp

    def seed(self,batch,filenames,wafer="W1",mode="wafer_batch"):
        provider=runtime_provider()
        uploads=[{"filename":name,"content":png_bytes(pixel=30+i+len(provider.batches())*20)}
                 for i,name in enumerate(filenames)]
        bundle=provider.stage(uploads)
        try:
            if mode=="repeatability":
                for index,item in enumerate(bundle["inputs"]):
                    item["repeat_index"]=index+1
            if mode=="lighting":
                for item in bundle["inputs"]:
                    item["acquisition_condition"]={"condition_id":"CONDITION-"+batch,"magnification":20,"illumination_setting":"lm"}
            preview=provider.preview(bundle["inputs"],{
                "batch_id":batch,"wafer_id":wafer,"upload_mode":mode,
                "dataset_role":"evaluation","batch_status":"registered",
                "process_run_id":"PROCESS-"+wafer,"process_run":{"exposure_value":60},
                **({"acquisition_condition":{"condition_id":"CONDITION-"+batch,"magnification":20,"illumination_setting":"lm"}} if mode=="lighting" else {})})
            duplicate_ids=[r["image_id"] for r in preview["images"] if r["status"]=="DUPLICATE_CANDIDATE"]
            return provider.register(preview,approved=True,duplicate_ids=duplicate_ids)
        finally:
            bundle["files"].cleanup()

    def app(self,route="pages/wafer_map.py"):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page(route).run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).
        self.assertFalse(at.exception)
        return at

    def group(self,at,prefix):
        return next(item for item in at.get("button_group") if item.key.startswith(prefix))

    def assert_clean(self,at):
        self.assertFalse(at.exception)

    def test_provider_scope_all_batches_site_and_wafer_identity(self):
        a=self.seed("A",["w1_11_c.png","w1_11_lt.png"])
        b=self.seed("B",["w1_11_c.png","w1_12_rb.png"])
        c=self.seed("C",["w5_11_c.png"],wafer="W5",mode="lighting")
        p=runtime_provider();before=self.db_path.read_bytes()
        self.assertEqual(p.wafers()["wafer_id"].tolist(),["W1","W5"])
        self.assertEqual(p.image_count(wafer_id="W1"),4)
        self.assertEqual(p.image_count(wafer_id="W1",die_code="11",position_code="c"),2)
        self.assertEqual(len(p.images("B",wafer_id="W1")),2)
        self.assertTrue(p.images("B",wafer_id="W5").empty)
        self.assertTrue(p.images(wafer_id="' OR 1=1 --").empty)
        self.assertEqual({r["physical_site_id"] for r in p.wafer_sites("W1")},{"W1/D11/c","W1/D11/lt","W1/D12/rb"})
        self.assertEqual(p.wafer_sites("NO-SUCH-WAFER"),[])
        self.assertEqual(self.db_path.read_bytes(),before)
        self.assertEqual(set(p.images(wafer_id="W5")["image_id"]),set(c["image_ids"]))

    def test_complete_display_grid_preserves_missing_sites_and_multiple_images(self):
        self.seed("A",["w1_11_c.png","w1_11_lt.png"])
        self.seed("B",["w1_11_c.png"])
        p=runtime_provider();dies=die_snapshot("W1",p.wafer_sites("W1"),p.images(wafer_id="W1").to_dict("records"))
        self.assertEqual([d["die_code"] for d in dies],list(DIE_CODES))
        self.assertEqual(len(dies),24)
        self.assertEqual({d["criticality"] for d in dies},{"NOT_EVALUATED"})
        self.assertEqual(len(dies[0]["positions"]["c"]["image_ids"]),2)
        self.assertEqual(dies[0]["positions"]["c"]["physical_site_id"],"W1/D11/c")
        self.assertIsNone(dies[-1]["positions"]["rb"]["physical_site_id"])
        self.assertEqual(dies[-1]["positions"]["rb"]["image_ids"],[])
        for die in dies:
            for site in die["positions"].values():
                self.assertEqual((site["metrology"],site["defect"]),("결과 없음","결과 없음"))
        figure=build_inspection_wafer(dies=dies,selected_die_id="11")
        self.assertEqual(sum(len(trace.x) for trace in figure.data),24)
        self.assertEqual({trace.marker.color for trace in figure.data},{"#b0bac7"})

    def test_main_title_preserves_real_wafer_and_source(self):
        self.seed("A",["w1_11_c.png"])
        self.seed("C",["w5_11_c.png"],wafer="W5")
        at=self.app("pages/inspection_analysis.py")
        at.selectbox(key="ia_wafer_wafer_SQLite").select("W5").run()
        at.button(key="ia_wafer_title").click().run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["active_page_key"],"wafer_map")
        self.assertEqual(at.session_state["_wafer_source"],"SQLite")
        self.assertFalse(any(s.key in ("detail_wafer_source","ia_wafer_source") for s in at.selectbox))  # Selector removed.
        self.assertEqual(at.selectbox(key="detail_wafer_wafer_SQLite").value,"W5")
        self.assertTrue(any("나머지 그래프" in text.value for text in at.caption) is False)

    def test_preview_die_selection_opens_status_and_updates_native_lt_image(self):
        self.seed("A",["w1_11_lt.png","w1_11_c.png"])
        at=self.app("pages/inspection_analysis.py")
        prefix="ia_wafer_map_SQLite_W1_all"
        select_die(at.session_state,prefix,"11",popover=True);at.run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["active_page_key"],"inspection_analysis")
        record=runtime_provider().image_record(at.session_state["ia_sqlite_selected_image_id"])
        self.assertEqual((record["wafer_id"],record["die_code"],record["position_code"]),("W1","11","lt"))
        self.assertTrue(at.session_state[prefix+"_popover"])
        self.assertTrue(any("Wafer W1 · Die 11" in text.value for text in at.markdown))
        # TASK18-B-13A: the I&A header notice was removed; provisional/adoption state stays next to the results.
        self.assertFalse(any("SQLite 실제 등록정보" in text.value for text in at.caption))
        self.assertTrue(any("공식 채택" in text.value for text in at.caption))

    def test_site_multiple_images_selection_batch_filter_and_clear(self):
        first=self.seed("A",["w1_11_c.png","w1_11_lt.png"])
        second=self.seed("B",["w1_11_c.png"],mode="repeatability")
        p=runtime_provider();p.rename_batch("B","반복촬영")
        at=self.app()
        selector=next(row for row in at.selectbox if row.key.startswith("wafer_image_SQLite"))
        self.assertEqual(len(selector.options),2)
        selector.select(second["image_ids"][0]).run();self.assert_clean(at)
        self.assertEqual(at.session_state["wafer_browser_image_id"],second["image_ids"][0])
        self.assertIn(second["image_ids"][0],[row.value for row in at.code])
        self.assertTrue(any("반복촬영" in row.value for row in at.caption))
        at.selectbox(key="detail_wafer_batch_SQLite_W1").select("A").run()
        selector=next(row for row in at.selectbox if row.key.startswith("wafer_image_SQLite"))
        self.assertEqual(len(selector.options),1)
        self.assertIn(selector.value,first["image_ids"])
        at.selectbox(key="detail_wafer_batch_SQLite_W1").select(None).run()
        selector=next(row for row in at.selectbox if row.key.startswith("wafer_image_SQLite"))
        self.assertEqual(len(selector.options),2)

    def test_position_die_and_wafer_changes_never_keep_an_unrelated_image(self):
        self.seed("A",["w1_11_c.png","w1_11_lt.png","w1_12_rb.png"])
        other=self.seed("C",["w5_11_c.png"],wafer="W5")
        at=self.app()
        self.group(at,"wafer_position_").set_value("lt").run()
        selected=runtime_provider().image_record(at.session_state["wafer_browser_image_id"])
        self.assertEqual((selected["wafer_id"],selected["die_code"],selected["position_code"]),("W1","11","lt"))
        select_die(at.session_state,"detail_wafer_map_SQLite_W1_all","12");at.run()
        self.assert_clean(at)
        # TASK18-B-14B: Die 12 opens on its only imaged position (rb), never on Die 11's image.
        self.assertEqual(self.group(at,"wafer_position_").value,"rb")
        self.group(at,"wafer_position_").set_value("lt").run()
        self.assertIsNone(at.session_state["wafer_browser_image_id"])
        self.assertTrue(any(row.value=="등록 이미지 없음" for row in at.info))
        self.group(at,"wafer_position_").set_value("rb").run()
        selected=runtime_provider().image_record(at.session_state["wafer_browser_image_id"])
        self.assertEqual((selected["die_code"],selected["position_code"]),("12","rb"))
        at.selectbox(key="detail_wafer_wafer_SQLite").select("W5").run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["wafer_browser_image_id"],other["image_ids"][0])

    def test_real_preview_exact_file_missing_file_notice_and_no_fake_overlay(self):
        result=self.seed("A",["w1_11_c.png"])
        provider=runtime_provider();record=provider.image_record(result["image_ids"][0])
        path=Path(record["storage_path"]);original=hashlib.sha256(path.read_bytes()).hexdigest()
        import streamlit
        real=streamlit.image;seen=[]
        def image(value,*args,**kwargs):
            seen.append(value);return real(value,*args,**kwargs)
        with patch("streamlit.image",side_effect=image):
            at=self.app()
            seen.clear()
            for mode in ["Metrology","Defect","Metrology + Defect","원본만"]:
                self.group(at,"wafer_overlay_").set_value(mode).run()
                self.assert_clean(at)
        self.assertTrue(seen)
        self.assertTrue(all(Path(value)==path for value in seen))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original)
        # Only this isolated fixture path is mocked as missing; no file is deleted.
        exists=Path.is_file
        with patch.object(Path,"is_file",lambda value:False if value==path else exists(value)):
            at.run();self.assert_clean(at)
        self.assertTrue(any("원본 파일을 찾거나" in row.value for row in at.warning))

    def test_no_wafer_has_no_mock_fallback_or_insert(self):
        before=self.db_path.read_bytes()
        at=self.app()
        self.assertTrue(any("등록된 SQLite Wafer 없음" in row.value for row in at.info))
        self.assertFalse(at.get("plotly_chart"))
        self.assertFalse(at.get("imgs"))
        self.assertEqual(self.db_path.read_bytes(),before)

    def test_mock_and_sqlite_source_and_id_spaces_remain_separate(self):
        result=self.seed("A",["w1_11_c.png"])
        at=self.app()
        self.assertEqual(at.selectbox(key="detail_wafer_wafer_SQLite").options,["W1"])
        at.session_state["_wafer_source"]="Mock";at.run()  # Mock only via explicit session (no UI).
        self.assert_clean(at)
        self.assertTrue(at.session_state["wafer_browser_image_id"].startswith("IMG-DEMO-"))
        self.assertFalse(any(row.value in result["image_ids"] for row in at.code))
        self.assertEqual(page_data_source("wafer_map",{"_wafer_source":"SQLite"}),"SQLite")
        self.assertEqual(page_data_source("image_analysis",{"_wafer_source":"SQLite"}),"Mock")
        self.assertFalse(any(button.label in DIE_CODES for button in at.button))


if __name__=="__main__":
    unittest.main()
