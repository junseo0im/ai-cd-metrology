"""TASK 10 records: isolated JSON, no operational DB or research-image writes."""
from copy import deepcopy
import json
import os
import secrets
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from dashboard.record_defaults import definition,NOTICE
from dashboard import auth
from dashboard.record_store import (RecordStore,RecordError,RecordConflict,default_rows,
                                    tab_token,file_lock)

ROOT=Path(__file__).resolve().parents[2]
TMP=ROOT/"dashboard"/".tmp"


class RecordStoreTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(prefix="record_test_",dir=TMP)
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(TMP.resolve()))
        env=patch.dict(os.environ,{"IDA_RECORDS_DIR":str(self.root),"IDA_RECORDS_EDIT_ENABLED":"1"})
        env.start();self.addCleanup(env.stop)
        self.admin_guard=patch("dashboard.auth.is_admin",return_value=True)
        self.admin_guard.start();self.addCleanup(self.admin_guard.stop)
        self.store=RecordStore()
        self.doc=self.store.load("experiment_conditions")

    def change(self,tab="equipment",identity="magnification",value="40"):
        rows=deepcopy(self.doc["tabs"][tab]["rows"])
        next(row for row in rows if row["id"]==identity)["value"]=value
        return rows

    def save(self,rows,tab="equipment",document=None):
        doc=document or self.doc
        return self.store.save_tab("experiment_conditions",tab,rows,
                                  expected_token=tab_token(doc["tabs"][tab]))

    def test_first_load_initializes_once_and_defaults_are_not_invented(self):
        before=self.store.path("experiment_conditions").read_bytes()
        self.assertEqual(RecordStore(self.root).load("experiment_conditions"),self.doc)
        self.assertEqual(self.store.path("experiment_conditions").read_bytes(),before)
        tabs=self.doc["tabs"]
        self.assertEqual(len(tabs),7)
        equipment={r["id"]:r for r in tabs["equipment"]["rows"]}
        self.assertEqual(equipment["software"]["value"],"OptiView")
        self.assertEqual(equipment["model"]["value"],"")
        self.assertEqual(equipment["auto_exposure"]["status"],"검토 필요")
        for key in ["scale_x","scale_y","conversion"]:
            self.assertEqual(next(r for r in tabs["calibration"]["rows"] if r["id"]==key)["value"],"")
        critical=self.store.load("critical_rules")
        self.assertEqual(len(critical["tabs"]),4)
        codes=next(r for r in critical["tabs"]["defect"]["rows"] if r["id"]=="classes")
        self.assertEqual(codes["value"],"s / o / sc / p / pd")

    def test_save_survives_new_store_preserves_other_tabs_and_versions(self):
        result=self.save(self.change())
        again=RecordStore(self.root).load("experiment_conditions")
        self.assertEqual(result,again)
        self.assertEqual(result["record_version"],2)
        self.assertEqual(result["tabs"]["equipment"]["revision"],2)
        self.assertNotEqual(result["updated_at"],self.doc["updated_at"])
        for key in self.doc["tabs"]:
            if key!="equipment":
                self.assertEqual(result["tabs"][key],self.doc["tabs"][key])

    def test_invalid_numeric_and_status_inputs_never_change_file(self):
        path=self.store.path("experiment_conditions");before=path.read_bytes()
        for bad in ["abc","20 µm","-1","0","NaN","Infinity"]:
            with self.subTest(value=bad),self.assertRaises(RecordError):
                self.save(self.change(value=bad))
            self.assertEqual(path.read_bytes(),before)
        rows=deepcopy(self.doc["tabs"]["structure"]["rows"])
        next(r for r in rows if r["id"]=="dies")["value"]="2.5"
        with self.assertRaises(RecordError):self.save(rows,"structure")
        rows=self.change()
        rows[0]["status"]="UNKNOWN"
        with self.assertRaises(RecordError):self.save(rows)
        self.assertEqual(path.read_bytes(),before)

    def test_blank_review_or_missing_is_allowed_but_not_confirmed(self):
        rows=self.change(identity="model",value="New model")
        row=next(r for r in rows if r["id"]=="model")
        with self.assertRaises(RecordError):self.save(rows)
        row["status"]="검토 필요"
        result=self.save(rows)
        rows=deepcopy(result["tabs"]["equipment"]["rows"])
        row=next(r for r in rows if r["id"]=="model");row["value"]="";row["status"]="확정"
        with self.assertRaises(RecordError):self.save(rows,document=result)
        row["status"]="미입력"
        self.save(rows,document=result)

    def test_failed_atomic_replace_keeps_normal_file_and_cleans_owned_temporary(self):
        path=self.store.path("experiment_conditions");before=path.read_bytes()
        with patch("dashboard.record_store.os.replace",side_effect=PermissionError("test-only failure")):
            with self.assertRaises(PermissionError):self.save(self.change())
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),["experiment_conditions.json"])

    def test_corrupt_or_incompatible_json_is_not_reset(self):
        path=self.store.path("experiment_conditions")
        for content in ['{"broken":',json.dumps(dict(self.doc,schema_version=999)),
                        json.dumps(dict(self.doc,document_id="critical_rules"))]:
            path.write_text(content,encoding="utf-8");before=path.read_bytes()
            with self.assertRaises(RecordError):self.store.load("experiment_conditions")
            self.assertEqual(path.read_bytes(),before)

    def test_same_tab_conflict_rejected_but_different_tabs_merge(self):
        changed=self.save(self.change())
        with self.assertRaises(RecordConflict):self.save(self.change(value="30"))
        process=deepcopy(self.doc["tabs"]["process"]["rows"])
        next(r for r in process if r["id"]=="notes")["value"]="Independent tab"
        next(r for r in process if r["id"]=="notes")["status"]="검토 필요"
        result=self.save(process,"process")
        self.assertEqual(result["tabs"]["equipment"],changed["tabs"]["equipment"])
        self.assertEqual(result["record_version"],3)

    def test_existing_lock_and_revoked_permission_both_block_writes(self):
        path=self.store.path("experiment_conditions");before=path.read_bytes()
        with file_lock(path):
            with self.assertRaises(RecordConflict):self.save(self.change())
        with patch("dashboard.auth.is_admin",return_value=False):
            with self.assertRaises(PermissionError):self.save(self.change())
        self.assertEqual(path.read_bytes(),before)

    def test_default_restore_is_copy_only_until_explicit_save(self):
        saved=self.save(self.change());path=self.store.path("experiment_conditions")
        before=path.read_bytes();defaults=default_rows("experiment_conditions","equipment")
        defaults[0]["note"]="Only a copy"
        self.assertEqual(path.read_bytes(),before)
        clean=default_rows("experiment_conditions","equipment")
        result=self.save(clean,document=saved)
        self.assertEqual(result["tabs"]["equipment"]["rows"],clean)

    def test_fixed_document_paths_and_rows_cannot_be_retargeted(self):
        with self.assertRaises(ValueError):self.store.load("../ida.sqlite3")
        rows=self.change();rows[0]["item"]="renamed"
        with self.assertRaises(RecordError):self.save(rows)
        rows=self.change();rows[0]["id"]="unexpected"
        with self.assertRaises(RecordError):self.save(rows)


class RecordPageTests(unittest.TestCase):
    def setUp(self):
        RecordStoreTests.setUp(self)
        self.admin_guard.stop()  # UI tests exercise real login, not the storage-only mock.
        self.password=secrets.token_urlsafe(24)
        self.encoded=auth.hash_password(self.password)
        env=patch.dict(os.environ,{"IDA_ADMIN_USERNAME":"qa-admin","IDA_ADMIN_PASSWORD_HASH":self.encoded,
                                 "IDA_ADMIN_DISPLAY_NAME":"QA ADMIN","IDA_RECORDS_EDIT_ENABLED":"0"})
        env.start();self.addCleanup(env.stop)
        sessions=patch.object(auth,"_sessions",auth.AdminSessions())
        sessions.start();self.addCleanup(sessions.stop)

    def app(self,kind="experiment_conditions",*,admin=True):
        route="pages/experiment_environment.py" if kind=="experiment_conditions" else "pages/critical_criteria.py"
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        if admin:
            at.switch_page("pages/settings.py").run()
            at.text_input(key="admin_login_username").set_value("qa-admin")
            at.text_input(key="admin_login_password").set_value(self.password)
            at.button(key="admin_login_submit").click().run()
            self.assertFalse(at.exception)
        at.switch_page(route).run()
        self.assertFalse(at.exception)
        return at

    def editor_patch(self,edits):
        # AppTest has no native data-editor cell action. Keep its actual widget
        # and inject returned cell values at this one boundary. Chrome tests
        # exercise the real grid separately.
        import streamlit
        real=streamlit.data_editor
        def edit(table,*args,**kwargs):
            result=real(table,*args,**kwargs).copy()
            for (identity,column),value in edits.items():
                if identity in result.index:
                    result.loc[identity,column]=value
            return result
        return patch("streamlit.data_editor",side_effect=edit)

    def test_both_page_tabs_notice_and_readonly_default(self):
        with patch.dict(os.environ,{"IDA_RECORDS_EDIT_ENABLED":"0"}):
            for kind in ["experiment_conditions","critical_rules"]:
                at=self.app(kind,admin=False)
                self.assertEqual([t.label for t in at.tabs],[title for key,title,rows in definition(kind)["tabs"]])
                # TASK18-B-20: only the GUEST note above the tables; one '탭 수정 · 버전' caption per tab (under its table).
                self.assertEqual([row.value for row in at.info],["GUEST 조회 전용 — 기록 수정은 관리자 로그인이 필요합니다."])
                self.assertNotIn(NOTICE,[row.value for row in at.caption])
                self.assertFalse(any(row.value.startswith(("기록 편집 권한","마지막 수정","기록 파일:")) for row in at.caption))
                notes=[row.value for row in at.caption if row.value.startswith("탭 수정: ")]
                self.assertEqual(len(notes),len(definition(kind)["tabs"]))
                for tab in at.tabs:   # Inside each tab: table first, then its revision caption.
                    kinds=[type(node).__name__ for node in tab.children.values()]
                    self.assertLess(max(i for i,k in enumerate(kinds) if k=="Dataframe"),
                                    max(i for i,k in enumerate(kinds) if k=="Caption"),tab.label)
                edits=[row for row in at.button if row.key and row.key.startswith("record_") and row.key.endswith("_edit")]
                self.assertTrue(edits and all(row.disabled for row in edits))
                self.assertFalse(any(row.key and "_editor_" in row.key for row in at.dataframe))

    def test_edit_cancel_and_save_persist_after_new_session(self):
        at=self.app();prefix="record_experiment_conditions_equipment"
        before=self.store.path("experiment_conditions").read_bytes()
        at.button(key=prefix+"_edit").click().run()
        with self.editor_patch({("magnification","기록값"):"40"}):
            at.button(key=prefix+"_cancel").click().run()
        self.assertEqual(self.store.path("experiment_conditions").read_bytes(),before)
        at.button(key=prefix+"_edit").click().run()
        with self.editor_patch({("magnification","기록값"):"40"}):
            at.button(key=prefix+"_save").click().run()
        self.assertFalse(at.exception)
        loaded=self.store.load("experiment_conditions")
        self.assertEqual(next(r for r in loaded["tabs"]["equipment"]["rows"] if r["id"]=="magnification")["value"],"40")
        fresh=self.app()
        self.assertTrue(any("40" in row.value["기록값"].tolist() for row in fresh.dataframe if "항목" in row.value))

    def test_default_restore_stays_draft_then_cancel_or_save(self):
        self.save=RecordStoreTests.save.__get__(self)
        self.change=RecordStoreTests.change.__get__(self)
        with patch("dashboard.auth.is_admin",return_value=True):
            self.save(self.change())
        prefix="record_experiment_conditions_equipment";at=self.app()
        at.button(key=prefix+"_edit").click().run()
        path=self.store.path("experiment_conditions");before=path.read_bytes()
        at.button(key=prefix+"_restore").click().run()
        self.assertEqual(path.read_bytes(),before)
        self.assertTrue(any("초안" in row.value for row in at.info))
        at.button(key=prefix+"_cancel").click().run()
        self.assertEqual(path.read_bytes(),before)
        at.button(key=prefix+"_edit").click().run()
        at.button(key=prefix+"_restore").click().run()
        at.button(key=prefix+"_save").click().run()
        self.assertEqual(self.store.load("experiment_conditions")["tabs"]["equipment"]["rows"],
                         default_rows("experiment_conditions","equipment"))

    def test_invalid_number_keeps_editing_and_normal_file(self):
        at=self.app();prefix="record_experiment_conditions_equipment"
        at.button(key=prefix+"_edit").click().run()
        path=self.store.path("experiment_conditions");before=path.read_bytes()
        with self.editor_patch({("magnification","기록값"):"-1"}):
            at.button(key=prefix+"_save").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("0보다 큰" in row.value for row in at.error))
        self.assertEqual(path.read_bytes(),before)
        self.assertTrue(any(row.key==prefix+"_save" for row in at.button))

    def test_tab_drafts_do_not_cross_and_stale_save_is_rejected(self):
        at=self.app();a="record_experiment_conditions_equipment";b="record_experiment_conditions_process"
        at.button(key=a+"_edit").click().run()
        at.button(key=b+"_edit").click().run()
        at.button(key=b+"_restore").click().run()
        at.button(key=b+"_save").click().run()
        self.assertTrue(any(row.key==a+"_save" for row in at.button))
        latest=self.store.load("experiment_conditions")
        rows=deepcopy(latest["tabs"]["equipment"]["rows"]);rows[0]["note"]="Another editor"
        with patch("dashboard.auth.is_admin",return_value=True):
            self.store.save_tab("experiment_conditions","equipment",rows,expected_token=tab_token(latest["tabs"]["equipment"]))
        at.button(key=a+"_save").click().run()
        self.assertTrue(any("다른 세션" in row.value for row in at.error))
        self.assertEqual(self.store.load("experiment_conditions")["tabs"]["equipment"]["rows"][0]["note"],"Another editor")

    def test_server_gate_disables_existing_draft_and_corrupt_json_is_reported(self):
        at=self.app();prefix="record_experiment_conditions_equipment"
        at.button(key=prefix+"_edit").click().run()
        at.button(key="header_logout").click().run()
        self.assertFalse(any(row.key==prefix+"_save" for row in at.button))
        self.assertTrue(at.button(key=prefix+"_edit").disabled)
        self.store.path("experiment_conditions").write_text("{broken",encoding="utf-8")
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("JSON" in row.value for row in at.error))


if __name__=="__main__":
    unittest.main()
