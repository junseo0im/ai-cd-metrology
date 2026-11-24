"""Single-ADMIN auth tests; random credentials and isolated record JSON only."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import os
from pathlib import Path
import secrets
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from dashboard import auth
from dashboard.record_store import RecordStore,tab_token
import dashboard.tests.test_record_pages as records

ROOT=Path(__file__).resolve().parents[2]


class AuthCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password=secrets.token_urlsafe(24)
        cls.encoded=auth.hash_password(cls.password)
        cls.config=auth.AdminConfig("qa-admin",cls.encoded,"QA ADMIN")

    def test_salted_hash_verification_and_safe_representation(self):
        other=auth.hash_password(self.password)
        self.assertNotEqual(self.encoded,other)
        self.assertTrue(auth.verify_password(self.password,self.encoded))
        self.assertFalse(auth.verify_password(secrets.token_urlsafe(24),self.encoded))
        for value in ("",None,"plaintext","pbkdf2_sha256$1$aa$bb"):
            self.assertFalse(auth.verify_password(self.password,value))
        self.assertNotIn(self.password,self.encoded)
        self.assertNotIn(self.encoded,repr(self.config))

    def test_server_grants_bind_session_and_config_and_expire(self):
        now=[100.0];service=auth.AdminSessions(clock=lambda:now[0])
        token=service.login(self.config,"owner","qa-admin",self.password)
        self.assertTrue(service.authorized(self.config,"owner",token))
        self.assertFalse(service.authorized(self.config,"other-browser",token))
        self.assertFalse(service.authorized(self.config,"owner","ADMIN"))
        self.assertFalse(service.authorized(None,"owner",token))
        self.assertFalse(service.authorized(replace(self.config,username="changed"),"owner",token))
        service.revoke("other-browser",token)
        self.assertTrue(service.authorized(self.config,"owner",token))
        now[0]+=auth.SESSION_SECONDS+1
        self.assertFalse(service.authorized(self.config,"owner",token))
        token=service.login(self.config,"owner","qa-admin",self.password)
        service.revoke("owner",token)
        self.assertFalse(service.authorized(self.config,"owner",token))

    def test_failed_login_is_generic_and_throttled_across_sessions(self):
        now=[100.0];service=auth.AdminSessions(clock=lambda:now[0]);output=io.StringIO()
        with redirect_stdout(output):
            for index in range(auth.FAILURE_LIMIT):
                with self.assertRaises(auth.AuthenticationError) as error:
                    service.login(self.config,"session-"+str(index),"qa-admin",secrets.token_urlsafe(24))
                self.assertEqual(str(error.exception),"로그인 정보를 확인하세요.")
            with self.assertRaisesRegex(auth.AuthenticationError,"일시 제한"):
                service.login(self.config,"new-browser","qa-admin",self.password)
            self.assertFalse(service.authorized(self.config,"new-browser","ADMIN"))
            now[0]+=auth.COOLDOWN_SECONDS+1
            token=service.login(self.config,"new-browser","qa-admin",self.password)
            self.assertTrue(service.authorized(self.config,"new-browser",token))
        self.assertEqual(output.getvalue(),"")

    def test_server_configuration_unset_partial_malformed_and_secrets(self):
        clean={key:value for key,value in os.environ.items() if not key.startswith("IDA_ADMIN_")}
        with patch.dict(os.environ,clean,clear=True),patch("streamlit.secrets",{}):
            self.assertIsNone(auth.configured_admin())
            with patch.dict(os.environ,{"IDA_ADMIN_USERNAME":"qa-admin"}):
                self.assertIsNone(auth.configured_admin())
            with patch("streamlit.secrets",{"ida_admin":{"username":"qa-admin","password_hash":self.encoded}}):
                self.assertEqual(auth.configured_admin().username,"qa-admin")
            with patch.dict(os.environ,{"IDA_ADMIN_USERNAME":"qa-admin","IDA_ADMIN_PASSWORD_HASH":self.encoded}):
                self.assertEqual(auth.configured_admin().display_name,"ADMIN")
                with patch.dict(os.environ,{"IDA_ADMIN_PASSWORD_HASH":"malformed"}):
                    self.assertIsNone(auth.configured_admin())

    def test_interactive_setup_saves_hash_only_and_preserves_existing_sections(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"dashboard/.tmp") as directory:
            path=Path(directory)/"secrets.toml";path.write_text('[other]\nvalue="keep"\n',encoding="utf-8")
            output=io.StringIO()
            with patch("builtins.input",side_effect=["qa-admin","QA ADMIN"]),patch(
                    "dashboard.auth.getpass.getpass",side_effect=[self.password,self.password]),redirect_stdout(output):
                auth.setup_admin(path)
            doc=tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(doc["other"],{"value":"keep"})
            self.assertTrue(auth.verify_password(self.password,doc["ida_admin"]["password_hash"]))
            self.assertNotIn(self.password,path.read_text(encoding="utf-8"))
            self.assertNotIn(self.password,output.getvalue())
            self.assertNotIn(doc["ida_admin"]["password_hash"],output.getvalue())
            before=path.read_bytes()
            with self.assertRaises(ValueError):auth.setup_admin(path)
            self.assertEqual(path.read_bytes(),before)

    def test_direct_store_blocks_guest_and_legacy_flag_and_rechecks_before_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"dashboard/.tmp") as directory:
            store=RecordStore(directory);doc=store.load("critical_rules")
            tab_id=next(iter(doc["tabs"]));tab=doc["tabs"][tab_id];path=store.path("critical_rules")
            before=path.read_bytes()
            with patch.dict(os.environ,{"IDA_RECORDS_EDIT_ENABLED":"1"}):
                self.assertFalse(auth.is_admin())
                with self.assertRaises(PermissionError):
                    store.save_tab("critical_rules",tab_id,deepcopy(tab["rows"]),expected_token=tab_token(tab))
            with patch("dashboard.auth.is_admin",side_effect=[True,False]):
                with self.assertRaises(PermissionError):
                    store.save_tab("critical_rules",tab_id,deepcopy(tab["rows"]),expected_token=tab_token(tab))
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(path.with_suffix(".json.lock").exists())


class AuthPageTests(unittest.TestCase):
    setUp=records.RecordPageTests.setUp
    app=records.RecordPageTests.app

    def settings(self):
        return AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run().switch_page("pages/settings.py").run()

    def test_no_configuration_and_legacy_toggle_never_grant_editing(self):
        with patch("dashboard.auth.configured_admin",return_value=None),patch.dict(os.environ,{"IDA_RECORDS_EDIT_ENABLED":"1"}):
            at=self.settings()
            self.assertFalse(at.exception)
            self.assertTrue(any("관리자 계정 미설정" in item.value for item in at.info))
            self.assertFalse(any(b.key=="admin_login_submit" for b in at.button))
            for kind in ("experiment_conditions","critical_rules"):
                at=self.app(kind,admin=False)
                self.assertTrue(all(b.disabled for b in at.button if b.key and b.key.endswith("_edit")))

    def test_role_username_and_query_parameters_are_not_authentication(self):
        at=self.app(admin=False)
        at.session_state["role"]="ADMIN";at.session_state["username"]="qa-admin"
        at.session_state[auth.TOKEN_KEY]=secrets.token_urlsafe(32)
        at.query_params["role"]="ADMIN";at.run()
        self.assertFalse(at.exception)
        self.assertTrue(at.button(key="record_experiment_conditions_equipment_edit").disabled)
        self.assertFalse(any(item.key=="header_logout" for item in at.button))
        self.assertTrue(any("GUEST 조회 전용" in item.value for item in at.info))

    def test_wrong_password_then_real_login_header_and_logout_remove_draft(self):
        at=self.settings()
        at.text_input(key="admin_login_username").set_value("qa-admin")
        at.text_input(key="admin_login_password").set_value(secrets.token_urlsafe(24))
        at.button(key="admin_login_submit").click().run()
        self.assertTrue(any("로그인 정보를 확인" in item.value for item in at.error))
        self.assertFalse(any(item.key=="settings_logout" for item in at.button))
        at.text_input(key="admin_login_username").set_value("qa-admin")
        at.text_input(key="admin_login_password").set_value(self.password)
        at.button(key="admin_login_submit").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any(item.key=="settings_logout" for item in at.button))
        self.assertTrue(any("QA ADMIN" in str(item.proto) for item in at.get("popover")))
        self.assertFalse(any(item.value==self.password for item in at.text_input))
        at.switch_page("pages/critical_criteria.py").run()
        prefix="record_critical_rules_"+next(iter(self.store.load("critical_rules")["tabs"]))
        self.assertFalse(at.button(key=prefix+"_edit").disabled)
        at.button(key=prefix+"_edit").click().run()
        at.button(key="header_logout").click().run()
        self.assertTrue(at.button(key=prefix+"_edit").disabled)
        self.assertNotIn(prefix+"_draft",at.session_state)
        self.assertFalse(any(item.key==prefix+"_save" for item in at.button))

    def test_new_session_is_guest_and_sees_saved_admin_value(self):
        at=self.app();prefix="record_experiment_conditions_equipment"
        at.button(key=prefix+"_edit").click().run()
        with records.RecordPageTests.editor_patch(self,{("magnification","기록값"):"40"}):
            at.button(key=prefix+"_save").click().run()
        guest=self.app(admin=False)
        self.assertTrue(guest.button(key=prefix+"_edit").disabled)
        self.assertTrue(any("40" in item.value["기록값"].tolist() for item in guest.dataframe if "항목" in item.value))
        self.assertFalse(any(item.key=="header_logout" for item in guest.button))

    def test_header_uses_existing_settings_route_and_permission_policy(self):
        at=self.app(admin=False)
        at.button(key="header_login").click().run()
        self.assertEqual(at.session_state["active_page_key"],"settings")
        self.assertFalse(at.exception)
        self.assertTrue(any(item.value=="사용자 및 권한" for item in at.subheader))
        self.assertTrue(any("MEMBER: 향후 확장" in item.value for item in at.caption))


if __name__=="__main__":
    unittest.main()
