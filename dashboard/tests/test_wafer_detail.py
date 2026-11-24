"""TASK 18-B-10A.2: Metadata caption removal and Wafer detail (Wafer × 노광 × PEB).

The Wafer detail checks run on a byte copy of the operational DB (real multi-condition Wafers with
stored analysis). The operational DB itself is only hashed before/after and never opened for writing.
"""
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,os,shutil,sqlite3,tempfile,unittest

import streamlit as st
from streamlit.testing.v1 import AppTest

from database.init_db import DEFAULT_DB
from dashboard.tests.test_final_ui import FinalUIBase,APP

GONE="기존 Wafer 공정 기록"


class MetadataCaptionTests(FinalUIBase):
    def setUp(self):
        super().setUp()
        from database.migrate_v5 import migrate_database_v5
        self.assertEqual(migrate_database_v5(self.f.db)["version"],5)
        self.uploads=[]
        uploader=patch("streamlit.file_uploader",side_effect=lambda *a,**kw:[] if kw.get("disabled") else self.uploads)
        uploader.start();self.addCleanup(uploader.stop)

    def test_01_existing_wafer_caption_removed_inputs_kept(self):
        from database.tests.test_image_registry import png_bytes
        content=png_bytes(pixel=171)
        self.uploads=[SimpleNamespace(name="w1_12_c.png",getvalue=lambda:content)]   # W1 has a Wafer process record.
        at=self.app("pages/data_control.py")
        texts=[c.value for c in at.caption]+[m.value for m in at.markdown]
        self.assertFalse(any(GONE in t for t in texts))
        self.assertFalse(any("이 Batch의 실제 노광을 직접 입력하세요" in t for t in texts))
        labels=[n.label for n in at.number_input]
        self.assertIn("노광 조건",labels);self.assertIn("PEB Time (s, 선택)",labels)
        self.assertFalse(any("노광 단위" in w.label for w in [*at.selectbox,*at.text_input,*at.number_input]))
        # Missing exposure still blocks registration; entering it unblocks (policy unchanged).
        self.assertTrue(at.button(key="sqlite_register").disabled)
        next(n for n in at.number_input if n.label=="노광 조건").set_value(75).run()
        self.assertFalse(at.button(key="sqlite_register").disabled)
        source=Path(__file__).resolve().parents[1].joinpath("components","sqlite_input.py").read_text(encoding="utf-8")
        self.assertNotIn(GONE,source)


@unittest.skipUnless(DEFAULT_DB.is_file(),"operational DB copy needed")
class WaferDetailCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op_sha=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest()
        cls.tmp=Path(tempfile.mkdtemp(prefix="wafer_detail_"))
        cls.db=cls.tmp/"ida_copy.sqlite3"
        shutil.copyfile(DEFAULT_DB,cls.db)
        cls.env=patch.dict(os.environ,{"IDA_DB_PATH":str(cls.db),"IDA_STORAGE_ROOT":str(cls.tmp/"raw"),
                                       "IDA_STAGING_ROOT":str(cls.tmp/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(cls.tmp/"out")})
        cls.env.start()
        cls.conn=sqlite3.connect(f"file:{cls.db}?mode=ro",uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close();cls.env.stop();shutil.rmtree(cls.tmp,ignore_errors=True)
        assert hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest()==cls.op_sha,"operational DB changed"

    # Independent SQL (effective process = Batch record first, else Wafer record).
    def expected(self,wafer,exposure,peb):
        rows=self.conn.execute(
            "SELECT b.batch_id,r.exposure_value,b.peb_temperature_c,r.exposure_unit FROM upload_batches b JOIN wafers w ON w.wafer_id=b.wafer_id "
            "JOIN process_runs r ON r.process_run_id=COALESCE(b.process_run_id,w.process_run_id) WHERE b.wafer_id=?",(wafer,)).fetchall()
        fmt=lambda v,none:none if v is None else f"{v:g}"
        # TASK18-B-13A.2 display rule, re-implemented here: unrecorded unit -> mJ/cm², recorded unit as stored.
        label=lambda e,u:"미확인" if e is None else f"{e:g} "+(u if u else "mJ/cm²")
        ids=[b for b,e,p,u in rows if label(e,u)==exposure and fmt(p,"—")==peb]
        marks=",".join("?"*len(ids))
        images=self.conn.execute(f"SELECT COUNT(*) FROM images WHERE batch_id IN ({marks})",ids).fetchone()[0]
        analysed=self.conn.execute(f"SELECT COUNT(DISTINCT a.image_id) FROM analysis_image_runs a JOIN images i ON i.image_id=a.image_id "
                                   f"WHERE i.batch_id IN ({marks})",ids).fetchone()[0]
        mapped=self.conn.execute(f"SELECT COUNT(*) FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                                 f"WHERE i.batch_id IN ({marks})",ids).fetchone()[0]
        return set(ids),images,analysed,mapped

    def conditions(self):
        from dashboard.sqlite_provider import runtime_provider
        from dashboard.components.sqlite_input import analysed_per_batch_wafer,condition_accumulation
        provider=runtime_provider()
        return condition_accumulation(provider.batch_wafers(),analysed_per_batch_wafer(provider))

    def open_detail(self,wafer,exposure,peb):
        at=AppTest.from_file(APP,default_timeout=60)
        at.session_state["selected_sqlite_wafer"]={"wafer_id":wafer,"exposure":exposure,"peb":peb}
        at.run();at.switch_page("pages/wafer_detail.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        return at

    def test_02_wafer_cell_click_passes_exact_row(self):
        table=self.conditions().rename(columns={"Wafer":"Wafer ID"})
        target=next(i for i,(w,_) in enumerate(zip(table["Wafer ID"],table.index)) if (table["Wafer ID"]==w).sum()>1)
        row=table.iloc[target]
        original=st.dataframe
        def fake(data=None,*args,**kwargs):
            result=original(data,*args,**kwargs)
            if kwargs.get("key")=="dcc_wafer_conditions":
                return {"selection":{"rows":[],"columns":[],"cells":[(target,"Wafer ID")]}}
            return result
        at=AppTest.from_file(APP,default_timeout=60).run()
        with patch("streamlit.dataframe",side_effect=fake):
            at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        self.assertEqual(at.session_state["selected_sqlite_wafer"],
                         {"wafer_id":row["Wafer ID"],"exposure":row["노광"],"peb":row["PEB (s)"]})
        self.assertEqual(at.session_state["active_page_key"],"wafer_detail")
        self.assertEqual(at.title[0].value,"Wafer 상세")

    def test_03_non_wafer_cell_does_not_navigate(self):
        original=st.dataframe
        def fake(data=None,*args,**kwargs):
            result=original(data,*args,**kwargs)
            if kwargs.get("key")=="dcc_wafer_conditions":
                return {"selection":{"rows":[],"columns":[],"cells":[(0,"노광")]}}
            return result
        at=AppTest.from_file(APP,default_timeout=60).run()
        with patch("streamlit.dataframe",side_effect=fake):
            at.switch_page("pages/data_control.py").run()
        self.assertEqual(at.session_state["active_page_key"],"data_control")

    def test_04_each_condition_matches_sqlite_and_never_mixes(self):
        table=self.conditions()
        multi=[w for w in set(table["Wafer"]) if (table["Wafer"]==w).sum()>1]
        self.assertTrue(multi,"copy should contain a Wafer with several conditions")
        wafer=sorted(multi)[0]
        rows=table[table["Wafer"]==wafer].to_dict("records")
        seen=[]
        for row in rows:
            ids,images,analysed,mapped=self.expected(wafer,row["노광"],row["PEB (s)"])
            self.assertEqual((len(ids),images,analysed),(row["Batch 수"],row["등록 이미지 수"],row["분석 결과 이미지 수"]))
            at=self.open_detail(wafer,row["노광"],row["PEB (s)"])
            values=" ".join(m.value for m in at.main.markdown)
            for text in (wafer,f"{len(ids)}개",f"{images}장",f"{analysed}장"):
                self.assertIn(">"+text+"<",values,text)
            self.assertIn("미입력" if row["PEB (s)"]=="—" else row["PEB (s)"],values)
            buttons={b.key[len("wafer_detail_batch_"):] for b in at.button if (b.key or "").startswith("wafer_detail_batch_")}
            self.assertEqual(buttons,ids)                                 # Exactly this condition's Batches.
            matrix=next(d.value for d in at.dataframe if "Die ID" in d.value.columns)
            self.assertEqual(list(matrix.columns),["Die ID","lt","c","rb"])
            self.assertEqual(len(matrix),24)
            registered=sum(int(v.split(" ")[1]) for col in ("lt","c","rb") for v in matrix[col] if v!="미등록")
            self.assertEqual(registered,mapped)
            self.assertEqual(at.selectbox(key="wafer_detail_condition_"+wafer).value,(row["노광"],row["PEB (s)"]))
            seen.append(ids)
        for a in range(len(seen)):
            for b in range(a+1,len(seen)):
                self.assertFalse(seen[a]&seen[b])                         # Conditions are disjoint.

    def test_05_batch_click_opens_batch_detail_and_ia_preset(self):
        table=self.conditions()
        row=table.iloc[0]
        at=self.open_detail(row["Wafer"],row["노광"],row["PEB (s)"])
        key=next(b.key for b in at.button if (b.key or "").startswith("wafer_detail_batch_"))
        at.button(key=key).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["active_page_key"],"batch_detail")
        self.assertEqual(at.session_state["selected_sqlite_batch_id"],key[len("wafer_detail_batch_"):])
        self.assertEqual(at.title[0].value,"Batch 상세")
        at=self.open_detail(row["Wafer"],row["노광"],row["PEB (s)"])
        at.button(key="wafer_detail_open_ia").click().run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        self.assertEqual(at.session_state["active_page_key"],"inspection_analysis")
        self.assertEqual(at.session_state["_wafer_selected_SQLite"],row["Wafer"])
        at=self.open_detail(row["Wafer"],row["노광"],row["PEB (s)"])
        at.button(key="wafer_detail_back").click().run()
        self.assertEqual(at.session_state["active_page_key"],"data_control")

    def test_06_unknown_condition_falls_back_without_error(self):
        at=self.open_detail("W1","999","—")
        self.assertTrue(any("선택한 노광·PEB 조건의 Batch가 없습니다" in w.value for w in at.warning))
        at=self.open_detail("NOPE","1","—")
        self.assertTrue(any("선택한 SQLite Wafer가 없습니다" in i.value for i in at.info))


if __name__=="__main__":
    unittest.main()
