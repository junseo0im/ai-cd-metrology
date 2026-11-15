"""Mock/SQLite presentation contract; all SQLite I/O uses explicit temporary fixtures."""
from pathlib import Path
import unittest

import dashboard.tests.test_sqlite_input as fixtures
from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider


class UnifiedDataControlTests(unittest.TestCase):
    setUp=fixtures.SQLiteInputTests.setUp
    app=fixtures.SQLiteInputTests.app
    uploaded=fixtures.SQLiteInputTests.uploaded
    confirm_process=fixtures.SQLiteInputTests.confirm_process
    seed_registered_image=fixtures.SQLiteInputTests.seed_registered_image
    duplicate_choices=fixtures.SQLiteInputTests.duplicate_choices

    def cards(self,at):
        result={}
        def visit(node,path=()):
            proto=getattr(node,"proto",None)
            identity=getattr(proto,"id","")
            if "card_data_" in identity:
                key=identity[identity.index("card_data_"):]
                result[key]=(path,proto.height_config.pixel_height,node)
            children=getattr(node,"children",{})
            if isinstance(children,dict):
                for key,child in children.items():
                    visit(child,path+(key,))
        visit(at._tree)
        return result

    def test_same_four_cards_titles_tabs_and_modes_across_sources(self):
        original=mock.catalog()
        at=self.app()
        sqlite_cards=self.cards(at)
        sqlite_headers=[row.value for row in at.subheader]
        sqlite_tabs=[tab.label for tab in at.tabs]
        sqlite_modes=list(at.radio(key="input_mode").options)
        self.assertFalse(at.get("progress"))  # Never reuse the Mock 60% for real data.
        self.assertTrue(at.button(key="sqlite_register").disabled)
        at.radio(key="data_control_source").set_value("Mock").run()
        self.assertFalse(at.exception)
        mock_cards=self.cards(at)
        self.assertEqual(set(mock_cards),{"card_data_input","card_data_pipeline","card_data_batches","card_data_progress"})
        self.assertEqual({key:value[:2] for key,value in sqlite_cards.items()},
                         {key:value[:2] for key,value in mock_cards.items()})
        self.assertEqual(sqlite_headers,[row.value for row in at.subheader])
        self.assertEqual(sqlite_tabs,[tab.label for tab in at.tabs])
        self.assertEqual(sqlite_tabs,["이미지 등록","파일명 확인·수정"])
        self.assertEqual(sqlite_modes,list(at.radio(key="input_mode").options))
        self.assertEqual(sqlite_modes,["Wafer Batch","Repeatability","Lighting","Single Image"])
        self.assertEqual(len(at.get("progress")),1)
        self.assertTrue(any('class="pipeline"' in row.value for row in mock_cards["card_data_pipeline"][2].markdown))
        self.assertTrue(any('class="image-tiles"' in row.value for row in mock_cards["card_data_pipeline"][2].markdown))
        self.assertEqual(mock.catalog(),original)
        self.assertTrue(runtime_provider().batches().empty)

    def test_real_summary_all_batch_rows_bounded_scroll_and_detail_route(self):
        for index in range(7):
            self.seed_registered_image(pixel=40+index)
        at=self.app()
        cards=self.cards(at)
        overview=cards["card_data_pipeline"][2]
        self.assertTrue(any("**전체 Batch** 7" in row.value and "**전체 Image** 7" in row.value
                            for row in overview.markdown))
        self.assertTrue(any("Wafer Batch 7" in row.value for row in overview.caption))
        self.assertTrue(any('class="pipeline"' in row.value for row in overview.markdown))
        self.assertTrue(any('class="image-tiles"' in row.value for row in overview.markdown))
        status=cards["card_data_progress"][2]
        self.assertTrue(any("등록 완료 7장" in row.value for row in status.success))
        table=status.dataframe[0].value
        self.assertEqual(table["상태"].tolist(),["완료","미연결","미연결","미연결"])
        batch_buttons=lambda:[row for row in at.button if row.key and row.key.startswith("sqlite_batch_")]
        self.assertEqual(len(batch_buttons()),7)
        self.assertFalse(any(row.key and row.key.startswith("batch_table_page_") for row in at.selectbox))
        expected=runtime_provider().batches().sort_values("created_at",ascending=False,kind="stable")["batch_id"].tolist()
        self.assertEqual([row.key.removeprefix("sqlite_batch_") for row in batch_buttons()],expected)
        def find_table(node):
            if str(getattr(getattr(node,"proto",None),"id","")).endswith("-batch_table"):
                return node
            children=getattr(node,"children",{})
            if isinstance(children,dict):
                for child in children.values():
                    found=find_table(child)
                    if found is not None:
                        return found
            return None
        self.assertEqual(find_table(at._tree).proto.height_config.pixel_height,160)
        # The oldest row, previously hidden on page 2, still opens its real detail.
        chosen=batch_buttons()[-1]
        identity=chosen.key.removeprefix("sqlite_batch_")
        self.assertNotIn(identity,chosen.label)
        self.assertIn("Batch",chosen.label)
        chosen.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value,"Batch 상세")
        self.assertEqual(at.session_state["selected_sqlite_batch_id"],identity)
        self.assertEqual(at.session_state["batch_detail_source"],"SQLite")

    def test_details_collapsed_and_duplicate_policy_survives_source_switch(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("w1_12_rb.png",80)]
        at=self.app()
        self.confirm_process(at)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        plan=at.session_state["sqlite_registration_plan"]
        self.assertEqual(plan["selected_count"],1)
        for title in ("공통 Metadata","검증 목록","전체 등록 이미지 조회","이미지 상세 정보"):
            expand=next(row for row in at.expander if row.label==title)
            self.assertFalse(expand.proto.expanded)
        self.duplicate_choices(at)[0].select("HOLD").run()
        self.assertTrue(at.button(key="sqlite_register").disabled)
        at.radio(key="data_control_source").set_value("Mock").run()
        self.assertFalse(at.exception)
        self.assertFalse(any(row.key=="sqlite_register" for row in at.button))
        at.radio(key="data_control_source").set_value("SQLite").run()
        self.assertFalse(at.exception)
        self.assertEqual(self.duplicate_choices(at)[0].value,"HOLD")
        self.assertTrue(at.button(key="sqlite_register").disabled)
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sqlite_last_result"]["registered_count"],2)
        self.assertEqual(runtime_provider().image_count(),3)

    def test_mock_single_preview_and_four_filters_do_not_touch_sqlite(self):
        at=self.app()
        at.radio(key="data_control_source").set_value("Mock").run()
        at.radio(key="input_mode").set_value("Single Image").run()
        next(row for row in at.button if row.label=="Mock Metadata 미리보기").click().run()
        self.assertFalse(at.exception)
        preview=at.session_state["mock_metadata_preview"]
        self.assertEqual(preview["mode"],"Single Image")
        self.assertFalse(preview["saved_to_database"])
        filters=next(row for row in at.selectbox if row.label=="등록 유형 필터")
        for mode,count in (("Wafer Batch",72),("Repeatability",3),("Lighting",3),("Single Image",0)):
            at.selectbox(key=filters.key).select(mode).run()
            self.assertFalse(at.exception)
            self.assertTrue(any(f"등록 이미지 {count}장 · Mock"==row.value for row in at.caption))
        self.assertTrue(runtime_provider().batches().empty)
        self.assertFalse((self.root/"raw").exists())


if __name__=="__main__":
    unittest.main()
