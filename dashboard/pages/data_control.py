import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,card_title
from dashboard.components.pipeline import processing_pipeline,image_status_tiles
from dashboard.components.tables import batch_table
from dashboard.components.data_overview import accumulation_overview,processing_progress
from dashboard.components import sqlite_input

page_title("Data Control Center","입력·축적·Batch 상태를 확인합니다. 분석 기능은 Mock/미연결입니다.","A / DATA CONTROL")



def toggle_card(key):
    st.session_state["dcc_expanded_card"]=None if st.session_state.get("dcc_expanded_card")==key else key


def control_card_title(key,title,description=""):
    title_col,action_col=st.columns([5,1.1],vertical_alignment="center")
    with title_col:
        card_title(title,description)
    action_col.button("복원" if st.session_state.get("dcc_expanded_card")==key else "확대",
                      key="dcc_expand_"+key,on_click=toggle_card,args=(key,),width="stretch",
                      help="선택 카드를 넓게 보기 / 기본 2×2 배치로 복원")


def remember_source():
    st.session_state["data_control_selected_source"] = st.session_state["data_control_source"]


source = st.radio("데이터 소스",["Mock","SQLite"],horizontal=True,key="data_control_source",
                  index=["Mock","SQLite"].index(st.session_state.get("data_control_selected_source","Mock")),
                  on_change=remember_source)
def render_input_card():
    control_card_title("input","데이터 입력","Mock 미리보기 / SQLite 승인 후 등록")
    registration_tab,review_tab = st.tabs(["이미지 등록","파일명 확인·수정"])
    with registration_tab:
        saved_common = st.session_state.get("sqlite_uploads",{}).get("common_metadata",{}) if source=="SQLite" else {}
        mode_col,files_col = st.columns([4,1.3],vertical_alignment="center")
        with mode_col:
            modes=list(sqlite_input.MODE_CODES)
            mode = st.radio("입력 모드",modes,
                            horizontal=True,key="input_mode",on_change=sqlite_input.input_mode_changed,
                            help="필수 · Batch는 등록 작업 단위이며 Wafer는 물리적 촬영 대상입니다.",
                            index=modes.index(saved_common.get("mode","Wafer Batch")))
        if source == "Mock":
            with files_col,st.popover("파일 · 미연결",width="stretch"):
                st.file_uploader("이미지 선택 · UI만 제공",accept_multiple_files=True,disabled=True)
        dataset_role = st.radio("Dataset Role",data.DATASET_ROLES,index=data.DATASET_ROLES.index(saved_common.get("dataset_role","EVALUATION")),
                                horizontal=True,key="dataset_role",help="필수 · 학습 Train/Val/Test 분할과 별개입니다.",
                                format_func=lambda role:role+" — "+data.DATASET_ROLE_LABELS[role])
        if source == "SQLite":
            sqlite_input.render_registration(mode,dataset_role,upload_slot=files_col)
        else:
            with st.form("mock_metadata",border=False):
                exposure = st.selectbox("노광 조건 (예시)",[50,60,70,80],index=1,key="input_exposure")
                upload_requested = st.form_submit_button("이미지 업로드",width="stretch")
                submitted = st.form_submit_button("Mock Metadata 미리보기",type="primary",width="stretch")
            if upload_requested:
                st.info("이미지 업로드 미연결 · 향후 다중 이미지/ZIP 입력을 연결할 위치입니다. 실제 업로드·저장은 수행하지 않습니다.")
            if submitted:
                st.session_state["mock_metadata_preview"] = data.upload_batch_preview(
                    mode=mode,exposure_example=exposure,dataset_role=dataset_role)
                st.success("선택값 미리보기입니다. 실제 Batch 등록·파일 저장은 수행하지 않았습니다.")
            if "mock_metadata_preview" in st.session_state:
                st.json(st.session_state["mock_metadata_preview"],expanded=False)

    with review_tab:
        if source == "SQLite":
            sqlite_input.render_review()
        else:
            items = {item["mock_image_id"]:item for item in data.filename_review_items()}
            review_list,edit_panel = st.columns([1,1.35],gap="medium")
            with review_list:
                selected = st.radio("확인 대상 · Mock",list(items),key="filename_review_image",
                    format_func=lambda key:items[key]["filename"]+" — "+items[key]["review_reason"])
                st.caption("실제 이미지 없음 · 자동 파싱/정상 판정 없음")
            item = items[selected]
            previews = st.session_state.setdefault("filename_review_previews",{})
            values = previews.get(selected,item["metadata"])
            with edit_panel:
                st.caption("선택 이미지 Metadata · 미리보기 전용")
                die_options = [None,*data.REVIEW_DIE_CODES]
                position_options = [None,"lt","c","rb"]
                with st.form("filename_review_"+selected,border=False):
                    wafer_col,die_col = st.columns(2)
                    wafer = wafer_col.text_input("Wafer ID",value=values["wafer_id"] or "",
                                                 key="review_wafer_"+selected,placeholder="필수 · 예: W1",help="필수 · 촬영 Wafer ID")
                    die = die_col.selectbox("Die ID",die_options,index=die_options.index(values["die_code"]),
                        format_func=lambda value:value or "확인 필요",key="review_die_"+selected,help="필수 · 11~18 / 21~28 / 31~38")
                    position_col,defect_col = st.columns(2)
                    position = position_col.selectbox("촬영 위치",position_options,
                        index=position_options.index(values["position_code"]),
                        format_func=lambda value:value or "확인 필요",key="review_position_"+selected,help="필수 · lt / c / rb")
                    legacy = defect_col.text_input("결함 코드 (선택)",value=values["legacy_defect_code"] or "",
                                                    key="review_defect_"+selected,placeholder="선택 · 미상은 공란")
                    confirmed = st.form_submit_button("수정 내용 확인",width="stretch")
                if confirmed:
                    previews[selected] = {
                        "mock_image_id":selected,"original_filename":item["filename"],
                        "wafer_id":wafer or None,"die_code":die,"position_code":position,
                        "legacy_defect_code":legacy or None,"review_status":"preview_only",
                        "review_reason":item["review_reason"],
                        "saved_to_database":False,"file_renamed":False}
                    st.success("수정값을 Session State 미리보기로만 유지합니다.")
                if selected in previews:
                    st.json(previews[selected],expanded=False)


def render_overview_card():
    control_card_title("pipeline","데이터 축적 현황")
    if source=="Mock":
        accumulation_overview()
        processing_pipeline()
        image_status_tiles()
    else:
        sqlite_input.render_overview()


def render_batches_card():
    control_card_title("batches","Batch 목록 및 전체 이미지 조회","Batch를 선택하면 기존 상세페이지로 이동합니다.")
    sqlite_input.render_batches(source=source)


def render_progress_card():
    control_card_title("progress","등록·처리 상태 요약")
    if source=="Mock":
        processing_progress()
    else:
        sqlite_input.render_progress()


renderers={"input":render_input_card,"pipeline":render_overview_card,
           "batches":render_batches_card,"progress":render_progress_card}
expanded=st.session_state.get("dcc_expanded_card")
if expanded in renderers:
    # Keep every card mounted: keyed widgets/drafts survive expand and restore.
    with st.container(border=True,height="content",key="card_data_"+expanded):
        renderers[expanded]()
    remaining=[key for key in renderers if key!=expanded]
    for start in range(0,len(remaining),2):
        columns=st.columns(2,gap="medium")
        for col,key in zip(columns,remaining[start:start+2]):
            with col,st.container(border=True,height=480 if key in ("input","pipeline") else 320,key="card_data_"+key):
                renderers[key]()
else:
    top_left,top_right=st.columns(2,gap="medium")
    with top_left,st.container(border=True,height=480,key="card_data_input"):
        render_input_card()
    with top_right,st.container(border=True,height=480,key="card_data_pipeline"):
        render_overview_card()
    bottom_left,bottom_right=st.columns(2,gap="medium")
    with bottom_left,st.container(border=True,height=320,key="card_data_batches"):
        render_batches_card()
    with bottom_right,st.container(border=True,height=320,key="card_data_progress"):
        render_progress_card()
