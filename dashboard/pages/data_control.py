import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import inline_title,card_title
from dashboard.components.pipeline import processing_pipeline,image_status_tiles
from dashboard.components.tables import batch_table
from dashboard.components.data_overview import accumulation_overview,processing_progress
from dashboard.components import sqlite_input
from dashboard.components.analysis_management import render_analysis_management

inline_title("Data Control Center","등록·Batch 조회·분석 대상 미리보기·축적 현황을 확인합니다.")



def toggle_card(key):
    st.session_state["dcc_expanded_card"]=None if st.session_state.get("dcc_expanded_card")==key else key


def control_card_title(key,title,description="",*,separator=False):
    """Title + description on one line; a small zoom icon (same toggle as the former 확대/복원 button)."""
    expanded=st.session_state.get("dcc_expanded_card")==key
    with st.container(key="card_head_"+key,horizontal=True,vertical_alignment="center",gap="small"):
        st.subheader(title,anchor=False,width="content")
        if description:
            st.caption(("|  " if separator else "")+description,width="stretch")
        else:
            st.space("stretch")
        st.button(":material/close_fullscreen:" if expanded else ":material/open_in_full:",
                  key="dcc_expand_"+key,on_click=toggle_card,args=(key,),type="tertiary",
                  help="기본 2×2 배치로 복원" if expanded else "카드 확대")


# Operational default is SQLite. Mock remains only for explicit internal/test selection
# (session `data_control_selected_source="Mock"`); there is no UI switch and no fallback.
source=st.session_state.get("data_control_selected_source","SQLite")
def render_input_card():
    control_card_title("input","데이터 입력","업로드 → Metadata 확인 → 승인 후 SQLite 등록",separator=True)
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
            with files_col,st.popover("파일 (Mock)",width="stretch"):
                st.file_uploader("Mock 모드에서는 업로드하지 않습니다",accept_multiple_files=True,disabled=True)
        dataset_role = st.radio("Dataset Role",data.DATASET_ROLES,index=data.DATASET_ROLES.index(saved_common.get("dataset_role","EVALUATION")),
                                horizontal=True,key="dataset_role",help="필수 · 학습 Train/Val/Test 분할과 별개입니다.",
                                format_func=lambda role:role+" — "+data.DATASET_ROLE_LABELS[role])
        if source == "SQLite":
            sqlite_input.render_registration(mode,dataset_role,upload_slot=files_col)
        else:
            with st.form("mock_metadata",border=False):
                exposure = st.selectbox("노광 조건 (예시)",[50,60,70,80],index=1,key="input_exposure")
                peb = st.number_input("PEB Time (s, 선택 · Mock)",value=None,step=1.0,
                                      key="input_peb_temperature_c",placeholder="미입력 유지")
                upload_requested = st.form_submit_button("이미지 업로드",width="stretch")
                submitted = st.form_submit_button("Mock Metadata 미리보기",type="primary",width="stretch")
            if upload_requested:
                st.info("이미지 업로드 미연결 · 향후 다중 이미지/ZIP 입력을 연결할 위치입니다. 실제 업로드·저장은 수행하지 않습니다.")
            if submitted:
                st.session_state["mock_metadata_preview"] = data.upload_batch_preview(
                    mode=mode,exposure_example=exposure,dataset_role=dataset_role,peb_temperature_c=peb)
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
    control_card_title("pipeline","데이터 축적 현황","SQLite 등록·분석 이력",separator=True)
    if source=="Mock":
        accumulation_overview()
        processing_pipeline()
        image_status_tiles()
    else:
        sqlite_input.render_overview()


def render_batches_card():
    control_card_title("batches","Batch 목록","Batch를 선택하면 상세 페이지로 이동합니다.",separator=True)
    sqlite_input.render_batches(source=source)


def render_progress_card():
    control_card_title("progress","등록·처리 상태 요약")
    if source=="Mock":
        processing_progress()
    else:
        sqlite_input.render_progress()


def render_analysis_card():
    control_card_title("analysis","분석 실행 관리","대상 확인 → READY 이미지 수동 실행",separator=True)
    render_analysis_management(source)


renderers={"input":render_input_card,"batches":render_batches_card,
           "analysis":render_analysis_card,"pipeline":render_overview_card}
heights={"input":480,"batches":480,"analysis":620,"pipeline":620}  # TASK18-B-10A: top cards fit their default content
widths={"top":[1,1.3],"bottom":[1.45,1]}
expanded=st.session_state.get("dcc_expanded_card")
if expanded is not None and expanded not in renderers:
    st.session_state["dcc_expanded_card"]=None
    expanded=None
if expanded in renderers:
    # Keep every card mounted: keyed widgets/drafts survive expand and restore.
    with st.container(border=True,height="content",key="card_data_"+expanded):
        renderers[expanded]()
    remaining=[key for key in renderers if key!=expanded]
    for start in range(0,len(remaining),2):
        columns=st.columns(2,gap="medium")
        for col,key in zip(columns,remaining[start:start+2]):
            with col,st.container(border=True,height=heights[key],key="card_data_"+key):
                renderers[key]()
else:
    top_left,top_right=st.columns(widths["top"],gap="medium")
    with top_left,st.container(border=True,height=heights["input"],key="card_data_input"):
        render_input_card()
    with top_right,st.container(border=True,height=heights["batches"],key="card_data_batches"):
        render_batches_card()
    bottom_left,bottom_right=st.columns(widths["bottom"],gap="medium")
    with bottom_left,st.container(border=True,height=heights["analysis"],key="card_data_analysis"):
        render_analysis_card()
    with bottom_right,st.container(border=True,height=heights["pipeline"],key="card_data_pipeline"):
        render_overview_card()
