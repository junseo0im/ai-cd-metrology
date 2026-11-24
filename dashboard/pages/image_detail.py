"""Minimal SQLite image detail: native original + stored metadata, no analysis."""
from html import escape
import streamlit as st
from dashboard.sqlite_provider import runtime_provider
from dashboard.components.layout import page_title
from dashboard.components.sqlite_input import render_original_preview
from dashboard.components.sqlite_batch_detail import display_value
from dashboard.navigation import navigate

# Clear only the prior table selection so Back does not immediately reopen it.
table_key=st.session_state.pop("sqlite_image_table_to_clear",None)
if table_key:
    st.session_state.pop(table_key,None)
provider=runtime_provider()
image_id=st.query_params.get("image_id",st.session_state.get("selected_sqlite_image_id"))
expected_batch=st.query_params.get("batch_id",st.session_state.get("selected_sqlite_image_batch_id"))
record=provider.image_record(image_id,batch_id=expected_batch)
batch_id=record["batch_id"] if record else expected_batch
with st.container(key="sqlite_image_header"):
    title,back=st.columns([4,1],vertical_alignment="top")
    with title:
        page_title("Image 상세","SQLite 원본·촬영 Metadata 조회 · 분석 결과는 Inspection & Analysis에서 조회","A / IMAGE DETAIL")
    if back.button("← Batch 상세로 돌아가기",disabled=not batch_id,width="stretch"):
        st.query_params.clear()
        navigate("batch_detail",selected_sqlite_batch_id=batch_id,batch_detail_source="SQLite")
if record is None:
    st.warning("선택한 SQLite Image를 찾을 수 없습니다. Batch 이미지 목록에서 Image ID를 다시 선택하세요.")
    if not batch_id and st.button("Data Control Center로 이동"):
        st.query_params.clear()
        navigate("data_control",data_control_selected_source="SQLite")
    st.stop()

st.session_state["selected_sqlite_image_id"]=record["image_id"]
st.session_state["selected_sqlite_image_batch_id"]=record["batch_id"]
st.session_state["selected_sqlite_batch_id"]=record["batch_id"]
st.session_state["data_control_selected_source"]="SQLite"
left,right=st.columns([1.5,1])
with left,st.container(border=True,key="sqlite_image_original"):
    st.subheader("등록 원본 이미지")
    st.caption(f"{record['width_px']} × {record['height_px']} px · 원본 비율 유지")
    render_original_preview(provider,record)
with right,st.container(border=True,key="sqlite_image_metadata"):
    st.subheader("Image Metadata")
    st.markdown("#### A. 이미지 식별·소속 정보")
    st.caption("Image ID · 전체 복사 가능")
    st.code(record["image_id"],language=None,wrap_lines=True)
    groups=[
        [("원본 파일명","original_filename"),("Batch ID","batch_id"),
         ("Batch 이름","batch_name"),("자동 Batch 번호","automatic_batch_display_name"),
         ("Wafer ID","wafer_id"),("Die ID","die_code"),("Pattern Site","position_code"),
         ("Physical Site ID","physical_site_id")],
        [("노광","exposure_value"),("노광 단위","exposure_unit"),("PEB (s)","peb_temperature_c"),("배율","magnification"),
         ("조명","illumination_setting"),("Auto Exposure","auto_exposure"),
         ("반복 번호","repeat_index"),("촬영 일시","captured_at")],
        [("Upload Mode","upload_mode"),("Dataset Role","dataset_role"),("등록 일시","created_at"),
         ("Metadata 상태","metadata_status"),("SHA-256","sha256")]]
    for index,fields in enumerate(groups):
        if index:
            st.markdown("#### "+("B. 촬영 조건" if index==1 else "C. 등록·관리 정보"))
        rows="".join("<tr><th scope='row'>"+escape(label)+"</th><td>"+
                     escape(display_value(record.get(key)))+"</td></tr>" for label,key in fields)
        st.markdown('<table class="sqlite-image-kv"><tbody>'+rows+"</tbody></table>",unsafe_allow_html=True)

st.info("이 화면은 측정·추론을 실행하지 않습니다 · 저장된 분석 결과는 Inspection & Analysis에서 조회합니다.")
