"""SQLite Wafer detail (TASK18-B-10A.2): one Wafer × 노광 × PEB condition, read-only, Batch detail layout."""
import sqlite3
import streamlit as st
from dashboard.components.layout import page_title
from dashboard.navigation import navigate
from dashboard.sqlite_provider import runtime_provider
from dashboard.components.sqlite_input import analysed_per_batch_wafer,condition_batches
from dashboard.components.sqlite_wafer_detail import (wafer_conditions,condition_label,summary_cards,batch_list,
                                                      image_status,analysis_status,analysis_overview,ia_context)

# Clear the Wafer table's cell selection so returning to Data Home does not reopen this page.
table_key=st.session_state.pop("wafer_table_to_clear",None)
if table_key:
    st.session_state.pop(table_key,None)

selected=st.session_state.get("selected_sqlite_wafer") or {}
wafer_id=selected.get("wafer_id")
with st.container(key="sqlite_wafer_header"):
    title,back=st.columns([4,1],vertical_alignment="top")
    with title:
        page_title("Wafer 상세","SQLite 등록·분석 저장 결과 조회 · 선택한 노광·PEB 조건만 표시","A / WAFER DETAIL")
    if back.button("← Data Home",width="stretch",key="wafer_detail_back"):
        navigate("data_control")

try:
    provider=runtime_provider()
    batches=provider.batch_wafers()   # One row per Batch × Wafer (v6 multi-Wafer Batches split correctly).
    conditions=wafer_conditions(batches,wafer_id) if wafer_id else []
except (sqlite3.Error,OSError,RuntimeError,ValueError) as exc:
    st.error("SQLite Wafer 조회 실패 · 값을 0이나 예시로 대체하지 않습니다. "+str(exc))
    st.stop()

if not wafer_id or not conditions:
    st.info("선택한 SQLite Wafer가 없습니다. Data Home의 Wafer별 공정조건 표에서 Wafer ID를 선택하세요.")
    st.stop()

current=(selected.get("exposure"),selected.get("peb"))
if current not in conditions:
    st.warning("선택한 노광·PEB 조건의 Batch가 없습니다. 이 Wafer의 다른 조건을 선택하세요.")
    current=conditions[0]
# Same Wafer, different conditions are separate views; they are never merged.
choice=st.selectbox("공정 조건",conditions,index=conditions.index(current),format_func=lambda c:condition_label(*c),
                    key="wafer_detail_condition_"+wafer_id,
                    help="같은 Wafer라도 노광·PEB 조건이 다르면 따로 조회합니다. 조건을 합산하지 않습니다.")
if choice!=current:
    st.session_state["selected_sqlite_wafer"]={"wafer_id":wafer_id,"exposure":choice[0],"peb":choice[1]}
exposure,peb=choice

try:
    chosen=condition_batches(batches,wafer_id,exposure,peb)
    analysed=analysed_per_batch_wafer(provider)
    images=provider.condition_images(chosen["batch_id"].tolist(),wafer_id)   # This Wafer only.
    latest=provider.latest_analysis(wafer_id)
except (sqlite3.Error,OSError,RuntimeError,ValueError) as exc:
    st.error("SQLite Wafer 상세 조회 실패. "+str(exc))
    st.stop()

image_ids=images["image_id"].tolist()
overview=analysis_overview(image_ids,latest)
analysed_ids={i for i in image_ids if i in latest}
summary_cards(wafer_id,chosen,images,overview,exposure,peb)

st.subheader("Batch 목록")
st.caption("수량은 이 Wafer 기준입니다 · Batch 이름을 누르면 Batch 상세로 이동하며, Batch 상세는 Batch 전체(포함된 모든 Wafer)를 표시합니다.")
batch_list(chosen,analysed,wafer_id)

left,right=st.columns([1.6,1])
with left:
    image_status(images,analysed_ids)
with right:
    analysis_status(overview)

if st.button("Inspection & Analysis에서 이 Wafer 보기",key="wafer_detail_open_ia",
             help="I&A Wafer Map에 이 Wafer와 노광·PEB 조건을 선택한 상태로 엽니다."):
    navigate("inspection_analysis",**ia_context(wafer_id,chosen))
