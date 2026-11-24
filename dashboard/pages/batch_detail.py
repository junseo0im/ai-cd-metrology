import sqlite3
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,select_batch,metric_summary
from dashboard.components.charts import status_donut
from dashboard.navigation import navigate
from dashboard.components.tables import data_table

if st.session_state.get("batch_detail_source")=="SQLite":
    from dashboard.components.sqlite_input import render_registered_images
    from dashboard.sqlite_provider import runtime_provider
    from dashboard.components.sqlite_batch_detail import batch_metadata,batch_processing,wafer_metadata,wafer_composition
    with st.container(key="sqlite_batch_header"):
        title,back=st.columns([4,1],vertical_alignment="top")
        with title:
            page_title("Batch 상세","SQLite 등록 Metadata · 분석 결과는 Inspection & Analysis에서 조회","A / BATCH DETAIL")
        if back.button("← Data Home",width="stretch"):
            st.session_state.pop("batch_detail_source",None)
            navigate("data_control")
    provider=runtime_provider()
    selected=st.session_state.get("selected_sqlite_batch_id")
    record=provider.batch_record(selected)
    if record is None:
        st.info("선택한 SQLite Batch가 없습니다.")
    else:
        batch_metadata(record)
        if int(record.get("wafer_count",1) or 1)>1:
            wafer_composition(provider,record)   # v6: several Wafers in this Batch.
        batch_processing(provider,record)
        st.subheader("Batch 소속 이미지")
        render_registered_images(provider,selected,detailed=True)
        if st.session_state.pop("sqlite_batch_name_saved",None)==selected:
            st.success("Batch 이름을 DB에 저장했습니다. 이름이 비어 있으면 자동 표시명을 사용합니다.")
        with st.expander("Batch 이름 수정"):
            with st.form("sqlite_batch_name_form_"+selected,border=False):
                name=st.text_input("Batch 이름 (선택)",value=record["batch_name"],
                    placeholder="예: W5 중앙 반복촬영 3회",key="sqlite_saved_batch_name_"+selected)
                st.caption("공란으로 저장하면 자동 Batch 표시명으로 돌아갑니다. ID·소속 이미지는 변경하지 않습니다.")
                save_name=st.form_submit_button("Batch 이름 저장")
            if save_name:
                try:
                    provider.rename_batch(selected,name)
                except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
                    st.error("Batch 이름 저장 실패: "+str(exc))
                else:
                    st.session_state["sqlite_batch_name_saved"]=selected
                    st.rerun()
        wafer_metadata(record)
    st.stop()

page_title("Batch 상세","Processing Matrix와 상태 비율을 확인하는 Mock 화면입니다.","A / BATCH DETAIL")
if "invalid_batch_link" in st.session_state:
    st.error("존재하지 않는 예시 Batch ID입니다. Data Home에서 다시 선택하세요.")
    st.page_link("pages/data_control.py",label="Data Home")
    st.stop()
batch_id = select_batch("detail_batch")
row = data.batch(batch_id)
st.caption(row["wafer_id"]+" · "+row["mode"]+" · 노광 "+str(row["exposure"])+" (전부 예시)")
left,right = st.columns([1.6,1])
with left,st.container(border=True):
    st.subheader("Processing Matrix · 예시")
    data_table(data.processing_matrix(batch_id),hide_index=True,width="stretch")
    st.caption("행은 이미지 ID에 연결된 예시 상태입니다. 실제 처리 이력이 아닙니다.")
with right,st.container(border=True):
    st.subheader("원형 상태 비율 · 예시")
    status_donut(batch_id)
metric_summary(batch_id)
if st.button("← Data Home"):
    navigate("data_control")
