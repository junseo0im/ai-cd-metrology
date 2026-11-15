import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,analysis_modes,select_batch,select_image
from dashboard.components.mlflow import render_mlflow
from dashboard.components.charts import exposure_chart
from dashboard.components.inspection_wafer import inspection_wafer,reset_selection
from dashboard.components.images import image_panel
from dashboard.components.tables import data_table
from dashboard.navigation import navigate

page_title("Inspection & Analysis","Pattern 선택, 집계 변화, 이미지와 Batch 결과를 연결합니다. 모든 표시값은 예시입니다.","B / INSPECTION")
controls,selection = st.columns([2,1],vertical_alignment="bottom")
with controls:
    analysis_modes("Wafer Batch")
with selection:
    batch_id = select_batch("inspection_batch","Wafer Batch")
left,right = st.columns(2,gap="medium")
with left,st.container(border=True,height=355,key="card_ia_map"):
    title_col,reset_col,select_col = st.columns([1.1,1.6,1],vertical_alignment="center")
    title_col.subheader("Wafer Map")
    reset_col.button("전체 보기 / 선택 초기화",key="inspection_map_reset",width="stretch",
                     on_click=reset_selection,args=(batch_id,))
    selector_slot = select_col.container()
    inspection_wafer(batch_id)
    with selector_slot,st.popover("이미지 선택",width="stretch"):
        image_id = select_image(batch_id,"ia_image")
        st.caption("위치별 실이미지 미연결 · 각 Die의 lt/c/rb는 같은 기존 Mock 이미지를 공유합니다.")
with right,st.container(border=True,height=355,key="card_ia_trend"):
    trend_title,simulation = st.columns([4,1],vertical_alignment="center")
    trend_title.subheader("공정 조건별 그래프 · 누적 변화")
    if simulation.button("Simulation",width="stretch"):
        navigate("simulation",_restore_ia_overlay=st.session_state.get("ia_overlay",True))
    graph_tab,mlflow_tab = st.tabs(["공정 조건별 그래프","MLflow"])
    with graph_tab:
        exposure_chart("ia_exposure",height=205)
    with mlflow_tab:
        render_mlflow()

with st.container(border=True,height=350,key="card_ia_results"):
    image_col,summary_col,comparison_col = st.columns([1.1,1,1.1],gap="large")
    with image_col:
        image_title,overlay_control = st.columns([1.2,1],vertical_alignment="center")
        image_title.subheader("선택 이미지")
        with overlay_control:
            overlay = st.toggle("Overlay 예시",value=st.session_state.pop("_restore_ia_overlay",True),key="ia_overlay")
        image_panel(image_id,overlay,caption=False)
        st.caption(image_id+" · "+("Overlay 예시" if overlay else "원본 영역 예시")+" / Mock")
        _,detail_button,_ = st.columns([1,1,1])
        if detail_button.button("이미지 상세 열기",key="ia_detail",width="content"):
            navigate("image_analysis",selected_image_id=image_id,selected_batch_id=batch_id)
    with summary_col:
        st.subheader("Batch 결과 표 · 예시")
        data_table(data.batch_summary_table(batch_id),height=228,row_height=27)
        st.caption(batch_id+" · Line/Edge는 UI 예시 항목입니다.")
    with comparison_col:
        st.subheader("계측·설계 비교 · 미연결")
        data_table(data.design_comparison_table(),height=147,row_height=29)
        st.caption("Calibration 적용값·설계치 미확정 · 오차율 미계산")
        st.info("표시 구조만 제공하며 실제 계측·설계 비교는 수행하지 않습니다.")
