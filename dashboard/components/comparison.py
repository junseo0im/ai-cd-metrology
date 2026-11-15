"""Shared 50/50 comparison layout; provider supplies all sample values."""
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,analysis_modes,card_title
from dashboard.components.images import comparison_images
from dashboard.components.charts import comparison_chart
from dashboard.components.tables import data_table


def render(mode):
    labels = "R1 / R2 / R3" if mode=="Repeatability" else "lg / lm / lb"
    description = "같은 physical site의 반복 촬영 비교 UI" if mode=="Repeatability" else "같은 physical site의 조명 비교 UI"
    page_title(mode,description+" · "+labels+" (예시)","B / COMPARISON")
    analysis_modes(mode)
    left,right = st.columns([1,1],gap="large")
    with left,st.container(border=True):
        card_title("이미지 비교 · "+labels,"정적 Mock illustration · 실제 촬영/정렬/보정 없음")
        overlay = st.toggle("Overlay 비교 (예시)",key="comparison_overlay_"+mode)
        comparison_images(mode,overlay=overlay)
    with right,st.container(border=True):
        card_title("결과 및 그래프 · 예시","계산된 repeatability 또는 lighting 성능이 아닙니다.")
        comparison_chart(mode)
        frame = data.images(mode=mode)
        data_table(frame[["label","outer_um_example","inner_um_example","gap_um_example"]].rename(
            columns={"label":"조건 (예시)","outer_um_example":"Outer µm (예시)",
                     "inner_um_example":"Inner µm (예시)","gap_um_example":"Gap µm (예시)"}),
                     hide_index=True,width="stretch")
        st.caption("Identity 연결 예시 · 동일 physical_site_id, 서로 다른 image_id")
        data_table(frame[["image_id","physical_site_id","measurement_run_id"]],
                     hide_index=True,width="stretch")
        st.info("정렬·보정·Mean/Std·통계 검정·실제 분석은 미구현입니다.")
