import streamlit as st
from dashboard.components.layout import page_title
from dashboard.components.inspection_wafer import wafer_context,render_die_map,render_site_images
from dashboard.navigation import navigate

page_title("Wafer Map","물리적 Wafer → Die → lt/c/rb → 등록 원본 탐색 · 분석 결과는 Inspection & Analysis에서 조회","B / MAP")
if st.button("← Inspection & Analysis",key="wafer_back"):
    navigate("inspection_analysis")
left,right=st.columns([1.1,1.3],gap="medium")
with left,st.container(border=True,key="wafer_browser_map"):
    context=wafer_context("detail_wafer",batch_filter=True)
    selected=render_die_map(context,"detail_wafer_map") if context else None  # Legend is inside (collapsed).
with right,st.container(border=True,key="wafer_browser_site"):
    if selected:
        render_site_images(context,selected)
    else:
        st.info("Wafer를 선택하면 결과가 표시됩니다.")
