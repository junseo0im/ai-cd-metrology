import streamlit as st
from dashboard.components.layout import page_title,placeholder,detail_navigation

page_title("Critical 기준","향후 측정·결함별 판정 기준과 심각도를 기록할 공간입니다.","CRITERIA / PLACEHOLDER")
measurement,defect=st.tabs(["측정 Critical 기준","결함 Critical 기준"])
with measurement:
    placeholder("측정 Critical 기준 · 미확정",
                "허용 편차율, Thickening/Narrowing 기준, 측정 심각도를 기록할 공간입니다. 실제 Threshold와 최종 판정 로직은 구현하지 않습니다.")
with defect:
    placeholder("결함 Critical 기준 · 미확정",
                "Open / Short / Scratch / Particle의 판정 기준과 심각도를 기록할 공간입니다. 확정된 기준값과 판정 결과는 없습니다.")
detail_navigation()
