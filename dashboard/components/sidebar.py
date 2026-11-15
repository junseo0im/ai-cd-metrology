"""Persistent common navigation with A/B contextual links."""
import streamlit as st
from dashboard.navigation import A_PAGES,B_PAGES,COMMON,PAGE_SPECS,page_data_source


def render(pages,active):
    with st.sidebar:
        st.markdown("### IDA Workspace")
        st.caption("Inspection & data orchestration")
        if active in A_PAGES:
            st.markdown("**A / DATA CONTROL**")
            st.page_link(pages["data_control"],label="Data Home",icon=":material/folder_open:")
            st.page_link(pages["calibration"],label="Calibration",icon=":material/straighten:")
        elif active in B_PAGES:
            st.markdown("**B / INSPECTION & ANALYSIS**")
            for key,label in [("inspection_analysis","I&A Home"),("wafer_map","Wafer Map"),
                              ("image_analysis","이미지 분석"),("results_report","결과 리포트"),
                              ("repeatability","반복촬영 분석"),("lighting","밝기 분석")]:
                st.page_link(pages[key],label=label,icon=PAGE_SPECS[key][3])
        st.markdown('<div class="nav-current"><small>현재 페이지</small><br><strong>'+
                    PAGE_SPECS[active][1]+'</strong></div>',unsafe_allow_html=True)
        st.divider()
        st.caption("COMMON NAVIGATION")
        for key in COMMON:
            st.page_link(pages[key],label=PAGE_SPECS[key][1],icon=PAGE_SPECS[key][3])
        st.divider()
        if page_data_source(active,st.session_state)=="SQLite":
            st.caption("● SQLITE · 이미지 등록 / 조회")
            st.caption("Calibration · 계측 · YOLO · MLflow · Roboflow\n미연결 / 분석 결과 없음")
        else:
            st.caption("● MOCK DATA ONLY")
            st.caption("Database · AI · MLflow · Roboflow\n미연결 / 실제 처리 없음")
