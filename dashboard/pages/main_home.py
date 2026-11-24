"""Main Home: Workspace entry, SQLite KPIs and Quick Access. Read-only."""
from html import escape
import sqlite3
import streamlit as st
from dashboard.navigation import navigate,PAGE_SPECS
from dashboard.sqlite_provider import runtime_provider

WORKSPACE_TITLES = {
    "data": "DATA WORKSPACE",
    "analysis": "ANALYSIS WORKSPACE",
}


def workspace_card(key,area,feature,description,button,route):
    with st.container(border=True,key="card_home_"+key):
        st.markdown('<div class="ws-title"><span class="ws-name">'+escape(WORKSPACE_TITLES[area])+'</span>'
                    '<span class="ws-feature"><span class="ws-sep">|</span>'+escape(feature)+'</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ws-desc">'+escape(description)+'</div>',unsafe_allow_html=True)
        if st.button(button,type="primary",width="stretch"):
            navigate(route)


# One content group, vertically centred by CSS (min-height from the viewport; scrolls naturally when short).
with st.container(key="home_center"):
    left,right = st.columns(2,gap="large")
    with left:
        workspace_card("data","data","Data Control Center","데이터 입력부터 Metadata, Calibration, Batch 관리까지.",
                       "Data Control Center 열기","data_control")
    with right:
        workspace_card("analysis","analysis","Inspection & Analysis","Wafer Map, 선폭·결함 분석 및 공정조건별 결과 확인.",
                       "Inspection & Analysis 열기","inspection_analysis")

    st.subheader("시스템 상태")
    try:
        status=runtime_provider().system_status()
    except (sqlite3.Error,OSError,RuntimeError,ValueError) as exc:
        # Never substitute 0 or example values for an unreadable operational DB.
        st.error("SQLite 상태를 조회할 수 없습니다. DB 연결을 확인하세요. "+str(exc))
    else:
        stats=st.columns(5)
        stats[0].metric("등록 Batch",status["batches"])
        stats[1].metric("등록 이미지",status["images"])
        stats[2].metric("분석 완료 이미지",status["analysed_images"],
                        help="이미지별 최신 분석 실행 기준 · 같은 이미지의 재분석은 1장으로 계산 · 공식 품질 PASS 아님")
        stats[3].metric("검토 필요 이미지",status["review_images"],
                        help="최신 결과가 품질 검토·부분 계측·PD 정책 검토인 이미지 (PD 계측 제외 "
                             f"{status['pd_excluded_images']}장 · 실패 {status['failed_images']}장 별도)")
        run=status["latest_run"]
        stats[4].metric("최근 분석 실행",run["execution_status"] if run else "실행 이력 없음",
                        help=(f"{run['started_at'][:19].replace('T',' ')} UTC · {run['requested_images']}장 · 실패 {run['failed_images']}장"
                              if run else None))
    with st.container(border=True):
        st.caption("QUICK ACCESS")
        for col,key in zip(st.columns(3),["dataset","model_performance","settings"]):
            with col:
                st.page_link(PAGE_SPECS[key][0],label=PAGE_SPECS[key][1],icon=PAGE_SPECS[key][3])
