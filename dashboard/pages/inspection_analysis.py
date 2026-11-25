import pandas as pd
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import inline_title,analysis_modes,select_batch,select_image
from dashboard.components.inspection_wafer import inspection_wafer,sync_mock_image,render_ia_selected_image
from dashboard.components.process_analysis import render_process_analysis
from dashboard.components.ia_analysis_result import render_analysis_summary,render_analysis_details,render_die_summary,render_image_summary
from dashboard.components.inspection_wafer import die_hover_lines,CRITICALITY_STYLES
CRITICALITY_COLORS={k.value:c for k,(_,c) in CRITICALITY_STYLES.items()}
from dashboard.components.ia_modes import render_mode_tab
from dashboard.components.images import image_panel
from dashboard.components.tables import data_table
from dashboard.navigation import navigate

inline_title("Inspection & Analysis","")  # TASK18-B-13A: subtitle removed.
sqlite_images=st.session_state.get("_wafer_source","SQLite")=="SQLite"


def render_sqlite_page():
    """Operational I&A: SQLite only, four tabs. Stored AI results are read, never recomputed."""
    st.session_state["ia_sqlite_selected_image_id"]=None
    wafer_tab,width_tab,repeat_tab,light_tab=st.tabs(["Wafer / Image","선폭 / 오차율 / 결함","Repeatability","Lighting"])
    with wafer_tab:
        left,right=st.columns(2,gap="medium")  # TASK18-B-14B: Map 50% · images 50%
        with left,st.container(border=True,key="card_ia_map"):
            map_context,map_die=inspection_wafer(None)
        with right,st.container(border=True,key="card_ia_results"):
            render_ia_selected_image(map_context,map_die)
            ia_view=st.session_state.get("ia_analysis_view")
            st.divider()
            if map_die is not None:
                # TASK18-B-15: the selected Die (current filter, lt/c/rb), same status/colour/reason as the Map.
                # TASK18-B-16: image table (selected position) | Die table (lt/c/rb); narrow -> Die above image.
                location=st.session_state.get("ia_sqlite_selected_location") or {}
                position=location.get("position_code","")
                selected_image=st.session_state.get("ia_sqlite_selected_image_id")
                with st.container(key="ia_summary_pair"):
                    image_col,die_col=st.columns(2,gap="medium")
                    with die_col:
                        st.subheader(f"선택 Die 분석 요약 (lt/c/rb)",anchor=False)
                        render_die_summary(map_die," · ".join(die_hover_lines(map_die)[1:]),
                                           CRITICALITY_COLORS.get(map_die["criticality"],CRITICALITY_COLORS["NOT_EVALUATED"]))
                    with image_col:
                        st.subheader(f"선택 이미지 분석 요약 ({position or '위치 없음'})",anchor=False)
                        if selected_image:
                            render_image_summary(map_die,position,selected_image)
                        else:
                            st.info("선택 위치에 등록 이미지 없음")
                st.caption("유효 계측값만 평균 (괄호: 유효 측정 수) · 오차율은 설계 10 µm 기준 · "
                           "결함 개수는 모델 검출만 (PD 의심 보류 제외) · AI 잠정 결과")
                if ia_view and ia_view["state"]=="ERROR":
                    st.error(ia_view["message"])
                elif ia_view and ia_view["state"]=="OK":
                    render_analysis_details(ia_view,location.get("position_code",""))  # Selected image's detail.
                elif st.session_state.get("ia_sqlite_selected_image_id"):
                    st.info("이 이미지의 저장된 분석 결과가 없습니다 · Data Control Center에서 분석을 실행할 수 있습니다.")
    provider=map_context["provider"] if map_context else None
    with width_tab,st.container(border=True,key="card_ia_trend"):
        # TASK18-B-13A: title, scope summary, '분석 조건' and Simulation share one compact row.
        render_process_analysis("SQLite",provider,map_context.get("scope_label") if map_context else None,
                                title="공정 조건별 분석",on_simulation=lambda:navigate("simulation"))
    with repeat_tab,st.container(border=True,key="card_ia_repeatability"):
        st.subheader("Repeatability · 반복 촬영 그룹")
        render_mode_tab(provider,"repeatability")
    with light_tab,st.container(border=True,key="card_ia_lighting"):
        st.subheader("Lighting · 조명 조건")
        render_mode_tab(provider,"lighting")


if sqlite_images:
    render_sqlite_page()
    st.stop()

# Explicit Mock selection only (internal/tests): preserved example layout.
controls,selection=st.columns([2,1],vertical_alignment="bottom")
with controls:
    analysis_modes("Wafer Batch")
with selection:
    batch_id=select_batch("inspection_batch","Wafer Batch")
left,right=st.columns(2,gap="medium")
with left:
    with st.container(border=True,height=355,key="card_ia_map"):
        map_context,map_die=inspection_wafer(batch_id)
        sqlite_images=st.session_state.get("_wafer_source","SQLite")=="SQLite"
        if not sqlite_images:
            sync_mock_image(batch_id)
            with st.popover("이미지 선택 · Mock",width="content"):
                image_id=select_image(batch_id,"ia_image")
                st.caption("I&A 하단 전용 Mock 이미지 · SQLite Wafer 탐색과 독립적입니다.")
    with st.container(border=True,height=350,key="card_ia_results"):
        image_col,summary_col=st.columns([1.1,1],gap="medium")
        with image_col:
            if sqlite_images:
                render_ia_selected_image(map_context,map_die)
            else:
                st.session_state["ia_sqlite_selected_image_id"]=None
                st.session_state["ia_sqlite_selected_location"]=None
                image_title,overlay_control=st.columns([1.2,1],vertical_alignment="center")
                image_title.subheader("선택 이미지")
                with overlay_control:
                    overlay=st.toggle("Overlay 예시",value=st.session_state.pop("_restore_ia_overlay",st.session_state.get("_ia_mock_overlay",True)),key="ia_overlay")
                st.session_state["_ia_mock_overlay"]=overlay
                image_panel(image_id,overlay,caption=False)
                st.caption(image_id+" · "+("Overlay 예시" if overlay else "원본 영역 예시")+" / Mock")
                _,detail_button,_=st.columns([1,1,1])
                if detail_button.button("이미지 상세 열기",key="ia_detail",width="content"):
                    navigate("image_analysis",selected_image_id=image_id,selected_batch_id=batch_id)
        with summary_col:
            ia_view=st.session_state.get("ia_analysis_view") if sqlite_images else None
            if ia_view and ia_view["state"]!="NONE":
                st.subheader("선택 이미지 분석 결과")
                render_analysis_summary(ia_view)  # Stored result only; no recompute/adoption.
            else:
                st.subheader("Batch 결과 표"+(" · 예시" if not sqlite_images else " · 미연결"))
                if sqlite_images:
                    data_table(pd.DataFrame({"항목":["Line Width","Gap","Edge"],
                        "평균 (µm)":["계측 결과 없음"]*3,"표준편차 (µm)":["데이터 없음"]*3}),height=135)
                    selected_id=st.session_state.get("ia_sqlite_selected_image_id")
                    selected=next((row for row in (map_context or {}).get("images",[]) if row["image_id"]==selected_id),None)
                    st.caption(("Batch: "+selected["batch_id"] if selected else "이미지 선택 후 실제 Batch 연결")+" · 계측/결함 결과 미연결")
                    st.info("결함 검출 건수: 미분석")
                else:
                    averages,_,_=data.batch_result_tables(batch_id)
                    data_table(pd.DataFrame({"항목":["Line Width","Gap","Edge"],
                        "평균 (µm)":[str(v)+" (예시)" for v in averages["평균 (µm 예시)"]],
                        "표준편차 (µm)":["데이터 없음"]*3}),height=135)
                    st.caption(batch_id+" · 기존 Mock 정적 예시 · Calibration 적용 실제 결과 아님")
                    with st.expander("결함·상태 예시"):
                        original=data.batch_summary_table(batch_id)
                        data_table(original[~original["항목"].str.startswith("평균 ")],height=160)
                st.caption("표준편차: 유효 이미지별 평균의 표본 표준편차. 0장 데이터 없음 / 1장 산출 불가 / 2장 이상 계산.")
with right,st.container(border=True,height=720,key="card_ia_trend"):
    title,simulation=st.columns([4,1],vertical_alignment="center")
    title.subheader("공정 조건별 분석")
    if simulation.button("Simulation",width="stretch"):
        navigate("simulation",_restore_ia_overlay=st.session_state.get("ia_overlay",True))
    source="SQLite" if sqlite_images else "Mock"
    render_process_analysis(source,map_context["provider"] if map_context and source=="SQLite" else None,
                            map_context.get("scope_label") if map_context and source=="SQLite" else None)
