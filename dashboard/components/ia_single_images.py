"""I&A 'Image' tab: stored analysis results of images registered as Single Image (read-only).

Lists every Single Image (newest first) and shows the selected image's stored result with the
same parts as the Wafer tab: 선폭/결함 Overlay, original, '선택 이미지 분석 요약' and details.
Nothing is written, re-analysed or adopted. Single Images are not shown on the Wafer Map and are
not part of the Die judgment (inspection_wafer.wafer_context, die_assessment.JUDGED_MODES); this
tab is where they are viewed.
"""
import sqlite3

import pandas as pd
import streamlit as st

from dashboard.components import ia_analysis_result as ia
from dashboard.components.sqlite_input import render_original_preview
from dashboard.components.tables import data_table, open_image_detail

UPLOAD_MODE = "single_image"
PAGE_SIZE = 100
LIST_KEY = "ia_single_list"
POSITIONS = ("lt", "c", "rb")


def single_images(provider):
    """Every Single Image, newest first (provider order: created_at DESC, image_id)."""
    chunks, offset = [], 0
    while True:
        frame = provider.images(upload_mode=UPLOAD_MODE, limit=PAGE_SIZE, offset=offset)
        chunks.append(frame)
        if len(frame) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.concat(chunks, ignore_index=True)


def image_status_text(image_id, item):
    """Same wording/colour as the Wafer tab's selected-image status (ia.image_status), unchanged."""
    die = {"positions": {p: {"image_ids": [], "analysis": {}} for p in POSITIONS}}
    die["positions"]["c"]["analysis"] = {image_id: item} if item else {}
    return ia.image_status(die, image_id)


def select_run(provider, record):
    """Latest stored run by default; a history selector only when several runs exist.

    Same behaviour as ia.select_run, with this tab's own widget key so the same image can be
    selected in the Wafer tab and here at the same time.
    """
    view = ia.load_image_analysis(provider, record)
    history = view["history"]
    if len(history) > 1:
        labels = {row["analysis_run_id"]: row for row in history}
        chosen = st.selectbox("분석 실행 이력", list(labels), format_func=lambda key: ia._run_label(labels[key]),
                              key="ia_single_run_" + record["image_id"], label_visibility="collapsed",
                              help="기본값: 가장 최근 실행. 공식 채택 여부는 별도로 표시합니다.")
        if chosen != view["row"]["analysis_run_id"]:
            view = ia.load_image_analysis(provider, record, chosen)
    return view


def _list_frame(images, latest):
    return pd.DataFrame({
        "파일명": images["original_filename"],
        "Wafer / Die / 위치": images["wafer_id"] + " / " + images["die_code"] + " / " + images["position_code"],
        "등록일": images["created_at"].astype(str).str[:16].str.replace("T", " ", regex=False),
        "분석 상태": [image_status_text(i, latest.get(i))[0] for i in images["image_id"]],
    })


def _selected_index(count):
    state = st.session_state.get(LIST_KEY) or {}
    selection = state.get("selection") if hasattr(state, "get") else getattr(state, "selection", None)
    rows = list((selection.get("rows") if hasattr(selection, "get") else getattr(selection, "rows", None)) or [])
    return rows[0] if rows and 0 <= rows[0] < count else 0


def render_single_image_tab(provider):
    """Image tab body. provider: SQLite runtime provider (None when it could not be opened)."""
    if provider is None:
        st.info("SQLite 데이터를 조회할 수 없습니다 · DB 연결을 확인하세요.")
        return
    try:
        images = single_images(provider)
        latest = provider.latest_analysis() if not images.empty else {}
    except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
        st.error("Single Image 조회 실패 · Mock으로 대체하지 않습니다. " + str(exc))
        return
    if images.empty:
        st.info("등록된 Single Image가 없습니다 · Data Control Center에서 Single Image로 등록할 수 있습니다.")
        return
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="card_ia_single_list"):
        st.subheader("Single 이미지 목록", anchor=False)
        st.caption("Data Control Center에서 Single Image로 등록한 이미지 · 등록일 최신순 · 분석 상태는 저장된 최신 실행 기준")
        data_table(_list_frame(images, latest), key=LIST_KEY, on_select="rerun", selection_mode="single-row",
                   height=min(560, (len(images) + 1) * 31 + 3))
        st.caption(f"{len(images)}장 · 행을 선택하면 오른쪽에 분석 결과가 표시됩니다")
    selected = images.iloc[_selected_index(len(images))]
    with right, st.container(border=True, key="card_ia_single_result"):
        _render_selected(provider, selected, latest)


def _render_selected(provider, selected, latest):
    image_id = selected["image_id"]
    position = selected["position_code"]
    head, identity = st.columns([1, 2], vertical_alignment="bottom")
    head.subheader("선택 이미지", anchor=False)
    identity.caption(f"{selected['wafer_id']} / Die {selected['die_code']} / {position} · Single Image")
    record = provider.image_record(image_id, batch_id=selected["batch_id"])
    if record is None:
        st.error("선택한 Single Image를 찾을 수 없습니다. 목록에서 다시 선택하세요.")
        return
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="ia_single_controls"):
        show_metrology = st.toggle("선폭 Overlay", value=True, key="ia_single_metrology_overlay",
                                   help="저장된 분석 결과의 Metrology Overlay (재추론 없음)")
        show_defect = st.toggle("결함 Overlay", value=True, key="ia_single_defect_overlay",
                                help="저장된 분석 결과의 Defect Overlay (재추론 없음)")
        view = select_run(provider, record)  # Read-only stored result.
    pair_left, pair_right = st.columns(2, gap="small")
    pair_left.markdown('<span class="ia-image-label">선폭 Overlay</span>', unsafe_allow_html=True)
    pair_right.markdown('<span class="ia-image-label">결함 Overlay</span>', unsafe_allow_html=True)
    with pair_left:
        ia.render_stage_overlay(view, "metrology", show_metrology)
    with pair_right:
        ia.render_stage_overlay(view, "defect", show_defect)
    original, original_info = st.columns(2, gap="small")
    original.markdown('<span class="ia-image-label">원본</span>', unsafe_allow_html=True)
    with original:
        render_original_preview(provider, record)
    with original_info:
        st.caption(position + " · " + record["original_filename"] + " · " + record["image_id"],
                   help="Image ID: " + record["image_id"])
        if st.button("이미지 상세 열기", key="ia_single_detail", type="tertiary"):
            open_image_detail(record["image_id"], record["batch_id"])
    st.divider()
    st.subheader("선택 이미지 분석 요약", anchor=False)
    item = latest.get(image_id)
    status, color = image_status_text(image_id, item)
    summary = ia.summarize({position: [image_id]}, {image_id: item} if item else {})
    ia.render_summary_table(summary, status, color, key="ia_single_summary_table")
    st.caption("유효 계측값만 평균 (괄호: 유효 측정 수) · 오차율은 설계 10 µm 기준 · "
               "결함 개수는 모델 검출만 (PD 의심 보류 제외) · AI 잠정 결과")
    if view["state"] == "ERROR":
        st.error(view["message"])
    elif view["state"] == "OK":
        ia.render_analysis_details(view, position)
    else:
        st.info("이 이미지의 저장된 분석 결과가 없습니다 · Data Control Center에서 분석을 실행할 수 있습니다.")
