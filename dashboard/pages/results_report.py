"""I&A 결과 리포트 (1차): 대상 선택 → 개요·포함 항목 → 미리보기 → Word(.docx) 다운로드.

Operational SQLite is read only. The preview and the Word file are generated from the same report
data (dashboard.report_builder) and bound to a signature of the scope + inputs; any change to the
target, condition, inputs or sections hides the old preview and its download.
"""
import streamlit as st
from dashboard import report_builder as rb
from dashboard.components.layout import page_title
from dashboard.components.tables import data_table
from dashboard.sqlite_provider import runtime_provider

page_title("결과 리포트", "저장된 검사 결과로 Word 리포트를 만듭니다.", "B / REPORT")
STATE = "_results_report"


def _load_targets(provider):
    batches = provider.batches()
    wafers = provider.wafers()
    return ({r["batch_id"]: r["display_name"] + " · " + r["batch_id"][-8:] for r in batches.to_dict("records") if r["image_count"]},
            [w for w, n in zip(wafers["wafer_id"], wafers["image_count"]) if n])


try:
    provider = runtime_provider()
    batch_names, wafer_ids = _load_targets(provider)
except Exception as exc:  # DB not reachable: show the state only.
    st.info("운영 SQLite를 읽을 수 없습니다 · " + type(exc).__name__)
    st.stop()

# ---- 1. 대상 선택 --------------------------------------------------------------------------------
with st.container(border=True, key="report_target"):
    st.subheader("1. 대상 선택")
    kind_col, target_col, condition_col = st.columns([1, 2.2, 2], vertical_alignment="bottom")
    kind = kind_col.segmented_control("대상 종류", ["Wafer", "Batch"], default="Wafer", key="report_kind") or "Wafer"
    options = wafer_ids if kind == "Wafer" else list(batch_names)
    if not options:
        st.info("등록 이미지가 있는 " + kind + "가 없습니다.")
        st.stop()
    target = target_col.selectbox(kind, options, key="report_target_" + kind,
                                  format_func=(lambda v: v) if kind == "Wafer" else batch_names.get)
    with st.spinner("선택 범위 조회 중…"):
        conditions = sorted(set(rb.target_images(provider, kind, target)["condition"]))
    choices = ([rb.ALL_CONDITIONS] if len(conditions) > 1 else []) + conditions
    condition = condition_col.selectbox(
        "노광 · PEB 조건", choices, key=f"report_condition_{kind}_{target}",
        format_func=lambda c: f"전체 조건 ({len(conditions)}개 · Wafer·조건별 구분)" if c == rb.ALL_CONDITIONS else c)
    scope = rb.scope(provider, kind, target, condition)
    images = scope["images"]
    analysed = sum(i in scope["analysis"] for i in images["image_id"])
    wafers = "" if kind == "Wafer" else " · " + ", ".join(sorted(set(images["wafer_id"])))
    st.caption(f"선택 범위: {kind} {target}{wafers} · "
               f"등록 {len(images)}장 · 분석 결과 {analysed}장 · 그룹 {len(scope['groups'])}개 (Wafer × 조건)")

if images.empty:
    st.info("선택 범위에 등록 이미지가 없습니다.")
    st.stop()
if not analysed:
    st.info("선택 범위에 저장된 분석 결과가 없습니다 · Data Control Center에서 분석 후 리포트를 만들 수 있습니다.")
    st.stop()

# ---- 2. 개요 입력 · 포함 항목 ----------------------------------------------------------------------
scope_key = f"{kind}_{target}_{condition}"
with st.container(border=True, key="report_form"):
    st.subheader("2. 개요 입력 · 포함 항목")
    suggested = f"{kind} {target} 검사 결과 리포트" + ("" if condition == rb.ALL_CONDITIONS else f" ({condition})")
    title = st.text_input("제목", value=suggested, key="report_title_" + scope_key)
    author_col, purpose_col = st.columns(2)
    author = author_col.text_input("작성자", key="report_author", placeholder="입력한 경우에만 문서에 표시")
    purpose = purpose_col.text_input("실험 목적", key="report_purpose", placeholder="입력한 경우에만 문서에 표시")
    notes = st.text_area("특이사항", key="report_notes", height=80, placeholder="입력한 경우에만 문서에 표시")
    st.markdown("**포함 항목** · 실험 개요 · 공정·분석 조건 · 분석 범위 · 계측 결과는 항상 포함")
    c1, c2, c3 = st.columns(3)
    sections = {"defects": c1.checkbox("결함 검출 결과", value=True, key="report_sec_defects"),
                "wafer_map": c2.checkbox("Wafer Map", value=True, key="report_sec_wafer_map"),
                "images": c3.checkbox("대표 원본 · Overlay", value=False, key="report_sec_images")}
    image_id = None
    if sections["images"]:
        candidates = images[images["image_id"].isin(scope["analysis"])].sort_values(["wafer_id", "die_code", "position_code"])
        labels = {r["image_id"]: f"{r['wafer_id']} · Die {r['die_code']} · {r['position_code']} · {r['original_filename']}"
                  for r in candidates.to_dict("records")}
        image_id = st.selectbox("대표 이미지 (선택 범위 안)", list(labels), format_func=labels.get, key="report_image_" + scope_key)

report_options = {"title": title.strip() or suggested, "author": author.strip(), "purpose": purpose.strip(),
                  "notes": notes.strip(), **sections, "image_id": image_id}
signature = rb.signature(scope, report_options)

# ---- 3. 미리보기 · 4. Word -------------------------------------------------------------------------
with st.container(border=True, key="report_preview"):
    head, action = st.columns([3, 1], vertical_alignment="center")
    head.subheader("3. 미리보기 · Word 다운로드")
    if action.button("미리보기 생성", type="primary", width="stretch", key="report_build"):
        with st.spinner("저장된 결과로 리포트 생성 중…"):
            report = rb.build(provider, scope, report_options)
            try:
                document, error = rb.docx_bytes(report), None
            except Exception as exc:  # Word blocked: report the cause, offer text + PNG fallback.
                document, error = None, type(exc).__name__ + ": " + str(exc)
        st.session_state[STATE] = {"signature": signature, "report": report, "docx": document, "error": error}
    saved = st.session_state.get(STATE)
    if not saved:
        st.caption("미리보기를 생성하면 같은 내용으로 Word 파일을 내려받을 수 있습니다.")
    elif saved["signature"] != signature:
        st.caption("대상·조건·입력·포함 항목이 바뀌었습니다 · 미리보기를 다시 생성하세요 (이전 문서는 다운로드되지 않습니다).")
    else:
        report = saved["report"]
        if saved["docx"]:
            st.download_button("Word(.docx) 다운로드", saved["docx"], file_name=rb.filename(report),
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               type="primary", key="report_download")
        else:
            st.error("Word 생성 실패 · " + saved["error"])
            st.download_button("본문 텍스트 (.txt)", rb.plain_text(report).encode("utf-8"), file_name="report.txt", key="report_txt")
            for index, chart in enumerate(report["charts"]):
                st.download_button(chart["title"] + " (PNG)", chart["png"], file_name=f"chart_{index + 1}.png",
                                   mime="image/png", key=f"report_png_{index}")
        st.markdown(f"#### {report['title']}")
        frames = rb.preview_frames(report)
        st.markdown("**1. 실험 개요**")
        data_table(frames["overview"])
        st.markdown("**2. 공정·분석 조건**")
        data_table(frames["process"])
        st.markdown("**3. 분석 범위**")
        data_table(frames["scope"])
        if report["scope"]["excluded"]:
            st.caption("계측 제외 사유: " + " · ".join(f"{k} {v}장" for k, v in report["scope"]["excluded"].most_common()))
        st.markdown("**4. 계측 결과**")
        st.caption("웨이퍼 전체 Line Width 통계")
        data_table(frames["wafer_stats"])
        st.caption(rb.WAFER_STATS_NOTE)
        for label, frame in frames["measure"].items():
            st.caption(label)
            data_table(frame)
        for chart in [c for c in report["charts"] if not c.get("wafer_map")]:
            st.image(chart["png"], caption=chart["title"])
        number = 5
        if "defects" in frames:
            st.markdown(f"**{number}. 결함 검출 결과**"); number += 1
            data_table(frames["defects"])
        if "wafer_map" in report["sections"]:
            st.markdown(f"**{number}. Wafer Map** · AI 잠정 위험도 v0.1"); number += 1
            maps = [c for c in report["charts"] if c.get("wafer_map")]
            for column, chart in zip(st.columns(max(1, min(3, len(maps)))) * len(maps), maps):
                column.image(chart["png"], caption=chart["title"])
        if report.get("image"):
            image = report["image"]
            st.markdown(f"**{number}. 대표 원본 · Overlay** · {image['filename']} · {image['location']}")
            shown = [(label, image[key]) for label, key in (("원본", "original"), ("Metrology Overlay", "metrology"),
                                                             ("Defect Overlay", "defect")) if image.get(key)]
            for column, (label, path) in zip(st.columns(max(1, len(shown))), shown):
                column.image(path, caption=label)
