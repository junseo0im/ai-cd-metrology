"""SQLite Wafer detail for one Wafer × effective 노광 × PEB condition (TASK18-B-10A.2).

Read-only presentation of stored rows. Layout and styles reuse the Batch detail page. Nothing is
recomputed: counts come from SQLite and the latest stored analysis per image (no new AI judgment).
"""
import pandas as pd
import streamlit as st
from dashboard.components.sqlite_batch_detail import _value
from dashboard.components.sqlite_input import exposure_text,peb_text,condition_batches
from dashboard.components.inspection_wafer import DIE_CODES,POSITION_ORDER,exposure_label,peb_label
from dashboard.navigation import navigate

REVIEW_OUTCOMES=("QUALITY_REVIEW_REQUIRED","PARTIAL_MEASUREMENT","POLICY_REVIEW_REQUIRED")
OUTCOME_LABELS={"COMPLETE":"완료","QUALITY_REVIEW_REQUIRED":"품질 검토 필요","PARTIAL_MEASUREMENT":"부분 계측",
                "POLICY_REVIEW_REQUIRED":"PD 정책 검토","MODEL_PD_EXCLUDED":"PD 계측 제외","FAILED":"실패"}


def wafer_conditions(batches,wafer_id):
    """Distinct (노광, PEB) labels of one Wafer, in the Wafer table's order."""
    rows=batches[batches["wafer_id"]==wafer_id]
    pairs={(exposure_text(e,u),peb_text(p)) for e,u,p in zip(rows["exposure_value"],rows["exposure_unit"],rows["peb_temperature_c"])}
    return sorted(pairs)


def condition_label(exposure,peb):
    return f"노광 {exposure} · PEB {peb if peb=='—' else peb+' s'}"


def site_matrix(images,analysed_ids):
    """Die ID × lt/c/rb: registered images and latest-analysis presence per physical site."""
    rows=[]
    for die in DIE_CODES:
        row={"Die ID":die}
        for position in POSITION_ORDER:
            hits=images[(images["die_code"]==die)&(images["position_code"]==position)]
            total=len(hits)
            done=int(hits["image_id"].isin(analysed_ids).sum())
            row[position]=("미등록" if not total else
                           f"등록 {total} · "+("분석 있음" if done==total else "분석 없음" if not done else f"분석 {done}/{total}"))
        rows.append(row)
    return pd.DataFrame(rows,columns=["Die ID",*POSITION_ORDER])


def analysis_overview(image_ids,latest):
    """Counts from the latest stored analysis of each image in the condition (each image once)."""
    items=[latest[i] for i in image_ids if i in latest]
    outcomes={}
    for item in items:
        outcomes[item["outcome"]]=outcomes.get(item["outcome"],0)+1
    return {"registered":len(image_ids),"analysed":len(items),"not_analysed":len(image_ids)-len(items),
            "metrology":sum(item.get("metrology_status")=="SUCCESS" and bool((item.get("line_measurement_count") or 0)
                            or (item.get("gap_measurement_count") or 0)) for item in items),
            "partial":sum(item.get("partial_measurement")==1 for item in items),
            "defect":sum(bool(item.get("defect_classes")) for item in items),
            "pd_excluded":outcomes.get("MODEL_PD_EXCLUDED",0),
            "review":sum(outcomes.get(o,0) for o in REVIEW_OUTCOMES),
            "failed":outcomes.get("FAILED",0),"outcomes":outcomes}


def ia_context(wafer_id,selected):
    """Session values that pre-select this Wafer (and its single 노광/PEB) in I&A Wafer Map.

    I&A drops any value missing from its own options, so a mismatch falls back to its defaults.
    """
    context={"_wafer_selected_SQLite":wafer_id}
    exposures={exposure_label(e,u) for e,u in zip(selected["exposure_value"],selected["exposure_unit"])}
    pebs={peb_label(p) for p in selected["peb_temperature_c"]}
    if len(exposures)==1:
        context[f"ia_wafer_exposure_SQLite_{wafer_id}"]=exposures.pop()
    if len(pebs)==1:
        context[f"ia_wafer_peb_SQLite_{wafer_id}"]=pebs.pop()
    return context


def summary_cards(wafer_id,selected,images,overview,exposure,peb):
    exposure_shown=exposure  # TASK18-B-13A.2: the condition label already carries the display unit.
    with st.container(key="sqlite_wafer_summary"):
        for column,(label,value) in zip(st.columns([1,1,1,1.1,1.2,1]),[
                ("Wafer ID",wafer_id),("관련 Batch",f"{len(selected)}개"),("등록 이미지",f"{len(images)}장"),
                ("분석 결과 이미지",f"{overview['analysed']}장"),("노광 조건",exposure_shown),
                ("PEB (s)","미입력" if peb=="—" else peb)]):
            with column,st.container(border=True):
                _value(label,value)


def batch_list(selected,analysed,wafer_id):
    """Batches of this Wafer × condition. Counts are THIS Wafer's; a multi-Wafer Batch also shows its total."""
    widths=[2.4,.7,.7,.8,.9,1.3,.8]
    rows=selected.sort_values("created_at",ascending=False,kind="stable").to_dict("records")
    height="content" if len(rows)<=6 else 38+6*42
    with st.container(height=height,border=False,key="wafer_batch_table"):
        for col,heading in zip(st.columns(widths,gap="small"),
                               ["Batch 이름","노광","PEB (s)","등록 이미지","분석 결과 이미지","등록 시각","상태"]):
            col.markdown("**"+heading+"**")
        for item in rows:
            cols=st.columns(widths,gap="small",vertical_alignment="center")
            identity=item["batch_id"]
            multi=int(item.get("wafer_count",1) or 1)>1
            scope=(" · 다중 Wafer Batch ("+str(item.get("wafer_list"))+") · Batch 상세는 Batch 전체 표시" if multi
                   else " · Batch 상세 열기")
            if cols[0].button(item["display_name"],key="wafer_detail_batch_"+identity,type="tertiary",
                              help="Batch ID: "+identity+scope):
                navigate("batch_detail",selected_sqlite_batch_id=identity,batch_detail_source="SQLite")
            cols[1].write(exposure_text(item["exposure_value"],item.get("exposure_unit")))
            cols[2].write(peb_text(item["peb_temperature_c"]))
            count=int(item["image_count"])
            total=int(item.get("batch_image_count",count) or count)
            cols[3].write(f"{count} / Batch 전체 {total}" if multi else str(count))
            cols[4].write(str(int(analysed.get((identity,wafer_id),0))))
            cols[5].write(str(item["created_at"])[:16].replace("T"," "))
            cols[6].write("등록 완료" if item["batch_status"]=="registered" else str(item["batch_status"]))


def image_status(images,analysed_ids):
    with st.container(border=True,key="sqlite_wafer_sites"):
        st.subheader("이미지 현황")
        mapped=images[images["die_code"].notna()&images["position_code"].notna()]
        sites=len({(d,p) for d,p in zip(mapped["die_code"],mapped["position_code"])})
        st.caption(f"Die × 촬영 위치별 등록 이미지와 최신 분석 결과 존재 여부 · 이 조건의 Batch만 포함 · "
                   f"등록 위치 {sites}/{len(DIE_CODES)*len(POSITION_ORDER)}"
                   +(f" · 위치 미지정 {len(images)-len(mapped)}장" if len(images)>len(mapped) else ""))
        st.dataframe(site_matrix(images,analysed_ids),hide_index=True,width="stretch",height=36*9+2,
                     key="wafer_detail_site_matrix")


def analysis_status(overview):
    with st.container(border=True,key="sqlite_wafer_analysis"):
        st.subheader("분석 결과")
        st.caption("저장된 이미지별 최신 분석 결과 기준 · 같은 이미지의 재분석은 1장 · AI 잠정 결과이며 공식 판정 아님")
        for column,(label,value,note) in zip(st.columns(2)*4,[
                ("분석 결과 이미지",overview["analysed"],f"등록 {overview['registered']}장 중"),
                ("미분석 이미지",overview["not_analysed"],"저장된 분석 결과 없음"),
                ("Line/Gap 측정",overview["metrology"],f"부분 계측 {overview['partial']}장 포함"),
                ("결함 검출 기록",overview["defect"],"저장된 검출이 1건 이상인 이미지"),
                ("PD 계측 제외",overview["pd_excluded"],"outcome=MODEL_PD_EXCLUDED"),
                ("검토 필요",overview["review"],"품질 검토·부분 계측·PD 정책 검토"),
                ("실패",overview["failed"],"outcome=FAILED")]):
            column.metric(label,f"{value}장",help=note)
        if overview["outcomes"]:
            with st.expander("최신 결과 상태 분포"):
                st.dataframe(pd.DataFrame([{"상태":OUTCOME_LABELS.get(k,k),"outcome":k,"이미지 수":v}
                                           for k,v in sorted(overview["outcomes"].items())]),
                             hide_index=True,width="stretch")
