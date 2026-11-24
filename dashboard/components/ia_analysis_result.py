"""I&A lower-left card: stored analysis result for the selected registered image (read-only).

Reads through AnalysisRepository(read_only=True): mode=ro connections only, official result
root, existing SHA checks. Nothing is recomputed, re-judged, re-inferred or adopted.
Default run = most recent stored run; adoption is shown separately and never inferred.
"""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from database.analysis_repository import AnalysisRepository,AnalysisStorageError,OPERATIONAL_RESULT_ROOT
from dashboard.components.tables import data_table

NO_RESULT_CAPTION="저장된 분석 결과 없음 · 원본만 표시"


def result_root():
    return Path(os.environ.get("IDA_ANALYSIS_OUTPUT_ROOT",OPERATIONAL_RESULT_ROOT))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load_image_analysis(provider,record,analysis_run_id=None):
    """Return a display view; state is NONE, OK or ERROR. Never writes."""
    view={"state":"NONE","history":[],"row":None,"batch_run":None,"detail":None,
          "overlays":{},"detail_error":None,"message":None}
    try:
        if provider.schema_info()["version"] not in (4,5,6):
            return view  # No analysis tables before v4.
        root=result_root()
        if not root.is_dir():
            with closing(provider.connect()) as c:
                count=c.execute("SELECT COUNT(*) FROM analysis_image_runs WHERE image_id=?",(record["image_id"],)).fetchone()[0]
            if count:
                view.update(state="ERROR",message="분석 결과 폴더를 찾을 수 없습니다.")
            return view
        repository=AnalysisRepository(provider.db_path,result_root=root,image_root=provider.storage_root,read_only=True)
        history=repository.image_history(record["image_id"])
        view["history"]=history
        if not history:
            return view
        ids=[row["analysis_run_id"] for row in history]
        row=history[ids.index(analysis_run_id)] if analysis_run_id in ids else history[0]
        view["row"]=row
        if row["image_id"]!=record["image_id"] or str(row["image_sha256"]).lower()!=str(record["sha256"]).lower():
            view.update(state="ERROR",message="분석 결과와 등록 이미지(ID/SHA-256)가 일치하지 않아 표시를 중단했습니다.")
            return view
        view["batch_run"]=repository.get_batch_run(row["batch_run_id"])
        view["state"]="OK"
        try:  # Repository verifies result/overlay SHA before returning measurements.
            detail=repository.get_analysis_run(row["analysis_run_id"])
            raw=detail.get("raw_result") or {}
            if raw.get("image_id")!=record["image_id"]:
                raise AnalysisStorageError("result.json image_id mismatch")
            view["detail"]=detail
        except (AnalysisStorageError,OSError,ValueError) as exc:
            view["detail_error"]=type(exc).__name__+": "+str(exc)
        applied=((view["detail"] or {}).get("raw_result") or {}).get("metrology",{}).get("calibration")
        if view["detail"] and not applied:
            view["registered_calibration"]=_registered_calibration(provider,record)
        for stage in ("defect","metrology"):
            rel=row[stage+"_overlay_path"]
            if rel is None:
                view["overlays"][stage]={"error":"저장된 Overlay 없음"}
                continue
            try:
                path=(root/rel).resolve(strict=True)
                if not path.is_relative_to(root.resolve()) or _sha(path)!=row[stage+"_overlay_sha256"]:
                    raise AnalysisStorageError("Overlay SHA-256/경로 불일치")
                view["overlays"][stage]={"path":str(path)}
            except (AnalysisStorageError,OSError) as exc:
                view["overlays"][stage]={"error":"Overlay 파일을 표시할 수 없습니다: "+type(exc).__name__}
    except (AnalysisStorageError,PermissionError,OSError,ValueError,RuntimeError,sqlite3.Error) as exc:
        view.update(state="ERROR",message="분석 결과 조회 실패: "+type(exc).__name__+": "+str(exc))
    return view


def _registered_calibration(provider,record):
    """Approved calibration linked by the existing read-only mapper, or None (never assumed)."""
    try:
        plan=provider.prepare_analysis_inputs(record["batch_id"],image_ids=[record["image_id"]])
        request=plan["items"][0].get("candidate_request") or {}
        return request.get("calibration")
    except (KeyError,IndexError,ValueError,OSError,RuntimeError,sqlite3.Error):
        return None


def _pd_excluded(row):
    """Model PD exclusion or heuristic-PD policy review: no line/gap measurement was made."""
    return row is not None and row["outcome"] in ("MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED")


EXCLUDED_TEXT={"MODEL_PD_EXCLUDED":("PD 검출 · 선폭 계측 제외","PD 검출로 선폭 계측 제외"),
               "POLICY_REVIEW_REQUIRED":("PD 의심(규칙 기반) · 정책 검토 필요 · 선폭 계측 보류",
                                         "PD 의심으로 선폭 계측 보류 (정책 검토 필요)")}


def _run_label(row):
    adopted="공식 채택" if row["is_adopted"] else "미채택"
    return f"{row['started_at'] or '시각 미기록'} · {row['outcome']} · {adopted}"


def select_run(provider,record):
    """Load the default (latest) run; show a selector only when several runs exist."""
    view=load_image_analysis(provider,record)
    history=view["history"]
    if len(history)>1:
        labels={row["analysis_run_id"]:row for row in history}
        chosen=st.selectbox("분석 실행 이력",list(labels),format_func=lambda key:_run_label(labels[key]),
            key="ia_analysis_run_"+record["image_id"],label_visibility="collapsed",
            help="기본값: 가장 최근 실행. 공식 채택 여부는 별도로 표시합니다.")
        if chosen!=view["row"]["analysis_run_id"]:
            view=load_image_analysis(provider,record,chosen)
    return view


def render_overlays(view,show_metrology,show_defect):
    if view["state"]!="OK":
        st.caption(NO_RESULT_CAPTION if view["state"]=="NONE" else "분석 결과 표시 불가 · 원본만 표시")
        return
    for stage,label,shown in (("metrology","Metrology Overlay",show_metrology),("defect","Defect Overlay",show_defect)):
        if not shown:
            continue
        item=view["overlays"].get(stage) or {}
        if item.get("path"):
            st.image(item["path"],width="stretch")
            st.caption(label+" · 저장된 분석 결과 (재추론 없음)")
        elif stage=="metrology" and _pd_excluded(view["row"]) and view["row"]["metrology_overlay_path"] is None:
            st.caption("Metrology Overlay 없음 · "+EXCLUDED_TEXT[view["row"]["outcome"]][1])
        else:
            st.warning(label+": "+item.get("error","표시할 수 없습니다."))
    st.caption("저장된 분석 결과 · Overlay는 토글로 표시")


BOTH_NOTE=("두 Overlay는 각각 원본이 합성된 저장 이미지이고 Defect Overlay는 일부 이미지에서 기울기 보정(deskew) "
           "좌표계로 저장되어, 한 장으로 정확히 겹칠 수 없어 나란히 표시합니다 (재합성·재추론 없음).")


def render_overlay_slot(view,show_metrology,show_defect):
    """I&A right image (TASK18-B-14B): the stored overlay(s) chosen by the toggles, never redrawn or re-run."""
    if view is None or view["state"]!="OK":
        st.info("Overlay 없음")
        if view is None:
            st.caption("저장된 분석 결과 없음 · 원본만 표시")
        else:
            st.caption(NO_RESULT_CAPTION if view["state"]=="NONE" else "분석 결과 표시 불가 · 원본만 표시")
        return
    if not (show_metrology or show_defect):
        st.info("Metrology 또는 Defect Overlay를 켜면 표시됩니다.")
        return
    if show_metrology and show_defect:
        st.caption(BOTH_NOTE)
    for stage,label,on in (("metrology","Metrology Overlay",show_metrology),("defect","Defect Overlay",show_defect)):
        if not on:
            continue
        item=view["overlays"].get(stage) or {}
        if item.get("path"):
            if show_metrology and show_defect:
                st.caption(label)
            st.image(item["path"],width="stretch")
        elif stage=="metrology" and _pd_excluded(view["row"]) and view["row"]["metrology_overlay_path"] is None:
            st.caption("Metrology Overlay 없음 · "+EXCLUDED_TEXT[view["row"]["outcome"]][1])
        else:
            st.warning(label+": "+item.get("error","표시할 수 없습니다."))


def render_in_place(view,show_metrology,show_defect,draw_original):
    """Overlay toggles replace the original at the same position with the stored composite.

    Stored overlays already contain the original, so they are never drawn on top of it again.
    Both toggles: the two stored composites side by side (an exact single merge is not possible).
    """
    if view is None or view["state"]!="OK" or not (show_metrology or show_defect):
        draw_original()
        if view is None:
            st.caption("저장된 분석 결과 없음 · 원본만 표시")
        elif view["state"]!="OK":
            st.caption(NO_RESULT_CAPTION if view["state"]=="NONE" else "분석 결과 표시 불가 · 원본만 표시")
        else:
            st.caption("원본 · Overlay 토글을 켜면 같은 위치에 저장된 Overlay를 표시합니다")
        return
    shown=[];notes=[]
    for stage,label,on in (("metrology","Metrology Overlay",show_metrology),("defect","Defect Overlay",show_defect)):
        if not on:
            continue
        item=view["overlays"].get(stage) or {}
        if item.get("path"):
            shown.append((label,item["path"]))
        elif stage=="metrology" and _pd_excluded(view["row"]) and view["row"]["metrology_overlay_path"] is None:
            notes.append((st.caption,"Metrology Overlay 없음 · "+EXCLUDED_TEXT[view["row"]["outcome"]][1]))
        else:
            notes.append((st.warning,label+": "+item.get("error","표시할 수 없습니다.")))
    if not shown:
        draw_original()
        st.caption("표시할 Overlay가 없어 원본을 표시합니다")
    elif len(shown)==1:
        label,path=shown[0]
        st.image(path,width="stretch")
        st.caption(label+" · 저장된 분석 결과 (재추론 없음) · 원본 위치에 표시")
    else:
        for column,(label,path) in zip(st.columns(2,gap="small"),shown):
            with column:
                st.image(path,width="stretch")
                st.caption(label+" · 저장된 분석 결과 (재추론 없음)")
        st.caption(BOTH_NOTE)
    for show,text in notes:
        show(text)


def _num(value,digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


STATUS_LABELS={"COMPLETE":("분석 완료","ready"),"QUALITY_REVIEW_REQUIRED":("검토 필요 · 품질","review"),
               "POLICY_REVIEW_REQUIRED":("검토 필요 · PD 정책","review"),"FAILED":("실패","blocked")}


def render_analysis_summary(view):
    """Summary column. Caller keeps the existing placeholder when state is NONE."""
    if view["state"]=="ERROR":
        st.error(view["message"])
        return
    row=view["row"];batch=view["batch_run"] or {}
    review=row["outcome"]
    excluded=_pd_excluded(row) and row["line_measurement_count"] is None and row["gap_measurement_count"] is None
    # TASK18-B-14B.1: compact cards. Display wording only; stored values and the outcome are unchanged.
    label,tone=STATUS_LABELS.get(review,(str(review),"review"))
    if review=="PARTIAL_MEASUREMENT":
        label="부분 계측 · 검토 필요 · 평균은 참고값"
    elif _pd_excluded(row):
        label=EXCLUDED_TEXT[review][0]
    # TASK18-B-14B.2: one '항목 | 결과' table (5 rows). Stored values only, formatted as before.
    def width(mean,count):
        if excluded:
            return EXCLUDED_TEXT[review][1]
        return ("-" if mean is None else f"{_num(mean)} µm")+f" ({count if count is not None else '-'}개)"
    error=("-" if row["signed_error_pct"] is None else
           f"{row['signed_error_pct']:+.2f}% / {row['absolute_error_pct']:.2f}%")
    data_table(pd.DataFrame({"항목":["분석 상태","공식 채택","Line Width","Gap","Line 오차 (Signed / Abs)"],
        "결과":[label,"채택" if row["is_adopted"] else "미채택",width(row["line_mean_um"],row["line_measurement_count"]),
               width(row["gap_mean_um"],row["gap_measurement_count"]),error]}),height=(5+1)*31+5)
    render_analysis_details(view)


def render_analysis_details(view,where=""):
    """Existing detail expanders of ONE image's stored run; `where` (e.g. 'lt') is shown in the labels."""
    if view is None or view["state"]!="OK":
        return
    row=view["row"];review=row["outcome"]
    excluded=_pd_excluded(row) and row["line_measurement_count"] is None and row["gap_measurement_count"] is None
    suffix=f" · {where}" if where else ""
    if view["detail_error"]:
        st.error("결과 파일 검증 실패 · 상세 측정값 표시를 중단했습니다. "+view["detail_error"])
        return
    detail=view["detail"]
    detections=detail["detections"]
    if detections:
        with st.expander(f"결함 검출 상세{suffix} ({len(detections)}건)"):
            data_table(pd.DataFrame([{"class":d.get("class_code"),"confidence":d.get("confidence"),
                "bbox (x1,y1,x2,y2)":", ".join(f"{v:.1f}" for v in d.get("bbox_xyxy",[])),"source":d.get("source")}
                for d in detections]))
    if excluded:
        return  # Nothing was measured or judged; no empty PN/PT tables.
    with st.expander("Line 판정 (PN/PT)"+suffix):
        data_table(pd.DataFrame([{"Line":j.get("line_id"),"판정":j.get("label"),"근거":j.get("reason") or "-",
            "평균 (µm)":j.get("mean_um"),"최소":j.get("min_um"),"최대":j.get("max_um"),
            "near_threshold":j.get("near_threshold"),"pitch_ok":j.get("pitch_ok")} for j in detail["judgments"]]))
    with st.expander("개별 측정값"+suffix):
        data_table(pd.DataFrame([{"구분":m.get("measurement_type"),"ID":m.get("line_id"),"Y 위치":m.get("y_fraction"),
            "폭 (µm)":m.get("width_um"),"폭 (px)":m.get("width_px"),"y_px":m.get("y_px"),
            "left_x_px":m.get("left_x_px"),"right_x_px":m.get("right_x_px")} for m in detail["measurements"]]))


# ---- TASK18-B-15: per-stage overlay slots and the selected Die summary ------------------------------------
STAGE_LABELS={"metrology":"선폭 Overlay","defect":"결함 Overlay"}


def render_stage_overlay(view,stage,shown):
    """One stored overlay (선폭 or 결함) of the selected image; 0 detections and 'not generated' are kept apart."""
    label=STAGE_LABELS[stage]
    if view is None or view["state"]!="OK":
        st.info("Overlay 없음")
        st.caption(NO_RESULT_CAPTION if view is None or view["state"]=="NONE" else "분석 결과 표시 불가 · 원본만 표시")
        return
    if not shown:
        st.caption(label+" 표시 꺼짐")
        return
    item=view["overlays"].get(stage) or {}
    row=view["row"]
    if item.get("path"):
        st.image(item["path"],width="stretch")
        if stage=="defect" and view.get("detail"):
            count=len(view["detail"]["detections"])
            st.caption(f"결함 검출 {count}건" if count else "결함 검출 0건 · Overlay에 표시할 결함 없음")
    elif row[stage+"_overlay_path"] is None:
        st.info(label+" 미생성")
        if stage=="metrology" and _pd_excluded(row):
            st.caption(EXCLUDED_TEXT[row["outcome"]][1])
    else:
        st.warning(label+": "+item.get("error","표시할 수 없습니다."))


def _die_images(die):
    """Judged image IDs of the Die per position (same set as the Map colour); all IDs when not judged."""
    assessment=die.get("assessment")
    ids={p:(assessment["positions"][p]["image_ids"] if assessment else die["positions"][p]["image_ids"])
         for p in ("lt","c","rb")}
    analysis={}
    for entry in die["positions"].values():
        analysis.update(entry.get("analysis") or {})
    return ids,analysis


SUMMARY_ROWS=["분석 상태","Line Width 평균 (측정 수)","Gap 평균 (측정 수)","선폭 오차율 (Signed / Abs)","결함 종류","결함 개수"]
PD_RED="#e56666";NEUTRAL=None


def summarize(ids,analysis):
    """{position: [image_id]} -> table values from stored results; valid measurements only (TASK18-B-15/16).

    Same rules for the Die table (lt/c/rb) and the image table (one image): PD hold / PD exclusion /
    partial results never enter averages; defects = model-postprocessed detections only.
    """
    from collections import Counter
    from dashboard.process_analysis import DESIGN_LINE_UM
    lines=[];gaps=[];pn=pt=0;held=[];excluded=[];partial=[];missing=[];used=[];defects=Counter();where={}
    for position,image_ids in ids.items():
        for image_id in image_ids:
            item=analysis.get(image_id)
            if not item:
                missing.append(position);continue
            for d in item.get("defect_details") or []:
                if d.get("source")=="MODEL_POSTPROCESSED":   # Detected defects only (no rule-based suspicion).
                    defects[d["class_code"]]+=1;where.setdefault(d["class_code"],set()).add(position)
            outcome=item.get("outcome")
            if outcome=="POLICY_REVIEW_REQUIRED":
                held.append(position)
            elif outcome=="MODEL_PD_EXCLUDED":
                excluded.append(position)
            elif item.get("partial_measurement"):
                partial.append(position)
            elif item.get("metrology_status")=="SUCCESS":
                used.append(position)
                lines+=item.get("line_widths_um") or [];gaps+=item.get("gap_widths_um") or []
                labels=[j.get("label") for j in item.get("judgments") or []]
                pn+=labels.count("pn");pt+=labels.count("pt")
    notes=" · ".join(x for x in (("보류 "+", ".join(held)) if held else "",("제외 "+", ".join(excluded)) if excluded else "",
                                 ("부분 계측 제외 "+", ".join(partial)) if partial else "",
                                 ("미분석 "+", ".join(sorted(set(missing)))) if missing else "") if x)
    def no_value():
        return ("미분석" if missing and not (held or excluded or partial) else
                "계측 보류" if held else "계측 제외" if (excluded or partial) else "유효 계측 없음")
    def average(values,extra=""):
        if not values:
            return no_value()+(f" ({notes})" if notes and len(ids)>1 else "")
        return f"{sum(values)/len(values):.3f} µm ({len(values)}개)"+extra+(f" · {notes}" if notes else "")
    line_mean=sum(lines)/len(lines) if lines else None
    signed=(line_mean-DESIGN_LINE_UM)/DESIGN_LINE_UM*100 if line_mean is not None else None
    absolute=abs(signed) if signed is not None else None   # Same rule as the stored per-image absolute_percent.
    names={"o":"Open","s":"Short","pd":"Pattern Damage","p":"Particle","sc":"Scratch"}
    multi=len(ids)>1
    kinds=" · ".join(f"{names.get(c,c)}({c})"+(" "+", ".join(p for p in ("lt","c","rb") if p in where[c]) if multi else "")
                     for c in sorted(defects)) or ("검출 없음 · PD 의심 보류는 분석 상태에 표시" if held else
                                                   "미분석" if missing and not used and not held and not excluded and not partial
                                                   else "검출 없음")
    unanalysed=bool(missing) and not (used or held or excluded or partial)
    scope=("집계 "+", ".join(sorted(set(used),key=("lt","c","rb").index)) if used else "집계 위치 없음")+(f" · {notes}" if notes else "")
    return {"line":average(lines,f" · PN {pn} / PT {pt} Line"),"gap":average(gaps),
            "error":no_value() if signed is None else f"{signed:+.2f}% / {absolute:.2f}%",
            "kinds":kinds,"count":"미분석" if unanalysed else f"{sum(defects.values())}건"+(" ("+" · ".join(f"{c} {n}" for c,n in sorted(defects.items()))+")" if defects else ""),
            "pn":pn,"pt":pt,"held":held,"excluded":excluded,"scope":scope,"line_n":len(lines),"gap_n":len(gaps),
            "line_mean":line_mean}


def die_summary(die):
    """Selected Die (current filter, lt/c/rb) aggregated from stored results; valid measurements only."""
    ids,analysis=_die_images(die)
    return summarize(ids,analysis)


def image_summary(die,position,image_id):
    """The selected image only (same rows/rules as the Die table)."""
    _,analysis=_die_images(die)
    return summarize({position:[image_id]},analysis)


def image_status(die,image_id):
    """Selected image status: PD detection / PD-suspect hold -> red '사용 불가' (same wording as the Map)."""
    _,analysis=_die_images(die)
    item=analysis.get(image_id)
    if not item:
        return "미분석 · 저장된 분석 결과 없음","#b0bac7"
    outcome=item.get("outcome")
    if outcome=="MODEL_PD_EXCLUDED" or any(d.get("source")=="MODEL_POSTPROCESSED" and d.get("class_code")=="pd"
                                          for d in item.get("defect_details") or []):
        return "사용 불가 · PD 검출",PD_RED
    if outcome=="POLICY_REVIEW_REQUIRED":
        return "사용 불가 · PD 의심으로 선폭계측 보류",PD_RED
    if item.get("partial_measurement"):
        return "부분 계측 · 평균 제외",NEUTRAL
    return STATUS_LABELS.get(outcome,(str(outcome),""))[0],NEUTRAL


def render_summary_table(summary,status,color,*,key):
    frame=pd.DataFrame({"항목":SUMMARY_ROWS,
                        "결과":[status,summary["line"],summary["gap"],summary["error"],summary["kinds"],summary["count"]]})
    if color:
        text="#ffffff" if color.lower() in ("#e56666","#f39b49") else "#172b46"
        frame=frame.style.apply(lambda col:[f"background-color:{color};color:{text};font-weight:700"
                                            if i==0 and col.name=="결과" else "" for i in range(len(col))],axis=0)
    data_table(frame,height=(6+1)*31+5,key=key)
    st.caption(summary["scope"])


def render_die_summary(die,status,color):
    """'선택 Die 분석 요약 (lt/c/rb)' table; status text/colour are the Map's (same die_hover_lines and style)."""
    summary=die_summary(die)
    render_summary_table(summary,status,color,key="ia_die_summary_table")
    return summary


def render_image_summary(die,position,image_id):
    """'선택 이미지 분석 요약 (위치)': the selected image only."""
    summary=image_summary(die,position,image_id)
    status,color=image_status(die,image_id)
    render_summary_table(summary,status,color,key="ia_image_summary_table")
    return summary
