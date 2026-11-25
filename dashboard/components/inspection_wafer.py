"""Shared read-only Wafer navigation. No analysis, severity aggregation, or DB writes."""
from enum import Enum
from html import escape
import sqlite3
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_provider as data
from dashboard.sqlite_provider import runtime_provider
from dashboard.navigation import navigate

class PatternCriticality(str, Enum):
    NORMAL = "NORMAL"
    MINOR = "MINOR"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNUSABLE = "UNUSABLE"
    NOT_EVALUATED = "NOT_EVALUATED"

# Provisional vocabulary only. No mapping from 64 combinations to a severity.
CRITICALITY_STYLES = {
    PatternCriticality.NORMAL: ("정상", "#35a86b"),
    PatternCriticality.MINOR: ("경미", "#a5d46f"),
    PatternCriticality.WARNING: ("주의", "#f4d25a"),
    PatternCriticality.CRITICAL: ("심각", "#f39b49"),
    PatternCriticality.UNUSABLE: ("사용 불가", "#e56666"),
    PatternCriticality.NOT_EVALUATED: ("미평가", "#b0bac7"),
}
POSITION_ORDER = ("lt", "c", "rb")
SINGLE_IMAGE_MODE = "single_image"  # Excluded from the Wafer Map; shown in the I&A Image tab.
DIE_CODES = tuple(f"{row}{col}" for row in range(1,4) for col in range(1,9))


def set_selection(state, regions, die_id=None, position=None):
    """Legacy Mock image selector state, independent of the new Die preview."""
    same = (die_id is not None and state.get("selected_die_id") == die_id
            and state.get("selected_position") == position)
    selected = None if same or die_id is None else next(
        r for r in regions if r["die_id"] == die_id and r["position"] == position)
    state["selected_die_id"] = selected["die_id"] if selected else None
    state["selected_position"] = selected["position"] if selected else None
    image_id = selected["image_id"] if selected else regions[0]["image_id"]
    state["selected_image_id"] = image_id
    state["_pending_map_image"] = image_id


def sync_mock_image(batch_id, image_key="ia_image"):
    """Keep the existing lower Mock image selector/Overlay behavior unchanged."""
    regions=data.inspection_wafer_regions(batch_id)
    if (st.session_state.get("_inspection_map_batch") != batch_id
            or st.session_state.get("_inspection_map_layout") != "8cols_3rows"):
        st.session_state["_inspection_map_batch"]=batch_id
        st.session_state["_inspection_map_layout"]="8cols_3rows"
        set_selection(st.session_state,regions)
    image_id=st.session_state.get(image_key)
    if ("_pending_map_image" not in st.session_state and image_id
            and image_id != st.session_state.get("selected_image_id")):
        match=next((r for r in regions if r["position"]=="c" and r["image_id"]==image_id),None)
        if match:
            set_selection(st.session_state,regions,match["die_id"],"c")


def analysis_texts(item):
    """Display text for one stored (AI, unadopted) result; values copied, never re-judged."""
    if item is None:
        return "미분석","미분석"
    defect=f"{len(item['defect_classes'])}건"+(" ("+", ".join(sorted(set(item["defect_classes"])))+")" if item["defect_classes"] else "")
    outcome=item["outcome"]
    if outcome=="MODEL_PD_EXCLUDED":
        return "PD 검출 · 계측 제외",defect
    if outcome=="POLICY_REVIEW_REQUIRED":
        return "PD 의심 · 계측 보류(정책 검토)",defect
    if item["metrology_status"]!="SUCCESS":
        return "계측 불가 ("+str(item["metrology_status"] or outcome)+")",defect
    widths=f"Line {item['line_mean_um']:.3f} / Gap {item['gap_mean_um']:.3f} µm" if (
        item["line_mean_um"] is not None and item["gap_mean_um"] is not None) else "평균 없음"
    if item["partial_measurement"]:
        return "부분 계측 · "+widths+" (평균 미채택)",defect
    return widths+(" · 검토 필요" if outcome=="QUALITY_REVIEW_REQUIRED" else ""),defect


def die_snapshot(wafer_id, sites, images, analysis=None, *, exclude_from_judgment=(), judge=True):
    """24 display slots; actual Site IDs/Images only. Empty slots are not DB records.

    analysis (optional) = {image_id: latest stored result}. With it, each Die also gets an
    AI provisional assessment (policy v0.1) and an official one (is_adopted=1 results only).
    criticality is the displayed colour; it stays NOT_EVALUATED until a view is applied.
    """
    from dashboard.die_assessment import assess_die
    modes={r["image_id"]:r.get("upload_mode") for r in images}
    adopted={k:v for k,v in (analysis or {}).items() if v and v.get("is_adopted")==1}
    site_lookup={(r["die_code"],r["position_code"]):r for r in sites}
    result=[]
    for code in DIE_CODES:
        positions={};counts={"images":0,"analysed":0,"review":0,"pd":0}
        for position in POSITION_ORDER:
            site=site_lookup.get((code,position),{})
            ids=[r["image_id"] for r in images
                 if r["wafer_id"]==wafer_id and r["die_code"]==code and r["position_code"]==position]
            entry={"physical_site_id":site.get("physical_site_id"),"image_ids":ids,"metrology":"결과 없음","defect":"결과 없음"}
            if analysis is not None:
                texts=[analysis_texts(analysis.get(image_id)) for image_id in ids]
                entry["analysis"]={image_id:analysis.get(image_id) for image_id in ids}
                entry["metrology"]=" / ".join(t[0] for t in texts) if texts else "이미지 없음"
                entry["defect"]=" / ".join(t[1] for t in texts) if texts else "이미지 없음"
                for image_id in ids:
                    item=analysis.get(image_id);counts["images"]+=1
                    if item:
                        counts["analysed"]+=1
                        counts["review"]+=item["outcome"]!="COMPLETE"
                        counts["pd"]+=item["outcome"] in ("MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED")
            positions[position]=entry
        die={"wafer_id":wafer_id,"die_code":code,"row":int(code[0]),"col":int(code[1]),
             "criticality":"NOT_EVALUATED","positions":positions,
             "analysis_counts":counts if analysis is not None else None}
        if analysis is not None and judge:
            # Same-SHA duplicates (other Batches) stay listed but are judged once.
            ids={pos:[i for i in entry["image_ids"] if i not in exclude_from_judgment] for pos,entry in positions.items()}
            die["assessment"]=assess_die(ids,analysis,modes)
            # TASK18-B-14B: AI view colour = 위험도 기준 v0.1 (map_risk). The v0.1 assessment itself, PASS/FAIL,
            # Row and the official view are unchanged; the former colour is kept for comparison only.
            from dashboard.die_assessment import map_risk
            risk=map_risk(die["assessment"])
            die["assessment"].update(legacy_display_risk=die["assessment"]["display_risk"],map_risk=risk,
                                     display_risk=risk["display_risk"])
            # TASK18-B-15: stored PD detection / PD-suspect line-metrology hold at any lt/c/rb -> UNUSABLE (red),
            # ahead of other positions' NOT_EVALUATED. Display only; score and PN/PT bounds unchanged.
            from dashboard.die_assessment import pd_unusable
            pd_state=pd_unusable(die["assessment"])
            die["assessment"]["pd_unusable"]=pd_state
            if pd_state["unusable"]:
                die["assessment"]["display_risk"]="UNUSABLE"
            die["official"]=assess_die(ids,adopted,modes)
        elif analysis is not None:
            die["judgment_blocked"]="서로 다른 노광/PEB 조건 포함 · 조건 선택 필요"
        result.append(die)
    return result


JUDGMENT_VIEWS={"AI 잠정 위험도":"assessment","공식 판정":"official"}


def apply_judgment_view(dies,view):
    """Colour = selected view's display risk; Dies without an assessment stay NOT_EVALUATED."""
    key=JUDGMENT_VIEWS.get(view,"assessment")
    for die in dies:
        die["criticality"]=die[key]["display_risk"] if die.get(key) else "NOT_EVALUATED"
        die["view_key"]=key if die.get(key) else None
    return dies


LEVEL_NAMES={k.value:label for k,(label,_) in CRITICALITY_STYLES.items()}
POSITIONS_JUDGED=("lt","c","rb")


def pd_reason(pd_state):
    """('사용 불가 · ...' reason, related positions/counts) for a PD-unusable Die."""
    reasons=[];where=[]
    if pd_state["detected"]:
        reasons.append("PD 검출")
        where.append(" · ".join(pd_state["detected"])+(f" (PD {pd_state['pd_count']}건)" if pd_state["pd_count"] else ""))
    if pd_state["held"]:
        reasons.append("PD 의심으로 선폭계측 보류")
        where.append(("보류 " if pd_state["detected"] else "")+" · ".join(pd_state["held"]))
    return "사용 불가 · "+" · ".join(reasons)," / ".join(where)


def risk_basis(assessment):
    """Actual evidence behind a scored colour, decisive part first: CD error, PN/PT Lines, O/S detections."""
    risk=assessment.get("map_risk") or {}
    cd=None;pnpt=[];found={};counts={}
    for position in POSITIONS_JUDGED:
        images=list(assessment["positions"][position]["images"].values())
        measured=[a for a in images if a["cd_state"]=="EVALUATED"]
        for a in measured:
            if a["cd_worst_error_pct"] is not None and (cd is None or a["cd_worst_error_pct"]>cd[0]):
                cd=(a["cd_worst_error_pct"],position)
        count=max((a["pn"]+a["pt"] for a in measured),default=0)
        if count:
            pnpt.append(f"{position} {count}")
        for a in images:
            for d in a["defects"]:
                if d.get("source")=="MODEL_POSTPROCESSED" and d.get("class_code") in ("o","s","pd"):
                    code=d["class_code"].upper()
                    found.setdefault(code,set()).add(position);counts[code]=counts.get(code,0)+1
    parts={"cd":f"CD 최대 오차 {cd[0]:.1f}% ({cd[1]})" if cd else None,
           "pnpt":(f"PN/PT Line {risk['pnpt_count']}개"+(" ("+" · ".join(pnpt)+")" if pnpt else ""))
                  if risk.get("pnpt_count") else None}
    s_cd=-1 if risk.get("s_cd") is None else risk["s_cd"]
    s_pnpt=-1 if risk.get("s_pnpt") is None else risk["s_pnpt"]
    text=[parts[k] for k in (("cd","pnpt") if s_cd>=s_pnpt else ("pnpt","cd")) if parts[k]]
    for code in ("O","S","PD"):
        if code in found:
            text.append(f"{code} {counts[code]}건 ("+" · ".join(p for p in POSITIONS_JUDGED if p in found[code])+")")
    return " · ".join(text) or "근거 없음"


def die_hover_lines(die):
    """Hover (TASK18-B-15): 'Wafer · Die' / risk or unusable reason / related positions·counts (max 3 lines)."""
    head=f"{die['wafer_id']} · Die {die['die_code']}"
    key=die.get("view_key") or ""
    assessment=die.get(key)
    if die.get("judgment_blocked"):
        return [head,"미평가 · "+die["judgment_blocked"]]
    if not assessment:
        return [head,"미평가 · 저장된 분석 결과 없음"]
    pd_state=assessment.get("pd_unusable") if key=="assessment" else None
    if pd_state and pd_state["unusable"]:
        return [head,*pd_reason(pd_state)]
    level=assessment["display_risk"]
    if level=="NOT_EVALUATED":
        risk=assessment.get("map_risk") or {}
        missing=assessment["missing_positions"]
        incomplete=[p for p in POSITIONS_JUDGED
                    if assessment["positions"][p]["present"] and not assessment["positions"][p]["complete"]]
        where=" / ".join(x for x in (("이미지 없음 "+" · ".join(missing)) if missing else "",
                                     ("평가 미완료 "+" · ".join(incomplete)) if incomplete else "") if x)
        reason="미평가 · 유효 계측 없음" if risk.get("state")=="NO_VALID_MEASUREMENT" else "미평가 · 평가 미완료"
        return [head,reason]+([where] if where else [])
    if key!="assessment":
        return [head,f"공식 {LEVEL_NAMES.get(level,level)} ({level})"]
    return [head,f"위험도 {LEVEL_NAMES.get(level,level)} ({level})",risk_basis(assessment)]


def die_hover(die):
    """One-line text of the hover reason (without the 'Wafer · Die' head)."""
    return " · ".join(die_hover_lines(die)[1:])


def risk_text(risk):
    """Short breakdown of map_risk (same values as the colour)."""
    if risk["score"] is None:
        return "유효 계측 없음 · 위험도 산출 불가" if risk["state"]=="NO_VALID_MEASUREMENT" else "lt/c/rb 평가 미완료"
    cd="-" if risk["s_cd"] is None else f"{risk['s_cd']}"
    pnpt="-" if risk["s_pnpt"] is None else f"{risk['s_pnpt']}"
    return f"점수 {risk['score']} = max(CD {cd}, PN/PT {pnpt}) + O/S/PD {risk['ospd']}"


def mock_snapshot(batch_id):
    """Existing shared-Die illustrations only, never masquerading as acquisition files."""
    batch=data.batch(batch_id)
    images=[];sites=[]
    for region in data.inspection_wafer_regions(batch_id):
        images.append({"wafer_id":batch["wafer_id"],"die_code":region["die_id"],
                       "position_code":region["position"],"image_id":region["image_id"]})
    return die_snapshot(batch["wafer_id"],sites,images)


def build_inspection_wafer(batch_id=None, selected_die_id=None, selected_position=None, *, dies=None,height=230):
    """One clickable, uniformly colored rectangle per Die inside a circular outline."""
    dies=mock_snapshot(batch_id) if dies is None else dies
    fig=go.Figure()
    for criticality,(_,color) in CRITICALITY_STYLES.items():
        part=[r for r in dies if r["criticality"]==criticality.value]
        if not part:
            continue
        fig.add_trace(go.Bar(x=[(r["col"]-4.5)*1.05 for r in part],
            y=[.84]*len(part),base=[3.5+(2-r["row"])*1.5-.42 for r in part],width=.8,
            marker=dict(color=color,line=dict(color="#71839a",width=1)),
            selected=dict(marker=dict(opacity=1)),unselected=dict(marker=dict(opacity=1)),
            customdata=[[r["die_code"],r["wafer_id"],
                         sum(len(p["image_ids"]) for p in r["positions"].values()),
                         die_hover(r),"<br>".join(escape(line) for line in die_hover_lines(r))] for r in part],
            name=criticality.value,
            # TASK18-B-15: max 3 lines; no score formula, internal codes or image reference counts.
            hovertemplate="%{customdata[4]}<extra></extra>"))
    fig.add_shape(type="circle",x0=-4.65,x1=4.65,y0=-1.15,y1=8.15,
                  line=dict(color="#8498b6",width=2),fillcolor="rgba(114,137,175,.08)",layer="below")
    for die in dies:
        cx=(die["col"]-4.5)*1.05;cy=3.5+(2-die["row"])*1.5
        fig.add_annotation(x=cx,y=cy,text=die["die_code"],showarrow=False,
                           font=dict(size=10,color="#172b46"))
        if die["die_code"]==selected_die_id:
            fig.add_shape(type="rect",x0=cx-.4,x1=cx+.4,y0=cy-.42,y1=cy+.42,
                          line=dict(color="#568ee9",width=3))
    fig.update_layout(height=height,barmode="overlay",showlegend=False,
        clickmode="event+select",dragmode=False,margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[-4.9,4.9],visible=False,fixedrange=True,constrain="domain")
    fig.update_yaxes(range=[-1.4,8.4],visible=False,fixedrange=True,scaleanchor="x",scaleratio=1)
    return fig


def _remember(widget, saved):
    st.session_state[saved]=st.session_state[widget]


def wafer_context(prefix, *, source_slot=None, wafer_slot=None, batch_filter=False, mock_batch=None):
    """Shared Wafer context. The operational Wafer Map is SQLite only (no source selector).

    Mock remains available only when explicitly set in session (_wafer_source="Mock", e.g. tests);
    an SQLite failure is shown as an error and never replaced by Mock data.
    """
    source="Mock" if st.session_state.get("_wafer_source")=="Mock" else "SQLite"
    st.session_state["_wafer_source"]=source
    provider=None
    if source=="SQLite":
        try:
            provider=runtime_provider()
            wafers=provider.wafers()["wafer_id"].tolist()
            batches=provider.batches()
            # Single Images are shown only in the I&A Image tab, never on the Wafer Map.
            batches=batches[batches["upload_mode"]!=SINGLE_IMAGE_MODE]
            mapped=({w for ids in batches["wafer_ids"] for w in ids} if "wafer_ids" in batches
                    else set(batches["wafer_id"]))
            wafers=[w for w in wafers if w in mapped]
        except (sqlite3.Error,OSError,RuntimeError,ValueError) as exc:
            st.error("SQLite Wafer 조회 실패 · Mock으로 대체하지 않습니다. "+str(exc))
            return None
    else:
        batches=data.batches("Wafer Batch")
        wafers=batches["wafer_id"].drop_duplicates().tolist()
    if not wafers:
        st.info("등록된 SQLite Wafer 없음 · Mock 데이터를 대신 표시하지 않습니다.")
        return None
    wafer_slot,*condition_slots=st.columns(4,gap="small") if provider else (wafer_slot or st,)
    saved_key="_wafer_selected_"+source
    default=data.batch(mock_batch)["wafer_id"] if mock_batch and source=="Mock" else wafers[0]
    wafer=st.session_state.get(saved_key,default)
    wafer=wafer if wafer in wafers else wafers[0]
    wafer_key=prefix+"_wafer_"+source
    st.session_state[wafer_key]=wafer
    wafer_slot=wafer_slot or st
    wafer=wafer_slot.selectbox("Wafer ID" if provider else "Wafer 선택",wafers,key=wafer_key,on_change=_remember,
        args=(wafer_key,saved_key),label_visibility="visible" if (batch_filter or provider) else "collapsed")
    st.session_state[saved_key]=wafer
    # v6: a Batch belongs to every Wafer it holds (not only its primary Wafer).
    batches=(batches[batches["wafer_ids"].map(lambda ids:wafer in ids)] if "wafer_ids" in batches
             else batches[batches["wafer_id"]==wafer])
    batch_id=None
    if provider:
        images=[]
        offset=0
        while True:
            page=provider.images(None,wafer_id=wafer,limit=100,offset=offset).to_dict("records")
            images.extend(r for r in page if r.get("upload_mode")!=SINGLE_IMAGE_MODE)
            if len(page)<100:
                break
            offset+=100
        scope=condition_filters(prefix,source,wafer,images,batches.to_dict("records"),slots=condition_slots)
        images=scope["images"];batch_id=scope["batch_id"]
        analysis=provider.latest_analysis(wafer)
        excluded=duplicate_exclusions(images,analysis)
        dies=die_snapshot(wafer,provider.wafer_sites(wafer),images,analysis,
                          exclude_from_judgment=excluded,judge=not scope["mixed"])
        return {"source":source,"provider":provider,"wafer_id":wafer,"batch_id":batch_id,"dies":dies,
                "images":images,"batches":batches.to_dict("records"),"filter_key":scope["key"],
                "scope_label":scope["label"],"mixed":scope["mixed"],"duplicates_excluded":len(excluded)}
    if batch_filter:
        names={r["batch_id"]:r.get("display_name",r["batch_id"]) for r in batches.to_dict("records")}
        batch_id=st.selectbox("Batch 필터", [None,*names],format_func=lambda key:"전체 Batch" if key is None else names[key],
            key=f"{prefix}_batch_{source}_{wafer}")
    # Existing Mock Wafer has one illustrated Batch. No synthetic acquisition data.
    mock_batch_id=batch_id or batches.iloc[0]["batch_id"]
    dies=mock_snapshot(mock_batch_id)
    images=data.images(mock_batch_id).to_dict("records")
    return {"source":source,"provider":provider,"wafer_id":wafer,"batch_id":batch_id,
            "dies":dies,"images":images,"batches":batches.to_dict("records"),"filter_key":"mock",
            "scope_label":wafer+" / Mock","mixed":False,"duplicates_excluded":0}


def exposure_label(value,unit):
    """I&A condition label; exposure shown in seconds (s) (display only, TASK18-B-18)."""
    from dashboard.units import exposure_display
    return exposure_display(value,unit,missing="노광 미입력")


def peb_label(value):
    return "PEB 미입력" if value is None or value!=value else f"{value:g} s"


def _numeric_key(label):
    head=label.split(" ")[0]
    try:
        return (0,float(head))
    except ValueError:
        return (1,label)  # "미입력" sorts after real values.


def _choice(label,options,key,format_func):
    """Selectbox whose stale value (from another Wafer/condition) is dropped, never reused."""
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key,None)
    return st.selectbox(label,options,key=key,format_func=format_func)


def condition_filters(prefix,source,wafer,images,batches,*,slots=None):
    """Wafer -> exposure -> PEB -> Batch; each list shows only values present above it.

    None means "전체". When the final scope spans >1 exposure or PEB condition, Die judgment is
    blocked (mixed=True) so different process conditions are never merged into one Die.
    """
    for record in images:
        record["_exposure"]=exposure_label(record.get("exposure_value"),record.get("exposure_unit"))
        record["_peb"]=peb_label(record.get("peb_temperature_c"))
    exposure_col,peb_col,batch_col=slots or st.columns(3,gap="small")
    exposures=sorted({r["_exposure"] for r in images},key=_numeric_key)
    with exposure_col:
        exposure=_choice("노광",([None] if len(exposures)>1 else [])+exposures,f"{prefix}_exposure_{source}_{wafer}",
                         lambda v:"전체 (조건 혼합)" if v is None else v) if exposures else None  # Label says 노광.
    scoped=apply_condition_scope(images,exposure=exposure)["images"]
    pebs=sorted({r["_peb"] for r in scoped},key=_numeric_key)
    with peb_col:
        peb=_choice("PEB",([None] if len(pebs)>1 else [])+pebs,f"{prefix}_peb_{source}_{wafer}",
                    lambda v:"전체 (조건 혼합)" if v is None else v) if pebs else None
    scoped=apply_condition_scope(images,exposure=exposure,peb=peb)["images"]
    # Short control label (automatic name + ID); the full user name is shown in the scope caption.
    names={r["batch_id"]:r.get("automatic_display_name") or r.get("display_name") or r["batch_id"] for r in batches}
    full={r["batch_id"]:r.get("display_name") or r["batch_id"] for r in batches}
    counts={}
    for record in scoped:
        counts[record["batch_id"]]=counts.get(record["batch_id"],0)+1
    batch_ids=[b for b in names if b in counts]
    with batch_col:
        batch_id=_choice("Batch",[None,*batch_ids],f"{prefix}_batch_{source}_{wafer}",
            lambda v:"전체 Batch" if v is None else f"{names[v]} · {v[6:14]} ({counts[v]}장)") if batch_ids else None
    result=apply_condition_scope(images,exposure=exposure,peb=peb,batch_id=batch_id)
    final,mixed=result["images"],result["mixed"]
    label=(f"{wafer} / "+(exposure and "노광 "+exposure or "전체 노광")+" / "+(peb or "전체 PEB")+" / "+
           (f"{full[batch_id]} · {batch_id[6:14]}" if batch_id else f"전체 Batch {len({r['batch_id'] for r in final})}개"))
    return {"images":final,"batch_id":batch_id,"mixed":mixed,"label":label,
            "key":"|".join(str(v) for v in (exposure,peb,batch_id))}


def apply_condition_scope(images,*,exposure=None,peb=None,batch_id=None):
    """Pure filter: None = all. mixed=True when >1 exposure/PEB condition remains."""
    final=[]
    for record in images:
        exposure_text=exposure_label(record.get("exposure_value"),record.get("exposure_unit"))
        peb_text=peb_label(record.get("peb_temperature_c"))
        if ((exposure is None or exposure_text==exposure) and (peb is None or peb_text==peb)
                and (batch_id is None or record["batch_id"]==batch_id)):
            final.append(record)
    conditions={(exposure_label(r.get("exposure_value"),r.get("exposure_unit")),peb_label(r.get("peb_temperature_c")))
                for r in final}
    return {"images":final,"mixed":len(conditions)>1,"conditions":sorted(conditions)}


def duplicate_exclusions(images,analysis):
    """Same capture (SHA-256) in several selected Batches is judged once; nothing is deleted.

    Representative = the copy with the most recent stored analysis, else the earliest image ID.
    """
    groups={}
    for record in images:
        if record.get("sha256"):
            groups.setdefault(str(record["sha256"]).lower(),[]).append(record["image_id"])
    excluded=set()
    for ids in groups.values():
        if len(ids)<2:
            continue
        analysed=[i for i in ids if analysis.get(i)]
        keep=max(analysed,key=lambda i:(analysis[i]["started_at"] or "",i)) if analysed else min(ids)
        excluded.update(i for i in ids if i!=keep)
    return excluded


def select_die(state,prefix,code,*,popover=False):
    if code not in DIE_CODES:
        return
    state[prefix+"_selected_die"]=code
    state[prefix+"_revision"]=state.get(prefix+"_revision",0)+1
    if popover:
        state[prefix+"_popover"]=True


def die_status(die,source):
    st.markdown(f"**Wafer {escape(die['wafer_id'])} · Die {die['die_code']}**")
    assessment=die.get("assessment")
    if not assessment:
        st.caption(source+" · NOT_EVALUATED / 미판정 · 등록 여부는 분석 결과가 아닙니다.")
    else:
        from dashboard.die_assessment import POLICY_VERSION,DEFECT_NAMES
        official=die.get("official") or {}
        st.caption(f"{POLICY_VERSION} · 공식 판정 {official.get('display_risk','NOT_EVALUATED')} (채택 결과 없으면 미평가)")
        risk=assessment.get("map_risk") or {}
        st.markdown(f"AI 잠정 위험도 **{assessment['display_risk']}**  \n"
                    f"O/S/PD 기준 판정 **{assessment['decision']}**"+(" · FAIL 예상" if assessment["expected_fail"] else "")+
                    (" · 최종 평가 완료" if assessment["color_complete"] else " · 최종 평가 미완료"))
        pd_state=assessment.get("pd_unusable") or {}
        if pd_state.get("unusable"):
            reason,where=pd_reason(pd_state)
            st.markdown(f":red[**{reason}**] · {where}")
        if risk:
            error=risk["cd_worst_error_pct"]
            st.caption("색상 근거: "+risk_text(risk)+" · CD 최대 오차 "+(f"{error:.1f}%" if error is not None else "없음")+
                       " · PN/PT "+("산출 불가" if risk["pnpt_count"] is None else f"{risk['pnpt_count']}개")+
                       (" (PD 계측 제외 이미지 미포함)" if risk["pnpt_partial"] and risk["pnpt_count"] is not None else "")+
                       f" · O/S/PD 검출 이미지 {risk['ospd_images']}장"
                       +(" ("+", ".join(DEFECT_NAMES[c]+f"({c})" for c in assessment["critical_defects"])+")" if assessment["critical_defects"] else ""))
        for reason in assessment["reasons"]+assessment["review"]+assessment["conflicts"]:
            st.caption("· "+reason)
        counts=die.get("analysis_counts")
        if counts:
            st.caption(f"AI 분석 {counts['analysed']}/{counts['images']}장 · 공식 채택 아님 · 결함 0건은 PASS 아님")
    for position in POSITION_ORDER:
        site=die["positions"][position]
        risk=""
        if assessment:
            # TASK18-B-14B: the same 기준 v0.1 inputs as the Die colour (not the former per-image grade).
            from dashboard.die_assessment import cd_score
            judged=assessment["positions"][position]
            measured=[a for a in judged["images"].values() if a["cd_state"]=="EVALUATED"]
            errors=[a["cd_worst_error_pct"] for a in measured if a["cd_worst_error_pct"] is not None]
            parts=[f"CD 최대 {max(errors):.1f}% ({cd_score(max(errors))}점)" if errors else "CD 없음",
                   f"PN/PT {max(a['pn']+a['pt'] for a in measured)}" if measured else "PN/PT 산출 불가"]
            if judged["critical_defects"]:
                parts.append("O/S/PD "+"/".join(judged["critical_defects"]))
            risk=((" · "+" · ".join(parts)+("" if judged["complete"] else " (평가 미완료)")) if judged["present"]
                  else " · 판정 대상 없음")
        st.markdown(f"**{position}** · Metrology {site['metrology']} / Defect {site['defect']}{risk}  \n"
                    f"이미지 {'참조' if source=='Mock' else '등록'} {len(site['image_ids'])}장")


def render_die_map(context,prefix, *, preview=False,popover_slot=None,view_slot=None):
    scope=f"{prefix}_{context['source']}_{context['wafer_id']}_{context['batch_id'] or 'all'}"
    # A changed exposure/PEB/Batch filter invalidates the selected Die and its image widgets.
    filter_key=context.get("filter_key")
    if st.session_state.get(scope+"_filter")!=filter_key:
        if scope+"_filter" in st.session_state:
            st.session_state.pop(scope+"_selected_die",None)
            st.session_state.pop(scope+"_popover",None)
            st.session_state[scope+"_revision"]=st.session_state.get(scope+"_revision",0)+1
        st.session_state[scope+"_filter"]=filter_key
    selected=st.session_state.get(scope+"_selected_die")
    if selected not in DIE_CODES:
        # TASK18-B-14B: open on the first Die that has registered images (detail: first Die as before).
        with_images=[d["die_code"] for d in context["dies"] if any(p["image_ids"] for p in d["positions"].values())]
        selected=(with_images[0] if with_images else None) if preview else DIE_CODES[0]
        if selected:
            st.session_state[scope+"_selected_die"]=selected
    revision=st.session_state.get(scope+"_revision",0)
    if preview:
        context["selection_revision"]=revision
    chart_key=scope+"_chart_"+str(revision)
    def clicked():
        points=st.session_state[chart_key]["selection"]["points"]
        if points:
            code=points[-1]["customdata"][0]
            select_die(st.session_state,scope,code,popover=preview)
    if preview:
        # TASK18-B-15: I&A shows the AI provisional risk only (no AI / official switch).
        apply_judgment_view(context["dies"],"AI 잠정 위험도")
    elif any(d.get("assessment") for d in context["dies"]):
        view=(view_slot or st).radio("판정 표시",list(JUDGMENT_VIEWS),horizontal=True,key=scope+"_judgment_view",
                      label_visibility="collapsed",help="AI 잠정 위험도(정책 v0.1)와 공식 판정(is_adopted=1만)을 구분합니다.")
        apply_judgment_view(context["dies"],view)
    # TASK18-B-14B: the I&A map uses the detail page's size (same geometry, clearer Die boxes).
    fig=build_inspection_wafer(dies=context["dies"],selected_die_id=selected,height=410)
    if preview:
        fig.update_annotations(font_size=12)
    with st.container(key="ia_wafer_visual" if preview else "wafer_detail_visual"):
        st.plotly_chart(fig,key=chart_key,on_select=clicked,selection_mode="points",
                        theme="streamlit",width="stretch",config={"displayModeBar":False})
    if preview and selected:
        die=next(r for r in context["dies"] if r["die_code"]==selected)
        st.markdown('<div class="ia-die-summary"><b>Die '+escape(selected)+'</b> · '+escape(die_hover(die))+'</div>',
                    unsafe_allow_html=True)
    criticality_legend(key=scope+"_legend")
    if preview:
        with popover_slot or st:
            with st.popover("Die 상태",key=scope+"_popover",on_change="rerun",width="stretch"):
                with st.container(key="wafer_die_popover_content"):
                    if selected:
                        die_status(next(r for r in context["dies"] if r["die_code"]==selected),context["source"])
                    else:
                        st.caption("Die 네모를 클릭하면 lt/c/rb 상태를 확인할 수 있습니다.")
    if any(d.get("assessment") for d in context["dies"]):
        st.caption("AI 잠정 위험도 기준 v0.1 · 공식 품질 아님 · 빨강: 사용 불가 (PD 검출·PD 의심 보류) · 회색: 미평가")
    else:
        st.caption("회색: 미판정 · 미등록 위치도 표시")
    render_scope_summary(context)
    return next((r for r in context["dies"] if r["die_code"]==selected),None)


def render_scope_summary(context):
    """Selected scope and its AI provisional decision/Fail Rate (same Dies as the map colours)."""
    if context["source"]!="SQLite":
        return
    st.caption("선택 범위: "+context["scope_label"])
    if not context["images"]:
        st.info("해당 조건 데이터 없음 · 색상·판정·Fail Rate 없음")
        return
    if context.get("mixed"):
        st.warning("서로 다른 노광/PEB 조건이 포함되어 Die 판정을 합산하지 않습니다 · 조건을 선택하세요.")
        return
    if context.get("duplicates_excluded"):
        st.caption(f"동일 촬영 이미지(SHA) 중복 {context['duplicates_excluded']}장은 판정에서 1장으로 계산 "
                   "(최신 분석 결과 우선) · 원본·이력은 유지")
    if any(d.get("assessment") for d in context["dies"]):
        from dashboard.die_assessment import summarize_wafer,rate_text
        s=summarize_wafer(context["dies"])
        st.caption(f"AI 잠정 · Die PASS {s['PASS']} / FAIL {s['FAIL']} / PENDING {s['PENDING']} · "
                   f"Die Fail Rate {rate_text(s['die_fail_rate'])} · Row {s['row_PASS']}/{s['row_FAIL']}/{s['row_PENDING']} "
                   f"(PASS/FAIL/PENDING) · Row Fail Rate {rate_text(s['row_fail_rate'])}")


LEGEND_SCORES={"NORMAL":"0점","MINOR":"1점","WARNING":"2점","CRITICAL":"3점"}


def criticality_legend(*,preview=False,key="wafer_legend"):
    """Collapsed '위험도 범례': colours, scores and the 기준 v0.1 rules (values shared with map_risk)."""
    from dashboard.die_assessment import CD_BOUNDS
    with st.expander("위험도 범례",expanded=False,key=key):
        chips="".join('<span class="wafer-legend-item" role="listitem"><i style="background:'+color+'"></i>'+
                      escape(k.value+" · "+label+(" · "+LEGEND_SCORES[k.value] if k.value in LEGEND_SCORES else ""))+"</span>"
                      for k,(label,color) in CRITICALITY_STYLES.items())
        st.markdown('<div class="wafer-legend" role="list" aria-label="위험도 색상 범례">'+chips+"</div>",unsafe_allow_html=True)
        a,b,c=CD_BOUNDS
        st.markdown(
            "**판정 방법** · Die의 lt/c/rb 3장을 함께 봅니다.  \n"
            f"- **선폭 오차** (설계 10 µm 대비, 가장 큰 Line 오차): {a}% 이하 0점 · {a}–{b}% 1점 · {b}–{c}% 2점 · {c}% 초과 3점  \n"
            "- **PN/PT 개수** (계측이 Line을 가늘게/굵게 판정한 수, 3장 합계): 0개 0점 · 1–2개 1점 · 3–5개 2점 · 6개 이상 3점  \n"
            "- **O/S/PD 검출**: 3장 중 한 장이라도 있으면 +1점 (여러 장·여러 개여도 +1점만)  \n"
            "- **최종** = 선폭·PN/PT 중 높은 점수 + O/S/PD 가산 (최대 3점)")
        st.caption("왜 이렇게? 선폭 오차와 PN/PT는 같은 문제를 다른 방식으로 본 것이라 높은 쪽만 쓰고, O/S/PD는 성격이 다른 결함이라 "
                   "한 단계 올립니다. 같은 결함을 여러 사진에서 중복으로 더하지 않아 위험도가 과도하게 오르지 않게 했습니다.")
        st.markdown("- **빨강 · 사용 불가 (UNUSABLE)**: lt/c/rb 중 한 곳이라도 저장된 결과에 PD 검출이 있거나 "
                    "'PD 의심으로 선폭계측 보류'가 기록된 Die. 다른 위치의 미평가·평가 미완료보다 우선합니다.  "+chr(10)+
                    "- **회색 · 미평가 (NOT_EVALUATED)**: PD 근거 없이 lt/c/rb 중 분석이 없거나 미완료, 또는 유효 계측값이 없는 Die.")
        st.caption("PD 의심 보류는 PD 검출 건수에 합산하지 않습니다 · P/SC는 점수에 넣지 않고 검토 항목으로 표시합니다.")
        st.caption("본 기준은 AI 잠정 위험도 표시용이며 공식 공정 합격·불합격 기준이 아닙니다. "
                   "임계값은 합의된 시범 후보이며 통계적으로 검증된 공정 허용오차가 아닙니다.")


def inspection_wafer(batch_id):
    """I&A preview: keep the popup and pass its exact context/Die to the image column."""
    title,view_slot,popover_slot=st.columns([1.1,2,1],gap="small",vertical_alignment="center")
    context=wafer_context("ia_wafer",mock_batch=batch_id)
    if title.button("Wafer Map",key="ia_wafer_title",type="tertiary",help="현재 탭에서 Wafer Map 상세 열기"):
        navigate("wafer_map",_wafer_source=context["source"] if context else "SQLite",
                 **({"_wafer_selected_"+context["source"]:context["wafer_id"]} if context else {}))
    selected=render_die_map(context,"ia_wafer_map",preview=True,popover_slot=popover_slot,view_slot=view_slot) if context else None
    return context,selected


def render_site_images(context,die):
    """Wafer Map detail (TASK18-B-14B): the selected Die's image first; position/image choices in a collapsed menu.

    Existing registered originals only, selected by exact IDs; overlays stay unconnected on this page.
    """
    from dashboard.components.sqlite_input import render_original_preview
    from dashboard.components.images import image_panel
    from dashboard.units import exposure_display
    scope=f"{context['source']}_{context['wafer_id']}_{context['batch_id'] or 'all'}_{die['die_code']}"
    head=st.container();image_slot=st.container(key="wafer_detail_image");status_slot=st.container()
    st.session_state["wafer_browser_image_id"]=None
    record=None;chosen=None;rows={}
    with st.expander("이미지 · 위치 변경",expanded=False,key="wafer_detail_controls"):
        with_images=[p for p in POSITION_ORDER if die["positions"][p]["image_ids"]]
        position=st.segmented_control("촬영 위치",POSITION_ORDER,
                                      default="c" if "c" in with_images or not with_images else with_images[0],
                                      required=True,key="wafer_position_"+scope)
        site=die["positions"][position]
        ids=site["image_ids"]
        overlay=st.segmented_control("Overlay 표시",["원본만","Metrology","Defect","Metrology + Defect"],
            default="원본만",key="wafer_overlay_"+scope,selection_mode="single",
            help="이 화면은 원본만 표시합니다 · 분석 Overlay는 Inspection & Analysis에서 확인")
        if ids:
            rows={r["image_id"]:r for r in context["images"] if r["image_id"] in ids}
            batches={r["batch_id"]:r.get("display_name",r["batch_id"]) for r in context["batches"]}
            def label(key):
                row=rows[key]
                filename=row.get("original_filename",row.get("asset",""))
                return filename+" · "+key+" · "+batches.get(row["batch_id"],row["batch_id"])
            chosen=st.selectbox("등록 이미지 선택" if context["source"]=="SQLite" else "Mock 이미지 선택",
                               ids,format_func=label,key="wafer_image_"+scope+"_"+position)
            st.session_state["wafer_browser_image_id"]=chosen
        st.caption("Physical Site ID: "+(site["physical_site_id"] or "미등록"))
        if chosen and context["provider"]:
            record=context["provider"].image_record(chosen,batch_id=rows[chosen]["batch_id"])
            if record:
                def stored(key):
                    value=record.get(key)
                    return "미기록" if value is None or value=="" else str(value)
                st.caption("Batch: "+record["batch_display_name"]+" · 노광 "+
                           exposure_display(record.get("exposure_value"),record.get("exposure_unit"),missing="미기록")+
                           " · 반복 "+stored("repeat_index")+" · 조명 "+stored("illumination_setting"))
                st.caption("배율 "+stored("magnification")+" · Auto Exposure "+stored("auto_exposure")+
                           " · 촬영 "+stored("captured_at"))
    with head:
        counts=" · ".join(p+" "+str(len(die["positions"][p]["image_ids"]))+"장" for p in POSITION_ORDER)
        st.subheader(f"Die {die['die_code']} · {position}",anchor=False)
        st.caption(f"Wafer {context['wafer_id']} · 이미지 {counts}")
    with image_slot:
        if overlay and overlay!="원본만":
            st.info("선택한 Overlay 결과가 없습니다. 원본만 표시합니다.")
        if not ids:
            st.info("등록 이미지 없음" if context["source"]=="SQLite" else "Mock 이미지 없음")
        elif context["provider"]:
            if not record or (record["wafer_id"],record["die_code"],record["position_code"])!=(context["wafer_id"],die["die_code"],position):
                st.error("선택 위치와 Image Metadata가 일치하지 않아 표시를 중단했습니다.")
            else:
                render_original_preview(context["provider"],record)
                st.code(chosen,language=None,wrap_lines=True)
                st.text(record["original_filename"])
        else:
            image_panel(chosen,overlay=False)
            st.caption("Mock 공용 예시 · lt/c/rb별 실제 취득 파일이 아닙니다.")
    with status_slot:
        st.caption(die_hover(die)+" · AI 잠정 · 공식 채택 아님")

def render_ia_selected_image(context,die):
    """I&A (TASK18-B-14B): original and stored overlay side by side, same identities as the Map."""
    from dashboard.components.sqlite_input import render_original_preview
    from dashboard.components.tables import open_image_detail
    from dashboard.components.ia_analysis_result import select_run,render_stage_overlay
    record=None;view=None
    st.session_state["ia_sqlite_selected_image_id"]=None
    st.session_state["ia_analysis_view"]=None
    st.session_state["ia_sqlite_selected_location"]=None
    with st.container(key="ia_sqlite_selection"):
        head,identity_slot=st.columns([1,2],vertical_alignment="bottom")
        head.subheader("선택 이미지",anchor=False)
        identity=identity_slot.empty()
        controls=st.container(horizontal=True,vertical_alignment="center",gap="small",key="ia_image_controls")
        chooser=st.container()
        # TASK18-B-15: 선폭 Overlay | 결함 Overlay on top, the original below; all three = the same image ID.
        pair=st.container(key="ia_image_pair")
        with pair:
            left,right=st.columns(2,gap="small")
            left.markdown('<span class="ia-image-label">선폭 Overlay</span>',unsafe_allow_html=True)
            right.markdown('<span class="ia-image-label">결함 Overlay</span>',unsafe_allow_html=True)
        original_row=st.container(key="ia_original_row")
        with original_row:
            original,original_info=st.columns(2,gap="small")
            original.markdown('<span class="ia-image-label">원본</span>',unsafe_allow_html=True)
        position=None
        with controls:
            if context is not None and die is not None:
                wafer=context["wafer_id"];code=die["die_code"]
                # A new Map click gets fresh lt/image widgets, including a return to an old Die.
                scope=f"{wafer}_{context['batch_id'] or 'all'}_{code}_{context.get('selection_revision',0)}"
                with_images=[p for p in POSITION_ORDER if die["positions"][p]["image_ids"]]
                position=st.segmented_control("촬영 위치",POSITION_ORDER,
                    default="lt" if "lt" in with_images or not with_images else with_images[0],required=True,
                    key="ia_sqlite_position_"+scope,label_visibility="collapsed")
            show_metrology=st.toggle("선폭 Overlay",value=True,key="ia_metrology_overlay",
                             help="저장된 분석 결과의 Metrology Overlay (재추론 없음)")
            show_defect=st.toggle("결함 Overlay",value=True,key="ia_defect_overlay",
                          help="저장된 분석 결과의 Defect Overlay (재추론 없음)")
        if context is None:
            identity.caption("등록된 Wafer 없음")
            original.info("Wafer를 선택하면 결과가 표시됩니다.");left.info("Overlay 없음");right.info("Overlay 없음")
        elif die is None:
            identity.caption(context["wafer_id"]+" / Die 미선택")
            original.info("해당 조건의 데이터 없음" if not context["images"] else "Wafer Map에서 Die를 선택하세요.")
            left.info("Overlay 없음");right.info("Overlay 없음")
        else:
            identity.caption(wafer+" / Die "+code+" / "+position)
            site=die["positions"][position]
            st.session_state["ia_sqlite_selected_location"]={
                "wafer_id":wafer,"die_code":code,"position_code":position,
                "physical_site_id":site["physical_site_id"]}
            rows={r["image_id"]:r for r in context["images"]
                  if r["image_id"] in site["image_ids"]
                  and (r["wafer_id"],r["die_code"],r["position_code"])==(wafer,code,position)
                  and (context["batch_id"] is None or r["batch_id"]==context["batch_id"])}
            if not rows:
                original.info("등록 이미지 없음");left.info("Overlay 없음");right.info("Overlay 없음")
            else:
                batches={r["batch_id"]:r.get("display_name",r["batch_id"]) for r in context["batches"]}
                def label(image_id):
                    row=rows[image_id]
                    parts=[row["original_filename"],image_id,batches.get(row["batch_id"],row["batch_id"])]
                    for field,name in (("repeat_index","반복"),("exposure_value","노광"),("illumination_setting","조명")):
                        value=row.get(field)
                        if value is not None and value!="" and value==value:
                            parts.append(name+" "+str(value))
                    return " · ".join(parts)
                if len(rows)>1:
                    chosen=chooser.selectbox("등록 이미지 선택",list(rows),format_func=label,
                        key="ia_sqlite_image_"+scope+"_"+position,label_visibility="collapsed",
                        help="동일 위치의 실제 원본 · 파일명 / Image ID / Batch / 기록된 촬영 조건")
                else:
                    chosen=next(iter(rows))
                candidate=context["provider"].image_record(chosen,batch_id=rows[chosen]["batch_id"])
                if candidate is None or (candidate["wafer_id"],candidate["die_code"],candidate["position_code"])!=(wafer,code,position):
                    original.error("선택 위치와 Image Metadata가 일치하지 않아 표시를 중단했습니다.")
                else:
                    record=candidate
                    st.session_state["ia_sqlite_selected_image_id"]=chosen
                    view=select_run(context["provider"],record)  # Read-only stored result.
                    st.session_state["ia_analysis_view"]=view
                    with left:
                        render_stage_overlay(view,"metrology",show_metrology)
                    with right:
                        render_stage_overlay(view,"defect",show_defect)
                    with original,st.container(key="ia_sqlite_original"):
                        render_original_preview(context["provider"],record)
                    original_info.caption(position+" · "+record["original_filename"]+" · "+record["image_id"],
                                          help="Image ID: "+record["image_id"])
        with original_info:
            if st.button("이미지 상세 열기",key="ia_sqlite_detail",type="tertiary",disabled=record is None):
                open_image_detail(record["image_id"],record["batch_id"])
