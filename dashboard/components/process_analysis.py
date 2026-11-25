"""Process-condition UI backed by source metadata and, for SQLite, stored AI analysis results.

Stored results are the latest run per image (unadopted, AI provisional). They are mapped into
dashboard.process_analysis's existing result contract; nothing is recomputed or judged here.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard import data_provider as mock
from dashboard.components.analysis_management import collect_images
from dashboard.components.tables import data_table
from dashboard.process_analysis import defect_ratio_table, ALL,MISSING,metadata_frame,condition_table


def stored_results(provider):
    """Latest stored run per image -> existing image_mean_um contract (read-only)."""
    results={}
    for image_id,item in provider.latest_analysis().items():
        calibration=item.get("calibration") or {}
        pd_state=item.get("pd_policy_state")
        engine_ok=item.get("defect_status")=="SUCCESS" and item.get("engine_status") in ("SUCCESS","SKIPPED")
        measured=item.get("metrology_status")=="SUCCESS"
        results[image_id]={
            "status":"SUCCESS" if engine_ok and (measured or pd_state in ("MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED")) else "FAILED",
            "pd_detected":True if pd_state in ("MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED") else False if pd_state=="NO_PD" else None,
            "partial_measurement":bool(item.get("partial_measurement")),
            "calibration":{"calibration_id":calibration.get("calibration_id"),
                           "valid":isinstance(calibration.get("um_per_px"),(int,float)) and calibration.get("um_per_px",0)>0,
                           "magnification":calibration.get("magnification")},
            "calibration_applied":measured,"unit":"um",
            "valid_line_widths_um":item.get("line_widths_um") or [],
            "valid_gap_widths_um":item.get("gap_widths_um") or [],
            "defect_classes":list(item.get("defect_classes") or []),
            "outcome":item.get("outcome")}
    return results


def source_metadata(source, provider=None):
    if source=="SQLite":
        return metadata_frame(collect_images(provider))
    images=mock.images().copy()
    batches=mock.batches().set_index("batch_id")
    images["exposure_value"]=images["batch_id"].map(batches["exposure"])
    images["exposure_unit"]=None
    images["peb_temperature_c"]=None
    return metadata_frame(images)


def _select(label,options,key,format_func=None):
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key,None)
    return st.selectbox(label,options,key=key,format_func=format_func or str,
                        index=0 if options else None,disabled=not options)


def _filter_scope(frame):
    scope=st.radio("공정 분석 범위",["전체","특정 Wafer","특정 Batch"],horizontal=True,key="ia_process_scope")
    if scope=="특정 Wafer":
        selected=_select("분석 Wafer",sorted(frame["wafer_id"].unique()),"ia_process_wafer")
        return frame[frame["wafer_id"]==selected]
    if scope=="특정 Batch":
        selected=_select("분석 Batch",list(frame["batch_id"].unique()),"ia_process_batch")
        return frame[frame["batch_id"]==selected]
    return frame


def wafer_assessments(frame,provider,analysis):
    """Same Die assessment as the Wafer Map (policy v0.1), over the images in this scope."""
    from dashboard.components.inspection_wafer import die_snapshot
    from dashboard.die_assessment import summarize_wafer
    result={}
    from dashboard.components.inspection_wafer import duplicate_exclusions
    for wafer in sorted(frame["wafer_id"].unique()):
        records=frame[frame["wafer_id"]==wafer].to_dict("records")
        # Same rules as the Wafer Map: never merge different exposure/PEB conditions into one Die.
        conditions={(r.get("exposure_condition"),r.get("peb_condition")) for r in records}
        if len(conditions)>1:
            result[wafer]=None
            continue
        dies=die_snapshot(wafer,provider.wafer_sites(wafer),records,analysis,
                          exclude_from_judgment=duplicate_exclusions(records,analysis))
        result[wafer]=summarize_wafer(dies)
    return result


def fail_area(frame,provider=None,analysis=None):
    with st.expander("Wafer별 Die/Row Fail 경향",expanded=False):
        wafers=sorted(frame["wafer_id"].unique())
        if wafers and provider is not None and analysis is not None:
            from dashboard.die_assessment import POLICY_VERSION,rate_text
            st.caption(POLICY_VERSION+" · AI 잠정 Fail Rate · 공식 수율/실제 공정 불량률 아님 · 공식 채택 결과 없음")
            summaries=wafer_assessments(frame,provider,analysis)
            blocked="조건 선택 필요 (노광/PEB 혼합)"
            data_table(pd.DataFrame([{"Wafer ID":wafer,"Die PASS":s["PASS"] if s else blocked,"Die FAIL":s["FAIL"] if s else "-",
                "Die PENDING":s["PENDING"] if s else "-","평가 완료 / 전체":f"{s['evaluated']} / {s['dies']}" if s else "-",
                "Die Fail Rate (AI 잠정)":rate_text(s["die_fail_rate"]) if s else "-",
                "Row PASS/FAIL/PENDING":f"{s['row_PASS']}/{s['row_FAIL']}/{s['row_PENDING']}" if s else "-",
                "Row Fail Rate (AI 잠정)":rate_text(s["row_fail_rate"]) if s else "-"} for wafer,s in summaries.items()]))
            data_table(pd.DataFrame([{"Wafer ID":wafer,"Row":r["row"],"PASS":r["PASS"],"FAIL":r["FAIL"],"PENDING":r["PENDING"],
                "Row 판정":r["verdict"],"Row Die Fail Rate":rate_text(r["die_fail_rate"])}
                for wafer,s in summaries.items() if s for r in s["rows"]]),height=min(300,(3*len(summaries)+1)*32+3))
            st.caption("이 표는 공정 분석 범위 기준입니다. 같은 조건의 여러 Batch는 함께 집계되며 동일 SHA 중복은 1장으로 판정합니다."
                       " Batch별 판정은 Wafer Map의 Batch 필터로 확인하세요.")
        elif wafers:
            st.caption("저장된 결함 결과 없음 · 공식 반복 이미지 선정 기준 미확정")
            data_table(pd.DataFrame([{"Wafer ID":wafer,"Die Fail Rate":"미평가","Die 평가 확정 / 전체":"미평가 / 24",
                "Row Fail Rate":"미평가","Row 평가 확정 / 전체":"미평가 / 3"} for wafer in wafers]))
        else:
            st.info("대상 Wafer 없음")
        st.caption("B 방식: lt/c/rb 필수 분석 완료 후 O/S/PD 검출 Die는 FAIL, 미검출은 PASS. 미완료·실패는 PENDING.")
        st.caption("Row의 8개 Die 평가 완료 후 PASS Die ≥5이면 Row PASS. 미완료는 PENDING이며 확정 전 FAIL 예상과 구분합니다.")
        st.caption("Fail Rate = 확정 FAIL / (확정 PASS + 확정 FAIL) ×100. PENDING은 분모에서 제외합니다.")
        st.info("Wafer 최종 Fail 판정 기준 미확정 — 기준 설정 후 적용 예정")


GRAPH_HEIGHT=300  # TASK18-B-13A: one compact common height for the three cards.


def _x_title(subset,axis):
    """In-graph X title (TASK18-B-13A.2): 노광 in its display unit, PEB in s. Mixed recorded units are not merged."""
    if axis!="exposure":
        return "PEB 조건 (s)"
    from dashboard.units import exposure_unit_text
    units={exposure_unit_text(u) for u in subset.get("exposure_unit",pd.Series(dtype=object))} or {"s"}
    return "노광 조건 ("+units.pop()+")" if len(units)==1 else "노광 조건 (조건별 단위)"


def _compact(fig,*,legend=False):
    """Layout only (size/margins/legend); traces, values and axes titles are unchanged."""
    fig.update_layout(height=GRAPH_HEIGHT,margin=dict(l=8,r=8,t=28,b=8),
                      legend=dict(orientation="h",y=1.02,yanchor="bottom",x=0,font=dict(size=11)) if legend else None,
                      font=dict(size=12))
    return fig


def _card_head(title,subtitle,help_text=None):
    from html import escape
    tip=('<span class="ia-graph-help" tabindex="0" title="'+escape(help_text)+'" aria-label="'+escape(help_text)+'">?</span>'
         if help_text else "")
    # Spans inside Streamlit's own paragraph: its -1rem container margin is balanced (div-only blocks collapse).
    st.markdown('<span class="ia-graph-title">'+escape(title)+tip+'</span><span class="ia-graph-sub">'+escape(subtitle)+'</span>',
                unsafe_allow_html=True)


def _condition_tab(frame,axis,source,results=None,provider=None,analysis=None):
    # Compact control row: count summary + the axis' secondary filter (both kept from the former layout).
    with st.container(horizontal=True,vertical_alignment="bottom",gap="small",key="ia_controls_"+axis):
        if axis=="exposure":
            values=sorted(frame["peb_temperature_c"].dropna().unique().tolist())
            with st.container(width=210):
                choice=_select("PEB 보조 필터",[ALL,MISSING,*values],"ia_process_peb_filter",
                    lambda v:"전체 PEB" if v==ALL else "PEB 미입력" if v==MISSING else f"{v:g} s")
            subset=frame if choice==ALL else frame[frame["peb_temperature_c"].isna()] if choice==MISSING else frame[frame["peb_temperature_c"]==choice]
            mixed="전체 조건 집계에는 서로 다른 PEB 조건이 포함될 수 있습니다." if choice==ALL else None
        else:
            values=list(frame["exposure_condition"].unique())
            with st.container(width=210):
                choice=_select("노광 보조 필터",[ALL,*values],"ia_process_exposure_filter",
                    lambda v:"전체 노광" if v==ALL else v)
            subset=frame if choice==ALL else frame[frame["exposure_condition"]==choice]
            mixed="전체 조건 집계에는 서로 다른 노광 조건이 포함될 수 있습니다." if choice==ALL else None
        error=st.radio("오차율 표시",["Absolute Error","Signed Error"],horizontal=True,key="ia_process_error_"+axis,
                       width="content",help="설계 대비 오차율 그래프의 표시 방식")
        st.caption(f"{source} · 필터 적용 이미지 {len(subset)}장 · 고유 Wafer {subset['wafer_id'].nunique()}개"
                   +(" · ⚠ "+("PEB" if axis=="exposure" else "노광")+" 혼합 포함" if mixed else ""),
                   help=mixed,width="stretch")
    summary=condition_table(subset,axis,results=results)
    has_values=not summary.empty and summary["평균 선폭 (µm)"].notna().any()
    valid=_sorted_conditions(summary.dropna(subset=["평균 선폭 (µm)"]),subset,axis) if has_values else None
    counts=(" · ".join(f"{r['공정 조건']} {int(r['유효 이미지 수'])}장" for r in valid.to_dict("records")) if has_values else "")
    no_result="저장된 계측 결과 없음 · 등록 Calibration 확인 후 분석 실행이 필요합니다"
    x_title=_x_title(subset,axis)
    axis_label=x_title
    unit_note=(" · 노광 조건은 시간(초, s) 단위로 표시 (DB 저장값 변경 없음)" if axis=="exposure" else "")
    with st.container(key="ia_graphs_"+axis):
        first,second,third=st.columns(3,gap="small")
        with first,st.container(border=True,key=f"ia_graph_{axis}_width"):
            _card_head("조건별 평균 Line Width",f"Y: 평균 Line Width (µm) ±1σ · X: {axis_label}",
                       "AI 잠정 분석 결과(공식 채택 아님) · 이미지별 최신 실행 · 부분 계측·PD 이미지는 평균에서 제외 · "
                       "SPC 관리선 없음 · 오차 막대 = 이미지별 평균 선폭 간 표본 표준편차(ddof=1) ±1σ · 표본 1개면 산출 불가"
                       +(" · 조건별 유효 이미지 수: "+counts if counts else "")+unit_note)
            if has_values:
                st.plotly_chart(_compact(width_line_chart(valid)).update_xaxes(title_text=x_title),width="stretch",theme="streamlit",key="ia_process_width_"+axis)
            else:
                st.info(no_result)
        with second,st.container(border=True,key=f"ia_graph_{axis}_error"):
            metric=error+" (%)"
            _card_head("설계 대비 오차율",f"Y: {metric} · 설계 10 µm 기준 · X: {axis_label}",
                       "설계 선폭 10 µm · Calibration 적용 µm 결과에만 비교 · 기록표와 엔진 자동 연결 없음")
            if has_values:
                st.plotly_chart(_compact(error_bar_chart(valid,metric)).update_xaxes(title_text=x_title),width="stretch",theme="streamlit",key="ia_process_error_chart_"+axis)
            else:
                st.info("분석 결과 없음 — 선폭·표준편차·오차율 산출 불가")
        with third,st.container(border=True,key=f"ia_graph_{axis}_defects"):
            _card_head("조건별 결함 개수 비율",f"Y: 결함 개수 비율 (%) · X: {axis_label}",
                       "비율 = 해당 조건의 검출 결함 건수 ÷ 현재 필터 전체 조건의 검출 결함 건수 × 100 · "
                       "검출 결함 = 이미지별 최신 실행의 모델 검출(규칙 기반 PD 의심 제외) · S에서 파생된 P는 원래 S와 1건으로 집계 · "
                       "동일 촬영(SHA) 중복은 1장으로 집계 · "
                       "AI 잠정 결과(공식 판정 아님)")
            ratio=defect_ratio_table(subset,axis,analysis) if analysis is not None else pd.DataFrame()
            analysed=int(ratio["결함 분석 완료 이미지 수"].sum()) if not ratio.empty else 0
            if ratio.empty or analysed==0:
                st.info("결함 분석 결과 없음 · 현재 조회 범위에 결함 분석이 완료된 이미지가 없습니다")
            elif ratio.attrs.get("total_defects",0)==0:
                st.info(f"조회 범위 전체 결함 0건 · 결함 분석 완료 이미지 {analysed}장 (비율 산출 불가)")
            else:
                ordered=_sorted_conditions(ratio,subset,axis)
                st.plotly_chart(_compact(defect_ratio_chart(ordered)).update_xaxes(title_text=x_title),
                                width="stretch",theme="streamlit",key="ia_process_defects_"+axis)
                waiting=ordered[ordered["결함 분석 완료 이미지 수"]==0]["공정 조건"].tolist()
                if waiting:
                    st.caption("결함 분석 결과 없음: "+", ".join(map(str,waiting))+" (0%와 구분해 막대 없이 표시)")
    fail_area(subset,provider,analysis)  # TASK18-B-13A.1: first of the four detail expanders.
    if subset.empty:
        st.info("선택 조건의 등록 이미지 없음")
    else:
        with st.expander("조건별 요약",expanded=False):
            data_table(summary.style.format(na_rep="미분석",precision=3),height=min(300,(len(summary)+1)*32+3))
        with st.expander("조건별·Wafer별 상세 결과",expanded=False):
            detail=condition_table(subset,axis,by_wafer=True)
            data_table(detail.style.format(na_rep="미분석",precision=3),height=min(330,(len(detail)+1)*32+3))
        with st.expander("대상 이미지 Metadata",expanded=False):
            columns=[c for c in ["image_id","batch_id","wafer_id","exposure_value","exposure_unit","peb_temperature_c"] if c in subset]
            data_table(subset[columns].style.format(na_rep="미기록",precision=3),height=min(330,(len(subset)+1)*32+3))


def _sorted_conditions(valid,subset,axis):
    """Order conditions by their stored numeric value (labels carry the unit text)."""
    column,value="exposure_condition","exposure_value"
    if axis!="exposure":
        column,value="peb_condition","peb_temperature_c"
    numeric=subset.groupby(column)[value].first().to_dict() if value in subset else {}
    import math
    def key(label):
        value=numeric.get(label)
        real=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        return (0,float(value)) if real else (1,0.0)  # Unrecorded values (e.g. "PEB 미입력") sort last.
    return valid.assign(_order=valid["공정 조건"].map(key)).sort_values("_order").drop(columns="_order")


def width_line_chart(valid):
    """Mean line width per condition: points + line between actual conditions only."""
    import plotly.graph_objects as go
    sd=valid["이미지 간 표준편차 (µm)"]
    # TASK18-B-18: ±1 SD of the same observations as the mean (image-mean line widths, sample SD, ddof=1).
    sd_values=[float(v) if v==v and v is not None else 0.0 for v in sd]
    sd_text=[f"{v:.3f} µm" if v==v and v is not None else "산출 불가" for v in sd]
    fig=go.Figure(go.Scatter(x=valid["공정 조건"],y=valid["평균 선폭 (µm)"],mode="lines+markers+text",
        text=[f"{v:.3f}" for v in valid["평균 선폭 (µm)"]],textposition="top center",
        error_y=dict(type="data",array=sd_values,visible=True,thickness=1.4,width=5),
        customdata=list(zip(valid["유효 이미지 수"],sd_text)),
        hovertemplate="%{x}<br>평균 선폭 %{y:.3f} µm<br>표준편차 %{customdata[1]}"
                      "<br>유효 표본 수 n = %{customdata[0]}<extra></extra>",name="평균 Line Width"))
    fig.update_layout(xaxis_title="공정 조건 (저장된 값)",yaxis_title="평균 Line Width (µm)",
                      xaxis_type="category",showlegend=False,margin=dict(t=30))
    return fig


def error_bar_chart(valid,metric):
    """Design error per condition (bar, unchanged form); hover shows the sample size."""
    fig=px.bar(valid,x="공정 조건",y=metric,hover_data={"유효 이미지 수":True},text=valid[metric].map(lambda v:f"{v:.2f}%"))
    fig.update_layout(yaxis_title=metric,xaxis_type="category")
    return fig


def defect_ratio_chart(table):
    """Defect share per condition; conditions without defect results get no bar (not 0 %)."""
    import plotly.graph_objects as go
    done=table["결함 분석 완료 이미지 수"]>0
    y=[float(v) if ok else None for v,ok in zip(table["결함 개수 비율 (%)"],done)]
    text=[f"{v:.1f}%" if ok else "결과 없음" for v,ok in zip(table["결함 개수 비율 (%)"].fillna(0),done)]
    per100=[f"{v:.1f}" if ok else "-" for v,ok in zip(table["이미지 100장당 결함 건수"].fillna(0),done)]
    fig=go.Figure(go.Bar(x=table["공정 조건"],y=y,text=text,textposition="outside",marker_color="#e0883e",
        customdata=list(zip(table["결함 건수"],table["결함 분석 완료 이미지 수"],per100,text,
                            table["결함 분석 미실행·결과 없음 이미지 수"],
                            table["S 파생 P (S와 1건 합산)"] if "S 파생 P (S와 1건 합산)" in table else [0]*len(table))),
        hovertemplate="%{x}<br>결함 건수 %{customdata[0]}건<br>결함 개수 비율 %{customdata[3]}"
                      "<br>결함 분석 완료 이미지 %{customdata[1]}장<br>이미지 100장당 결함 %{customdata[2]}건"
                      "<br>결과 없음 이미지 %{customdata[4]}장"
                      "<br>S 파생 P %{customdata[5]}건 (원래 S와 1건으로 합산)<extra></extra>"))
    fig.update_layout(yaxis_title="결함 개수 비율 (%)",xaxis_type="category",showlegend=False,margin=dict(t=30))
    fig.update_yaxes(rangemode="tozero")
    return fig


def _distribution(subset,axis,results):
    """Per-image Line/Gap means of the same eligible (READY) images used in the table."""
    from dashboard.process_analysis import image_mean_um
    from statistics import mean
    column="exposure_condition" if axis=="exposure" else "peb_condition"
    rows=[]
    for record in subset.to_dict("records"):
        result=(results or {}).get(record["image_id"])
        if image_mean_um(result,magnification=record.get("magnification"))["status"]!="READY":
            continue
        rows.append({"공정 조건":record[column],"구분":"Line","이미지 평균 (µm)":mean(result["valid_line_widths_um"])})
        if result.get("valid_gap_widths_um"):
            rows.append({"공정 조건":record[column],"구분":"Gap","이미지 평균 (µm)":mean(result["valid_gap_widths_um"])})
    return pd.DataFrame(rows)


def _scope_text(scope):
    if scope=="특정 Wafer":
        return "Wafer "+str(st.session_state.get("ia_process_wafer") or "-")
    if scope=="특정 Batch":
        return "Batch "+str(st.session_state.get("ia_process_batch") or "-")[:22]
    return "전체 Wafer·Batch"


def render_process_analysis(source,provider=None,map_scope=None,*,title=None,on_simulation=None):
    """TASK18-B-13A: one compact head row (title · scope summary · '분석 조건' · Simulation), then 노광/PEB tabs
    with three side-by-side graph cards. Filters, keys and the data path are unchanged."""
    head=st.container(horizontal=True,vertical_alignment="center",gap="small",key="ia_trend_head")
    with head:
        if title:
            st.subheader(title,anchor=False,width="content")
        summary_slot=st.empty()
    if source=="SQLite" and provider is None:  # No Wafer context (empty/unavailable DB): never fall back to Mock.
        st.info("SQLite 공정 데이터를 조회할 수 없습니다 · 등록 Wafer 없음 또는 DB 조회 실패")
        return
    frame=source_metadata(source,provider)
    # Source changes must not leave filters from the other provider active.
    if st.session_state.get("_ia_process_source")!=source:
        for key in ("ia_process_wafer","ia_process_batch","ia_process_peb_filter","ia_process_exposure_filter"):
            st.session_state.pop(key,None)
        st.session_state["_ia_process_source"]=source
    with head:
        with st.popover("분석 조건",icon=":material/tune:",key="ia_process_conditions"):
            scoped=_filter_scope(frame)
            if map_scope:
                st.caption("Wafer Map 범위: "+map_scope+" · 공정 분석 범위는 이 선택을 따릅니다 (서로 다를 수 있음)")
        if on_simulation is not None and st.button("Simulation",key="ia_simulation"):
            on_simulation()
    scope=st.session_state.get("ia_process_scope","전체")
    parts=["공정 분석 범위: "+_scope_text(scope)]
    if map_scope:
        parts.append("Wafer Map 범위: "+map_scope+" (별도)")
    summary_slot.caption(" · ".join(parts)+(" · 예시 데이터" if source=="Mock" else " · AI 잠정 결과 · 공식 채택 아님"),
                         width="stretch")
    analysis=provider.latest_analysis() if source=="SQLite" and provider is not None else None
    results=stored_results(provider) if analysis is not None else None
    exposure,peb=st.tabs(["노광","PEB"])
    with exposure:_condition_tab(scoped,"exposure",source,results,provider,analysis)
    with peb:_condition_tab(scoped,"peb",source,results,provider,analysis)
