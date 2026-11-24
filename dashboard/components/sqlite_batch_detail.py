"""SQLite Batch presentation. Colors describe stored state, never inferred analysis."""
from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.units import exposure_display,stored_unit_text
from dashboard.components.charts import theme
from dashboard.components.tables import image_button


def display_value(value):
    return "미기록" if value is None or pd.isna(value) or value=="" else str(value)


def _value(label,value):
    st.markdown('<div class="sqlite-info-label">'+escape(label)+'</div>'
                '<div class="sqlite-info-value">'+escape(display_value(value))+'</div>',unsafe_allow_html=True)


def batch_metadata(record):
    with st.container(key="sqlite_batch_summary"):
        for column,(label,value) in zip(st.columns([2,1,1,1]),[
                ("Batch 이름",record["batch_name"] or record["automatic_display_name"]),
                ("이미지 수",str(record["image_count"])+"장"),
                ("Upload Mode",record["upload_mode"]),("Dataset Role",record["dataset_role"])]):
            with column,st.container(border=True):
                _value(label,value)
    left,right=st.columns(2)
    with left,st.container(border=True,key="sqlite_batch_identity"):
        st.caption("실제 Batch ID · 복사 가능")
        st.code(record["batch_id"],language=None,wrap_lines=True)
        _value("자동 Batch 표시 번호",record["automatic_display_name"])
        st.caption("표시 번호와 사용자 이름은 조회 키가 아닙니다.")
    with right,st.container(border=True,key="sqlite_batch_metadata"):
        _value("사용자 지정 이름",record["batch_name"])
        _value("Wafer ID",record.get("wafer_list") or record["wafer_id"])  # v6: every Wafer of the Batch.
        _value("등록 일시",record["created_at"])


def wafer_metadata(record):
    if int(record.get("wafer_count",1) or 1)>1:
        with st.expander("연결된 Wafer / 공정 Metadata"):
            st.text("다중 Wafer Batch: "+str(record.get("wafer_list")))
            st.text("Wafer별 노광·PEB (s)는 위 'Wafer 구성' 표를 확인하세요. Batch 단일 공정 값은 없습니다.")
        return
    with st.expander("연결된 Wafer / 공정 Metadata"):
        for label,key in (("Wafer Label","wafer_label"),("Lot ID","lot_id"),
                          ("Process Run ID","process_run_id"),("노광","exposure_value"),
                          ("노광 단위","exposure_unit"),("PEB (s)","peb_temperature_c"),("Recipe","recipe_version"),
                          ("공정 일시","processed_at")):
            if key=="exposure_value":   # TASK18-B-13A.2: value + display unit; the unit row shows the stored state.
                st.text(label+": "+exposure_display(record.get(key),record.get("exposure_unit"),missing=display_value(None)))
            elif key=="exposure_unit":
                st.text(label+": "+stored_unit_text(record.get(key)))
            else:
                st.text(label+": "+display_value(record.get(key)))
        sources={"batch":"Batch 공정 기록","wafer":"Wafer 공정 기록에서 상속"}
        st.text("노광 출처: "+sources.get(record.get("exposure_source"),"미기록"))


def registration_donut(image_count):
    figure=go.Figure(go.Pie(labels=["DB 등록 이미지"],values=[image_count],hole=.72,
                           marker=dict(colors=["#45aa93"]),textinfo="none",
                           hovertemplate="%{label}: %{value}장<extra></extra>"))
    figure.add_annotation(text=str(image_count)+"장",showarrow=False,font=dict(size=24))
    figure.update_layout(showlegend=False)
    return theme(figure,210)


def metadata_stage(status):
    # Only explicit stored flags get confirmed/review/error colors.
    if status=="manual_confirmed":
        return "complete","✓","수동 확인","metadata_status=manual_confirmed · 수동 보완 확인 기록"
    if status in ("needs_review","needs_metadata"):
        return "review","!","검토 필요","metadata_status="+status
    if status in ("error","invalid","failed"):
        return "error","✕","오류","metadata_status="+status
    if status=="filename_parsed":
        return "neutral","—","파싱·저장","자동 파싱·저장됨 · 별도 수동 검토/QA 완료 기록은 아님"
    return "neutral","—","미기록" if not status else "저장 상태","metadata_status="+str(status)


def _stage(kind,icon,label,note):
    st.markdown('<span class="sqlite-stage sqlite-stage-'+kind+'" title="'+escape(note,quote=True)+
                '"><span aria-hidden="true">'+escape(icon)+'</span> '+escape(label)+'</span>',unsafe_allow_html=True)


def batch_processing(provider,record):
    matrix=provider.batch_processing_matrix(record["batch_id"])
    left,right=st.columns([1.6,1])
    with left,st.container(border=True,height=440,key="sqlite_batch_matrix"):
        st.subheader("Processing Matrix")
        st.caption("이 Batch의 이미지별 저장·처리 상태 · 물리적 Wafer Map 아님")
        with st.container(height=290,border=False,key="sqlite_batch_matrix_rows"):
            widths=[1.8,1.2,.8,1,1.2,1]
            for col,label in zip(st.columns(widths,gap="small"),[
                    "Image","Metadata","QA","Calibration","CD/Metrology","Defect AI"]):
                col.markdown("**"+label+"**")
            if matrix.empty:
                st.info("등록된 이미지가 없습니다.")
            for row in matrix.to_dict("records"):
                columns=st.columns(widths,gap="small",vertical_alignment="center")
                with columns[0]:
                    image_button(row["image_id"],record["batch_id"],
                                 key="sqlite_matrix_image_"+row["image_id"],filename=row["original_filename"])
                with columns[1]:
                    _stage(*metadata_stage(row["metadata_status"]))
                for column,label in zip(columns[2:],["QA","Calibration","Metrology","Defect AI"]):
                    with column:
                        _stage("neutral","—","I&A",label+" 결과는 Inspection & Analysis에서 조회")
        st.caption("✓ 실제 확인 기록 · ! 기록된 검토 필요 · ✕ 기록된 오류 · — 이 화면 미표시(I&A에서 조회)/미확인. 자동 파싱은 QA 완료가 아닙니다.")
    with right,st.container(border=True,height=440,key="sqlite_batch_processing"):
        st.subheader("등록 / 분석 상태")
        mode=st.segmented_control("표시 기준",["Whole","Metrology","Defect"],default="Whole",
                                  required=True,width="stretch",key="sqlite_batch_donut_"+record["batch_id"])
        with st.container(height=300,border=False,key="sqlite_batch_donut_area"):
            count=len(matrix)
            if mode=="Whole":
                st.markdown("**등록 이미지 "+str(count)+"장**")
                if count:
                    st.plotly_chart(registration_donut(count),theme="streamlit",width="stretch",
                                    key="sqlite_batch_registration_"+record["batch_id"])
                else:
                    st.info("등록된 이미지가 없습니다.")
                st.caption("저장된 이미지 수이며 전체 업로드 시도·실패 건수가 없어 성공률을 계산하지 않습니다.")
            elif mode=="Metrology":
                st.info("Metrology 결과는 Inspection & Analysis에서 조회합니다")
                st.caption(f"등록 이미지 {count}장 · 이 화면은 등록 정보만 표시")
            else:
                st.info("Defect AI 결과는 Inspection & Analysis에서 조회합니다")
                st.caption(f"등록 이미지 {count}장 · 이 화면은 등록 정보만 표시")


def wafer_composition(provider,record):
    """v6 multi-Wafer Batch: per-Wafer 노광/PEB, images, latest analysis and analysis-input readiness (read-only)."""
    batch_id=record["batch_id"]
    members=provider.batch_wafers()
    members=members[members["batch_id"]==batch_id]
    analysed={}
    for item in provider.latest_analysis().values():
        if item["batch_id"]==batch_id:
            analysed[item.get("wafer_id")]=analysed.get(item.get("wafer_id"),0)+1
    readiness={}
    note=""
    try:
        from dashboard.components.analysis_management import target_preview,prepare_selected_inputs
        preview=target_preview(provider,"Batch",batch_id)
        plan=prepare_selected_inputs(provider,preview)
        wafer_of=dict(zip(preview["images"]["image_id"],preview["images"]["wafer_id"]))
        for item in plan["items"]:
            counts=readiness.setdefault(wafer_of.get(item["image_id"]),{})
            counts[item["validation_status"]]=counts.get(item["validation_status"],0)+1
    except Exception as exc:  # Readiness is informational; registration data is still shown.
        note="분석 입력 상태를 조회하지 못했습니다: "+type(exc).__name__
    rows=[]
    for item in members.to_dict("records"):
        state=readiness.get(item["wafer_id"],{})
        rows.append({"Wafer ID":item["wafer_id"],
                     "노광":exposure_display(item["exposure_value"],item.get("exposure_unit")),
                     "PEB (s)":"—" if item["peb_temperature_c"] is None or item["peb_temperature_c"]!=item["peb_temperature_c"] else f"{item['peb_temperature_c']:g}",
                     "이미지 수":int(item["image_count"]),"분석 결과 이미지 수":int(analysed.get(item["wafer_id"],0)),
                     "READY":state.get("READY",0),"REVIEW_REQUIRED":state.get("REVIEW_REQUIRED",0),"BLOCKED":state.get("BLOCKED",0)})
    total={"Wafer ID":"Batch 합계","노광":"Wafer별","PEB (s)":"Wafer별",
           **{key:sum(r[key] for r in rows) for key in ("이미지 수","분석 결과 이미지 수","READY","REVIEW_REQUIRED","BLOCKED")}}
    with st.container(border=True,key="sqlite_batch_wafers"):
        st.subheader("Wafer 구성")
        st.caption(f"이 Batch에 포함된 Wafer {len(rows)}개 · 이미지는 각 이미지의 실제 Wafer 기준 · "
                   "분석 결과는 이미지별 최신 저장 결과 · READY/REVIEW_REQUIRED/BLOCKED는 분석 입력 준비 상태")
        st.dataframe(pd.DataFrame(rows+[total]),hide_index=True,width="stretch",key="sqlite_batch_wafer_table")
        if total["이미지 수"]!=int(record["image_count"]):
            st.warning(f"Wafer별 합계 {total['이미지 수']}장과 Batch 전체 {int(record['image_count'])}장이 다릅니다. 데이터를 확인하세요.")
        if note:
            st.caption(note)
