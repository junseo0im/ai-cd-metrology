"""Shared accumulation/status presentation; supplied SQLite values are registration metadata only."""
from html import escape
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.tables import data_table


def kpi_tiles(items, *, compact=False):
    """Small KPI tiles: (label, value[, tone[, tooltip]]). tone: ready / review / blocked / neutral."""
    tiles=[]
    for item in items:
        label,value=item[0],item[1]
        tone=item[2] if len(item)>2 else "neutral"
        tip=item[3] if len(item)>3 else ""
        tiles.append('<div class="dcc-kpi dcc-kpi-'+escape(tone or "neutral")+'"'+
                     (' title="'+escape(tip)+'"' if tip else "")+'><small>'+escape(str(label))+
                     '</small><b>'+escape(str(value))+'</b></div>')
    st.markdown('<div class="dcc-kpi-wrap"><div class="dcc-kpis'+(" dcc-kpis-compact" if compact else "")+'">'+"".join(tiles)+'</div></div>',
                unsafe_allow_html=True)


def chips(label, values, *, note=None, warn=None):
    """One labelled row of small value chips (sub-classification under a KPI row)."""
    parts=['<span class="dcc-chip-label">'+escape(label)+'</span>']
    parts+=['<span class="dcc-chip">'+escape(str(v))+'</span>' for v in values]
    if note:
        parts.append('<span class="dcc-chip-note">'+escape(note)+'</span>')
    if warn:
        parts.append('<span class="dcc-chip dcc-chip-warn">'+escape(warn)+'</span>')
    st.markdown('<div class="dcc-chips">'+"".join(parts)+'</div>',unsafe_allow_html=True)


def accumulation_overview(summary=None, *, source="Mock"):
    summary=data.accumulation_summary() if summary is None else summary
    history=summary["history"]
    if source=="Mock":
        st.markdown(f"**전체 Batch** {summary['batch_count']}　 **전체 Image** {summary['image_count']}　 "
                    f"**최근 추가** {summary['recent_image_count']} (예시)")
        st.caption(" / ".join(f"{mode} {count}" for mode,count in summary["mode_image_counts"].items())+" · 누적량은 예시 이력")
    else:
        kpi_tiles([("Batch",summary["batch_count"],"neutral","등록된 Batch 수"),
                   ("등록 이미지",summary["image_count"],"neutral","SQLite에 등록된 전체 이미지"),
                   ("분석 결과 이미지",summary["analysed_count"],"neutral","저장된 분석 결과가 1건 이상 있는 이미지 (이미지별 최신 실행)"),
                   ("검토 필요",summary["review_count"],"review","최신 분석 결과가 품질 검토·부분 계측·PD 정책 검토인 이미지")])
        modes=summary["mode_image_counts"];total=sum(modes.values())
        chips("입력 모드별 등록 이미지",[f"{mode} {count}" for mode,count in modes.items()],
              note=f"합계 {total} = 등록 이미지" if total==summary["image_count"] else None,
              warn=None if total==summary["image_count"] else f"합계 {total} ≠ 등록 이미지 {summary['image_count']}")
    fig=go.Figure(go.Scatter(
        x=[r["label"] for r in history],y=[r["image_count"] for r in history],name="등록 이미지 누적",
        mode="lines+markers",fill="tozeroy",line=dict(color="#568ee9",width=2),
        hovertemplate="%{x}<br>등록 누적 %{y}장<extra>"+source+"</extra>"))
    if source!="Mock":
        fig.add_trace(go.Scatter(x=[r["label"] for r in history],y=[r["analysed_count"] for r in history],
            name="분석 결과 이미지 누적",mode="lines+markers",line=dict(color="#e0883e",width=2),
            hovertemplate="%{x}<br>분석 결과 누적 %{y}장<extra>SQLite</extra>"))
    fig.update_layout(height=80 if source=="Mock" else 150,margin=dict(l=8,r=8,t=4,b=20),
                      showlegend=source!="Mock",legend=dict(orientation="h",y=1.15,x=0,font=dict(size=11)),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12),yaxis=dict(visible=source!="Mock",showgrid=False),xaxis=dict(showgrid=False))
    st.plotly_chart(fig,width="stretch",theme="streamlit",key=source.lower()+"_accumulation")


def processing_progress(*, image_count=None):
    if image_count is not None:
        import pandas as pd
        st.success(f"이미지 등록 완료 {image_count}장 · SQLite")
        stages=pd.DataFrame([
            {"처리 단계":"이미지 / Metadata 등록","상태":"완료" if image_count else "등록 데이터 없음"},
            {"처리 단계":"Calibration","상태":"업로드 기본 조건 자동 매칭"},
            {"처리 단계":"AI 분석","상태":"분석 실행 관리에서 실행"},
        ])
        data_table(stages,height=155,row_height=31)
        st.caption("등록 완료는 분석 완료를 의미하지 않습니다.")
        return
    progress=data.processing_progress_mock()
    st.progress(progress["overall_percent"],
                text=f"전체 진행률 {progress['overall_percent']}% · Mock / 계산식 미확정")
    data_table(data.processing_stage_table(),height=155,row_height=31)
    st.caption(f"검토 대기 {progress['review_pending']}건 · 오류 {progress['error_count']}건 (모두 예시)")
