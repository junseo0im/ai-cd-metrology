"""Shared accumulation/status presentation; supplied SQLite values are registration metadata only."""
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.tables import data_table


def accumulation_overview(summary=None, *, source="Mock"):
    summary=data.accumulation_summary() if summary is None else summary
    suffix=" (예시)" if source=="Mock" else " (SQLite)"
    recent="최근 추가" if source=="Mock" else "최근 Batch 이미지"
    st.markdown(f"**전체 Batch** {summary['batch_count']}　 **전체 Image** {summary['image_count']}　 "
                f"**{recent}** {summary['recent_image_count']}"+suffix)
    st.caption(" / ".join(f"{mode} {count}" for mode,count in summary["mode_image_counts"].items())
               +(" · 누적량은 예시 이력" if source=="Mock" else " · 실제 등록 수량"))
    history=summary["history"]
    fig=go.Figure(go.Scatter(
        x=[r["label"] for r in history],y=[r["image_count"] for r in history],
        mode="lines+markers",fill="tozeroy",line=dict(color="#568ee9",width=2),
        hovertemplate="%{x}<br>이미지 %{y}장<extra>"+source+"</extra>"))
    fig.update_layout(height=80,margin=dict(l=8,r=8,t=4,b=20),showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12),yaxis=dict(visible=False),xaxis=dict(showgrid=False))
    st.plotly_chart(fig,width="stretch",theme="streamlit",key=source.lower()+"_accumulation")


def processing_progress(*, image_count=None):
    if image_count is not None:
        import pandas as pd
        st.success(f"이미지 등록 완료 {image_count}장 · SQLite")
        stages=pd.DataFrame([
            {"처리 단계":"이미지 / Metadata 등록","상태":"완료" if image_count else "등록 데이터 없음"},
            {"처리 단계":"분석","상태":"미연결"},
            {"처리 단계":"Calibration","상태":"미연결"},
            {"처리 단계":"계측·YOLO","상태":"미연결"},
        ])
        data_table(stages,height=155,row_height=31)
        st.caption("등록 완료는 분석 완료를 의미하지 않습니다. 처리 진행률은 계산하지 않습니다.")
        return
    progress=data.processing_progress_mock()
    st.progress(progress["overall_percent"],
                text=f"전체 진행률 {progress['overall_percent']}% · Mock / 계산식 미확정")
    data_table(data.processing_stage_table(),height=155,row_height=31)
    st.caption(f"검토 대기 {progress['review_pending']}건 · 오류 {progress['error_count']}건 (모두 예시)")
