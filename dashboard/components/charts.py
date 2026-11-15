"""Provider-driven Plotly components. Native Streamlit theme handles live theme changes."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_provider as data

COLORS = {"완료 예시":"#45aa93","검토 예시":"#d2a34b","대기 예시":"#7398cb"}
MAP_COLORS = {"완료 예시":"#b8e5d8","검토 예시":"#f1d28b","대기 예시":"#bed4f7"}


def theme(fig,height=250):
    # Do not freeze text/grid/tooltip colors in Python: the Streamlit Plotly
    # frontend updates these when its native Light/Dark choice changes.
    fig.update_layout(height=height,margin=dict(l=12,r=12,t=10,b=12),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Segoe UI, Malgun Gothic, sans-serif",size=12),
                      legend_title_text="",legend=dict(orientation="h",y=-.1),
                      hoverlabel=dict(font_size=13))
    return fig


def status_donut(batch_id):
    frame = data.status_counts(batch_id)
    fig = px.pie(frame,names="status",values="count",hole=.72,color="status",color_discrete_map=COLORS)
    fig.update_traces(textinfo="percent",textposition="inside",hovertemplate="%{label}: %{value} (예시)<extra></extra>")
    st.plotly_chart(theme(fig,250),width="stretch",theme="streamlit",key="status_donut_"+batch_id)
    st.caption("처리 상태 비율 (예시) · 실제 처리 결과가 아닙니다")


def build_process_trend():
    frame = data.process_trend_summary()
    fig = px.line(frame,x="exposure",y="mean",color="series",markers=True,error_y="std",
                  custom_data=["sample_count","std"],
                  color_discrete_sequence=["#8297b8","#568ee9"],
                  labels={"exposure":"노광 조건 (예시)","mean":"평균 Outer µm (예시)","series":""})
    fig.update_traces(hovertemplate="노광 %{x}<br>평균 %{y:.3f} µm (예시)<br>n=%{customdata[0]}"
                     "<br>표준편차 %{customdata[1]:.3f} (예시)<extra>%{fullData.name}</extra>")
    fig.update_xaxes(tickvals=[50,60,70,80],showgrid=False)
    fig=theme(fig,260)
    fig.update_layout(legend=dict(orientation="h",y=1.04,yanchor="bottom",x=0),
                      margin=dict(l=12,r=12,t=34,b=30))
    fig.update_xaxes(title_standoff=5)
    return fig


def exposure_chart(key="exposure",height=None):
    frame = data.process_trend_summary()
    previous = int(frame[frame["series"]=="이전 평균 (예시)"]["sample_count"].sum())
    current = int(frame[frame["series"]=="현재 평균 (예시)"]["sample_count"].sum())
    fig=build_process_trend()
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig,width="stretch",theme="streamlit",key=key)
    st.caption(f"가상 Batch 반영 · 이전 n={previous} / 현재 n={current} · 오차 막대=예시 표준편차")


def build_wafer_figure(batch_id):
    frame = data.images(batch_id,mode="Wafer Batch")
    fig = go.Figure()
    fig.add_shape(type="circle",x0=-1,y0=-4.5,x1=8,y1=4.5,
                  line=dict(color="#8498b6",width=2),fillcolor="rgba(114,137,175,.08)",layer="below")
    for status,part in frame.groupby("status"):
        fig.add_trace(go.Scatter(
            x=part["site_col"]-1,y=2-part["site_row"],mode="markers+text",name=status,
            text=part["pattern_id"],textposition="middle center",
            textfont=dict(color="#142941",size=11),
            marker=dict(symbol="square",size=24,color=MAP_COLORS.get(status,"#bed4f7"),
                        line=dict(width=1,color="#8498b6")),
            customdata=part[["image_id","physical_site_id","pattern_id"]].to_numpy(),
            hovertemplate="%{customdata[2]}<br>%{customdata[0]}<br>%{customdata[1]}"
                          "<extra>Mock 3×8 position</extra>"))
    fig.update_layout(clickmode="event+select",dragmode=False,showlegend=True)
    fig.update_xaxes(range=[-1.3,8.3],visible=False,fixedrange=True,constrain="domain")
    fig.update_yaxes(range=[-4.8,4.8],visible=False,fixedrange=True,scaleanchor="x",scaleratio=1)
    fig = theme(fig,250)
    fig.update_layout(legend=dict(orientation="v",x=1,xanchor="right",y=1,yanchor="top"),
                      margin=dict(l=6,r=6,t=5,b=5))
    return fig


def wafer_map(batch_id,key="wafer_map"):
    event = st.plotly_chart(build_wafer_figure(batch_id),width="stretch",theme="streamlit",key=key,
                            on_select="rerun",selection_mode="points")
    st.caption("Mock 원형 Wafer · 3행×8열 / P01–P24 · 실제 제작·측정 위치 아님")
    points = event.get("selection",{}).get("points",[])
    selected = points[-1].get("customdata",[None])[0] if points else None
    valid = set(data.images(batch_id)["image_id"])
    if selected in valid and st.session_state.get(key+"_last") != selected:
        st.session_state[key+"_last"] = selected
        st.session_state["selected_image_id"] = selected
        st.session_state["_pending_map_image"] = selected
    return selected


def defect_chart(batch_id,key):
    frame = data.defect_counts(batch_id)
    fig = px.bar(frame,x="count",y="defect",orientation="h",color_discrete_sequence=["#7398e8"],
                 labels={"count":"이미지 수 (예시)","defect":"결함 표시 (예시)"})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(theme(fig,230),width="stretch",theme="streamlit",key=key)
    st.caption("결함 분포 예시 · detector/classifier 미실행")


def comparison_chart(mode):
    frame = data.images(mode=mode)
    fig = go.Figure()
    for label,column,color in [("Outer","outer_um_example","#568ee9"),("Inner","inner_um_example","#53b7a2"),
                               ("Gap","gap_um_example","#d8ad61")]:
        fig.add_trace(go.Scatter(x=frame["label"],y=frame[column],name=label+" (예시)",
                                 mode="lines+markers",line=dict(color=color,width=3)))
    fig.update_yaxes(title_text="사전 정의된 예시값 (µm)")
    st.plotly_chart(theme(fig,310),width="stretch",theme="streamlit",key="compare_"+mode)
