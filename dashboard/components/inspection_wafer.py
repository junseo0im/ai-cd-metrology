"""I&A-only Mock Die regions. No physical parsing, thresholds, or defect judgement."""
from enum import Enum
import plotly.graph_objects as go
import streamlit as st
from dashboard import data_provider as data

class PatternCriticality(str, Enum):
    NORMAL = "NORMAL"
    MINOR = "MINOR"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNUSABLE = "UNUSABLE"
    NOT_EVALUATED = "NOT_EVALUATED"

# Provisional UI vocabulary/colors only; no numerical decision rule.
CRITICALITY_STYLES = {
    PatternCriticality.NORMAL: ("정상", "#35a86b"),
    PatternCriticality.MINOR: ("경미", "#a5d46f"),
    PatternCriticality.WARNING: ("주의", "#f4d25a"),
    PatternCriticality.CRITICAL: ("심각", "#f39b49"),
    PatternCriticality.UNUSABLE: ("사용 불가", "#e56666"),
    PatternCriticality.NOT_EVALUATED: ("미판정", "#b0bac7"),
}
POSITION_ORDER = ("lt", "c", "rb")


def set_selection(state, regions, die_id=None, position=None):
    """Toggle a region or clear; reset shows the first existing Mock image.

    Each Die's three regions deliberately share its existing illustration.
    Position-specific acquisition files are not available in this UI fixture.
    Overlay preference is retained; no measurement/criticality is calculated.
    """
    same = (die_id is not None and state.get("selected_die_id") == die_id
            and state.get("selected_position") == position)
    selected = None if same or die_id is None else next(
        r for r in regions if r["die_id"] == die_id and r["position"] == position)
    state["selected_die_id"] = selected["die_id"] if selected else None
    state["selected_position"] = selected["position"] if selected else None
    image_id = selected["image_id"] if selected else regions[0]["image_id"]
    state["selected_image_id"] = image_id
    state["_pending_map_image"] = image_id
    state["_inspection_map_revision"] = state.get("_inspection_map_revision", 0) + 1


def build_inspection_wafer(batch_id, selected_die_id=None, selected_position=None):
    regions = data.inspection_wafer_regions(batch_id)
    fig = go.Figure()
    half_width, half_height = .4, .42
    band_height = 2 * half_height / 3
    for criticality, (label, color) in CRITICALITY_STYLES.items():
        subset = [r for r in regions if r["criticality"] == criticality.value]
        x = [(r["col"] - 4.5) * 1.05 for r in subset]
        base = [3.5 + (2 - r["row"]) * 1.5 + half_height
                - (POSITION_ORDER.index(r["position"]) + 1) * band_height for r in subset]
        fig.add_trace(go.Bar(
            x=x, y=[band_height] * len(subset), base=base, width=2 * half_width,
            name=criticality.value + " · " + label,
            marker=dict(color=color, line=dict(color="#8498b6", width=.6)),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
            customdata=[[r["die_id"], r["position"], r["image_id"], r["criticality"]]
                        for r in subset],
            hovertemplate="Die %{customdata[0]} / %{customdata[1]}<br>"
                          "%{customdata[3]} · UI 예시<br>Die 공용 Mock: %{customdata[2]}"
                          "<extra>실제 판정 아님</extra>"))
    fig.add_shape(type="circle", x0=-4.65, x1=4.65, y0=3.5-4.65, y1=3.5+4.65,
                  line=dict(color="#8498b6", width=2),
                  fillcolor="rgba(114,137,175,.08)", layer="below")
    for die in [r for r in regions if r["position"] == "c"]:
        cx, cy = (die["col"]-4.5)*1.05, 3.5+(2-die["row"])*1.5
        fig.add_shape(type="rect", x0=cx-half_width, x1=cx+half_width,
                      y0=cy-half_height, y1=cy+half_height,
                      line=dict(color="#8498b6", width=.8))
        fig.add_annotation(x=cx, y=cy+half_height+.12, text=die["die_id"], showarrow=False,
                           yanchor="bottom", font=dict(size=10))
    if selected_die_id is not None and selected_position in POSITION_ORDER:
        region = next(r for r in regions if r["die_id"] == selected_die_id
                      and r["position"] == selected_position)
        cx, cy = (region["col"]-4.5)*1.05, 3.5+(2-region["row"])*1.5
        top = cy+half_height-POSITION_ORDER.index(selected_position)*band_height
        # Selection outline is independent of all six criticality fill colors.
        fig.add_shape(type="rect", x0=cx-half_width, x1=cx+half_width,
                      y0=top-band_height, y1=top,
                      line=dict(color="#2864e8", width=2.5))
    fig.update_layout(height=250, barmode="overlay", clickmode="event+select", dragmode=False,
                      margin=dict(l=4, r=4, t=4, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Segoe UI, Malgun Gothic, sans-serif", size=11),
                      legend=dict(orientation="v", x=1, xanchor="right", y=1, yanchor="top",
                                  font=dict(size=11)))
    fig.update_xaxes(range=[-4.9,4.9], visible=False, fixedrange=True, constrain="domain")
    fig.update_yaxes(range=[-1.4,8.4], visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1)
    return fig


def inspection_wafer(batch_id, image_key="ia_image"):
    regions = data.inspection_wafer_regions(batch_id)
    if (st.session_state.get("_inspection_map_batch") != batch_id
            or st.session_state.get("_inspection_map_layout") != "8cols_3rows"):
        st.session_state["_inspection_map_batch"] = batch_id
        # Reset stale IDs/image bindings from the previous 3-column layout once.
        st.session_state["_inspection_map_layout"] = "8cols_3rows"
        set_selection(st.session_state, regions)
    # A manual Image ID change selects that Die's center region. Map callbacks
    # use _pending_map_image and must not be overwritten by an old selector value.
    widget_image = st.session_state.get(image_key)
    if ("_pending_map_image" not in st.session_state and widget_image
            and widget_image != st.session_state.get("selected_image_id")):
        match = next((r for r in regions if r["position"] == "c"
                      and r["image_id"] == widget_image), None)
        if match:
            set_selection(st.session_state, regions, match["die_id"], "c")
    revision = st.session_state.get("_inspection_map_revision", 0)
    chart_key = f"inspection_die_{batch_id}_{revision}"

    def handle_click():
        points = st.session_state[chart_key]["selection"]["points"]
        if points:
            die_id, position = points[-1]["customdata"][:2]
            set_selection(st.session_state, regions, die_id, position)

    st.plotly_chart(build_inspection_wafer(batch_id, st.session_state.get("selected_die_id"),
                                          st.session_state.get("selected_position")),
                    key=chart_key, on_select=handle_click, selection_mode="points",
                    theme="streamlit", width="stretch")
    selected = st.session_state.get("selected_die_id")
    position = st.session_state.get("selected_position")
    selection = f"선택 Die {selected} / {position}" if selected else "선택 없음 · 기본 예시 이미지"
    st.caption("Mock 8열×3행 · 각 Die 위 lt / 중 c / 아래 rb · " + selection)


def reset_selection(batch_id):
    set_selection(st.session_state, data.inspection_wafer_regions(batch_id))
