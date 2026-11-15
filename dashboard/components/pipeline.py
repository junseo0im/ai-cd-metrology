"""Provider-driven process and image-state components; no processing is performed."""
from html import escape
import streamlit as st
from dashboard import data_provider as data

STATE_COLORS = {"registered":"#8d9fb7","processing":"#568ee9","completed":"#46b5a1",
                "review":"#dcaa45","error":"#d77783"}


def processing_pipeline(rows=None):
    steps = []
    for row in data.catalog()["pipeline"] if rows is None else rows:
        steps.append('<div class="pipeline-step"><b>'+escape(row["stage"])+'</b><small>'+
                     escape(row["detail"])+'</small></div>')
    st.markdown('<div class="pipeline">'+'<span class="pipeline-arrow">›</span>'.join(steps)+'</div>',
                unsafe_allow_html=True)


def image_status_tiles(rows=None, *, notice=None):
    tiles = []
    for row in data.catalog()["tile_states"] if rows is None else rows:
        color = STATE_COLORS[row["state_key"]]
        tiles.append('<div class="status-tile"><div class="mini-image"></div><b>'+
                     escape(row["label"])+'</b><small><span class="state-dot" style="background:'+
                     color+'"></span>'+escape(row["state"])+'</small></div>')
    st.markdown('<div class="image-tiles">'+"".join(tiles)+'</div>',unsafe_allow_html=True)
    st.caption(notice or "고정 Mock 상태 · 실제 처리 중/완료/오류가 발생한 것이 아닙니다.")
