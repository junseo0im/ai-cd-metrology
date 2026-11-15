"""Bundled SVG illustrations only. Never opens a raw microscopy image."""
import streamlit as st
from dashboard import data_provider as data


def image_panel(image_id,overlay=False,caption=True,max_width=None):
    row = data.image(image_id)
    if row is None:
        st.info("선택된 예시 이미지가 없습니다.")
        return
    filename = row["overlay_asset"] if overlay else row["asset"]
    st.image(data.asset(filename).read_text(encoding="utf-8"),width=max_width or "stretch")
    if caption:
        st.caption(row["image_id"]+" · "+("Overlay 예시" if overlay else "원본 영역 예시")+
                   " / 실제 현미경 이미지·계측 결과 아님")


def comparison_images(mode,overlay=False):
    for row in data.images(mode=mode).to_dict("records"):
        st.markdown("**"+row["label"]+" · 예시 이미지**")
        image_panel(row["image_id"],overlay=overlay,max_width=420)
