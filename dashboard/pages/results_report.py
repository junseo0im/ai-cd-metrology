import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,select_batch,metric_summary,placeholder

page_title("결과 리포트","리포트 구성과 identity 연결을 확인하는 예시 화면입니다.","B / REPORT")
batch_id = select_batch("report_batch")
metric_summary(batch_id)
with st.container(border=True):
    st.subheader("리포트 구성 예시")
    st.dataframe(data.identity_frame(batch_id),hide_index=True,width="stretch")
    placeholder("리포트 발행 · Placeholder","PDF/CSV 발행·저장·승인·실제 성능지표 집계는 미구현입니다.")
