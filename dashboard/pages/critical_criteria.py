from dashboard.components.layout import page_title,detail_navigation
from dashboard.components.record_editor import render_records

page_title("Critical 기준","판정·집계 기준의 기록 관리 화면입니다. 실제 판정이나 계산은 수행하지 않습니다.","CRITERIA / RECORDS")
render_records("critical_rules")
detail_navigation()
