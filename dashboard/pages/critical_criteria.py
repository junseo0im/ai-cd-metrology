from dashboard.components.layout import page_title,detail_navigation
from dashboard.components.record_editor import render_records

page_title("Critical 기준","","CRITERIA / RECORDS")
render_records("critical_rules")
detail_navigation()
