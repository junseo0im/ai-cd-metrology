from dashboard.components.layout import page_title,detail_navigation
from dashboard.components.record_editor import render_records

page_title("실험 환경 및 조건","","EXPERIMENT / RECORDS")
render_records("experiment_conditions")
detail_navigation()
