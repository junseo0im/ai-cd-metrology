from dashboard.components.layout import page_title,detail_navigation
from dashboard.components.record_editor import render_records

page_title("실험 환경 및 조건","실험 정보·조건을 기록하고 검토합니다. 실제 분석 코드에는 자동 적용하지 않습니다.","EXPERIMENT / RECORDS")
render_records("experiment_conditions")
detail_navigation()
