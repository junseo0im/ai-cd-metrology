"""결함 검사 (RF-DETR + 도메인 규칙 후처리).

- defect_inference.py : 로컬 ONNX 추론 연결 (engines/defect/defect_adapter_r33 엔진 재사용, API 미사용)
- defect_rules.py     : line/gap 도메인 규칙 후처리
- aoi_pipeline.py     : 수평 보정 → 추론 → 후처리 → 마킹
- ui.py               : Streamlit 화면 (render_inspection)

이 패키지는 운영 DB에 아무것도 쓰지 않습니다.
"""
