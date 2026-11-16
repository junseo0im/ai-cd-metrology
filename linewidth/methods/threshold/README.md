# threshold 방법

녹색 채널 밝기 profile 에서 어두운 sidewall 띠를 찾고, 띠 바닥과 주변 밝기 사이
57% 지점(`SIDEWALL_THRESHOLD_FRACTION = 0.57`)을 경계로 잡는다. 경계는 선형 보간한 소수 픽셀.
공통 기준은 `../../docs/COMMON_PROTOCOL.md`, 기준값은 `../../team_protocol.py`.

## 실행 (`linewidth` 폴더에서)

```powershell
python methods/threshold/unified_measurement.py "사진 또는 폴더" --output ".\results"
```

- 항상 line·gap 을 잰다 (측정 종류 선택 메뉴 없음). 경로를 안 주면 입력을 묻는다.
- 결과: 사진별 `measurements_by_y.csv`(모든 대상), `predictions.csv`(대상별 Y 평균 요약),
  `report.html`, overlay.
- 성능 평가용 중앙 5개 CSV 는 `evaluation/evaluate.py suite` 가 만든다.
- 경계 비율 0.57 은 노트북1 사람 정답으로 다시 확인했다 (0.64 가 나은 세트도 있었지만
  노트북2 에서는 0.57 이 나아 유지).
