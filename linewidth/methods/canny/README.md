# canny (gradient) 방법

이름에는 Canny 가 남아 있지만 실제 경계는 signed Sobel-X 의 음·양 gradient peak 이다.
`cv2.Canny` 결과는 진단용이며 폭 계산에는 쓰지 않는다. 경계는 정수 픽셀 단위.
공통 기준은 `../../docs/COMMON_PROTOCOL.md`, 기준값은 `../../team_protocol.py`.

## 실행 (`linewidth` 폴더에서)

```powershell
python methods/canny/scripts/run_method2_team_protocol_compare.py "사진 또는 폴더" --output-root ".\results"
python -m pytest methods/canny/tests -q
```

- 항상 line·gap 을 잰다 (측정 종류 선택 옵션 없음).
- 기준 줄에서 중앙 5개 슬롯을 정하고 Y 25/45/65% 에서 같은 슬롯을 추적한다.
  슬롯을 못 찾으면 숨기지 않고 `status=error` 로 남긴다.
- 폴더 모드는 한 장이라도 기준 줄에서 대상이 5개 미만이면 멈춘다. 여러 장을 평가할 때는
  `evaluation/evaluate.py suite` 처럼 사진별로 호출한다.

예전 `run_canny_usable_selection.py`(sidewall 만 측정, 1/2/3 으로 최소·중간·최대 선 선택)는
line·gap 전체 측정 방향과 맞지 않아 2026-11-17 에 뺐다. git 기록에 남아 있다.

필수 패키지: `numpy`, `opencv-python`, `pytest`.
