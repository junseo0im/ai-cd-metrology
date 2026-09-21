# Canny / gradient sidewall measurement

이 폴더는 sidewall 검출과 사용 가능한 라인 선택을 위한 수정 코드이다.

> 이름에는 Canny가 남아 있지만 실제 폭은 signed Sobel-X의 음·양 gradient
> peak 사이 거리로 계산한다. `cv2.Canny` 결과는 진단용이며 폭 계산에는
> 사용하지 않는다.

## 실행 파일

- `scripts/run_method2_team_protocol_compare.py`
  - 공통 벤치마크용 고정 중앙 5개 슬롯
  - 누락을 숨기지 않고 prediction CSV와 overlay 저장
- `scripts/run_canny_usable_selection.py`
  - 전체 sidewall 검출 후 Y 위치 간 track 연결
  - 누락 위치의 국소 재검출
  - 유효 측정 수와 표준편차 조건으로 사용 가능 라인 판정
  - `1=최소`, `2=중간`, `3=최대` 평균 폭 track 선택

## 사용 가능 라인 선택 실행

```powershell
python .\scripts\run_canny_usable_selection.py "사진.png" --choice 2
```

기본 공통 조건:

- native Y 위치: 높이의 20%, 35%, 50%, 65%, 80%
- calibration: `1 native px = 0.1592 µm`
- 측면 3% 영역 제외
- 5개 Y 중 최소 4개 유효 측정
- 기본 허용 표준편차: `0.10 µm`

## 벤치마크 실행

```powershell
python .\scripts\run_method2_team_protocol_compare.py "사진.png" `
  --measurement-types 3 --output-root ".\results"
```

## 테스트

```powershell
python -m pytest .\tests -q
```

필수 패키지: `numpy`, `opencv-python`, `pytest`.
