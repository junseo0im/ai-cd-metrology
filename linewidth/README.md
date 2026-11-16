# 선폭 측정 (linewidth)

업데이트: 2026-11-17

IDA 패턴 현미경 사진에서 line·gap 폭을 재는 세 방법과, 공통 기준·성능 평가·대시보드 연동 코드.
대시보드에는 **UNet** 을 쓴다 (선택 근거: `docs/COMMON_PROTOCOL.md` 의 성능 평가 절).

## 폴더

```text
linewidth/
├─ team_protocol.py        공통 기준값 (보정값, Y 25/45/65%, line·gap, 중앙 5개, CSV 저장) — 한 곳에만 있음
├─ methods/                비교한 세 측정 방법 (알고리즘)
│  ├─ unet/                UNet sidewall 분할 → 안쪽 끝 사이 거리   (대시보드 채택)
│  ├─ canny/               Sobel-X gradient peak
│  └─ threshold/           녹색 채널 밝기 57% 지점
├─ metrology/              대시보드 공통 입출력 형식으로 UNet 결과를 내는 모듈 + 예시 JSON
├─ evaluation/             사람 정답 만들기·세 방법 실행·비교 (evaluate.py suite/compare)
├─ tracking/               MLflow(DagsHub) 기록 (IDA_Width_Measurement)
├─ training/               UNet 학습·mask 검수 도구 (학습 데이터는 저장소에 없음)
├─ vendor/                 팀 공용 MLflow 모듈 v0.2 사본 (main 에 합쳐지면 삭제)
├─ scripts/                mask 편집기 실행용 .bat
└─ docs/                   COMMON_PROTOCOL.md(측정 기준·평가 결과), CSV_FORMAT.md, TRAINING.md
```

## 설치

```powershell
cd linewidth
pip install -r requirements.txt
```

## 자주 쓰는 명령 (`linewidth` 폴더에서)

```powershell
# 측정 (세 방법 모두 항상 line·gap, 측정 종류 선택 메뉴 없음)
python methods/unet/unet_inference.py batch "사진폴더" --device cpu --expected-count 0
python methods/canny/scripts/run_method2_team_protocol_compare.py "사진폴더" --output-root .\results
python methods/threshold/unified_measurement.py "사진폴더" --output .\results

# 대시보드용 공통 형식 출력 (요청 JSON → 결과 JSON)
python metrology/unet_metrology.py requests.json --output results.json

# 성능 평가: 정답 CSV 와 세 방법 예측 CSV 비교
python evaluation/evaluate.py compare --ground-truth 정답.csv `
  --prediction unet=unet_predictions.csv --prediction canny=canny.csv --prediction threshold=threshold.csv `
  --output 비교결과
```

## 대시보드 연동 (metrology)

`metrology/unet_metrology.py` 의 `MetrologyEngine().analyze(request)` 가 요청 하나를 받아 결과 dict 를 돌려준다.

- 입력: `image_id`, `image_path` (필수), `wafer_id`, `die_id`, `site_code`, `batch_id`,
  `analysis_run_id`, `roi`, `calibration` (선택)
- 출력: `image_id`, `status`, `algorithm_version`, `calibration_id`, `measurements[]`, `error_code`
- `measurements[]`: `y_fraction`, `line_id`(세 Y 에서 같은 선은 같은 ID), `measurement_type`(line/gap),
  `width_px`, `width_um`, `left_x_px`, `right_x_px`, `center_x_px`, 양옆 sidewall 폭
- 양옆 3% 를 뺀 모든 line·gap 을 낸다. 평균·pn/pt/pd 판정·PD 제외는 통합 Runner 가 한다.
- 예시: `metrology/example_io.json`

## 성능 평가 흐름

1. `evaluation/make_manual_y_guide.py` — 사람이 표시할 가이드 그림(Y 25/45/65% 줄) 생성
2. 사람이 빨간 띠 양쪽 경계에 초록 표시선을 긋고 캡처 → `<사진>_i.png`
3. `evaluation/extract_line_gap_ground_truth.py` — 캡처에서 정답 CSV 추출
4. `evaluation/evaluate.py suite run` — 세 방법 실행 후 정답과 비교
5. `tracking/log_width_benchmark.py --core-only` — 핵심 지표 5개를 MLflow 에 기록
