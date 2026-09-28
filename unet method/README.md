# U-Net method

U-Net 분할 모델로 sidewall(및 line/gap) 선폭을 측정하는 방법이다. 팀 공통
프로토콜(`team_protocol.py`, `docs/COMMON_PROTOCOL.md`)을 그대로 따르므로
Canny·threshold 방법과 같은 형식의 `unet_predictions.csv`를 만든다.

세 방법의 예측 CSV를 사람 정답과 비교해 점수를 계산하는 `evaluate.py`도 이
폴더에 함께 있다.

## 파일

| 파일 | 역할 |
|---|---|
| `unet_inference.py` | U-Net 모델 정의, 추론, 마스크 후처리, 선폭 측정 (`single`, `batch`) |
| `shared.py` | 이미지 로드·중복 픽셀 제거, 경계 계산, 보정값, overlay 그리기 |
| `team_protocol.py` | 팀 공통 규칙 (`canny method/team protocol/team_protocol.py`와 동일) |
| `models/best_unet_manual.pt` | 학습된 모델 (1.9MB) |
| `evaluate.py` | 예측 CSV와 사람 정답 CSV 비교 (`compare`) |
| `docs/COMMON_PROTOCOL.md` | 세 방법 공통 측정 기준 |
| `docs/CSV_FORMAT.md` | 예측·정답·결과 CSV 열 설명 |
| `requirements.txt` | 검증한 라이브러리 버전 |

학습 코드와 학습 사진은 이 저장소에 올리지 않았다. 측정·비교에는 필요 없다.

## 설치

Python 3.13에서 확인했다.

```powershell
pip install -r requirements.txt
```

## U-Net 측정

모든 명령은 이 폴더(`unet method`)에서 실행한다.

여러 사진 측정 (팀 비교용, sidewall만, CPU):

```powershell
python unet_inference.py batch "사진폴더" --measurement-type 3 --device cpu --expected-count 0
```

- 결과: `outputs/batch_measurements/<실행시각>/`
  - `unet_predictions.csv`: 팀 비교에 제출하는 공통 형식 CSV
  - 사진별 폴더: 측정 JSON, overlay, mask
- `--expected-count 15`처럼 사진 수를 지정하면 개수가 다를 때 실행을 멈춘다.
- `--output 폴더`로 저장 위치를 바꿀 수 있다.

사진 한 장 측정:

```powershell
python unet_inference.py single "사진.png" --measurement-type 3
```

측정 종류 숫자: `1=line`, `2=gap`, `3=sidewall` (여러 개 가능: `1 2 3`).

## 측정 규칙 요약

- 좌표는 캡처 PNG의 2배 중복 픽셀을 제거한 native 픽셀 기준이다.
- 보정값: `1 native px = 0.1592 µm`
- Y 위치: native 높이의 20%, 35%, 50%, 65%, 80%
- 각 Y에서 화면 중앙의 연속된 5개 대상을 왼쪽부터 `S1`~`S5`로 저장한다.
- 대상이 5개보다 적거나 edge가 비정상이면 그 칸은 `status=error`로 남는다.
- 첫 사진은 모델 워밍업이라 폭은 저장하지만 `runtime_ms`는 비워 속도 평균에서
  뺀다. 속도 비교는 세 방법을 같은 노트북에서 다시 실행해 비교한다.

## 세 방법 비교 (평가 담당자)

```powershell
python evaluate.py compare --ground-truth "정답.csv" `
  --prediction unet="unet_predictions.csv" `
  --prediction canny="canny_predictions.csv" `
  --prediction threshold="threshold_predictions.csv" `
  --output "비교결과"
```

- 결과: `method_summary.csv`(방법별 MAE·RMSE·bias·성공률·처리시간),
  `image_metrics.csv`, `matched_measurements.csv`, `benchmark_summary.json`
- 빈 정답·예측 양식 만들기: `python evaluate.py compare --make-templates 양식폴더`
- `evaluate.py suite`는 평가 담당자 PC의 원본 데이터 폴더 구조에서만 동작하므로
  이 저장소에서는 `compare`만 사용한다.
