# 공통 CSV 형식

세 측정 방법은 알고리즘이 달라도 같은 열을 가진 `*_predictions.csv`를 만든다.
팀원은 사람 정답을 보지 않고 이 예측 CSV만 제출한다.

## 예측 CSV

| 열 | 의미 | 예시 |
|---|---|---|
| `method` | 측정 방법 이름 | `unet` |
| `image_id` | 확장자를 제외한 원본 파일명 | `sample_01` |
| `measurement_type` | `line`, `gap`, `sidewall` 중 하나 | `sidewall` |
| `y_fraction` | native 높이 기준 측정 위치 | `0.35` |
| `target_id` | 중앙 5개 대상의 왼쪽부터 번호 | `S3` |
| `width_px` | native 픽셀 폭 | `5.42` |
| `width_um` | `width_px × 0.1592` | `0.862864` |
| `center_x_px` | 측정 대상 중심의 native X | `575.2` |
| `left_x_px` | 왼쪽 edge의 native X | `572.5` |
| `right_x_px` | 오른쪽 edge의 native X | `577.9` |
| `runtime_ms` | 사진 한 장 전체 처리시간 | `842.1` |
| `status` | 정상 측정은 `ok`, 실패는 `error` | `ok` |
| `message` | 실패 이유 또는 참고사항 | 빈칸 가능 |

한 사진에서 sidewall만 평가하면 다음과 같이 25행이 생성된다.

```text
Y 위치 5개 × 중앙 sidewall 5개 = 25행
```

`runtime_ms`는 한 사진의 모든 행에 같은 값을 반복해도 된다. 비교 프로그램이
사진당 한 번으로 합친다. 첫 사진은 모델 워밍업이므로 빈칸으로 두고 속도 평균에서
제외한다.

## 사람 정답 CSV

| 열 | 의미 |
|---|---|
| `image_id` | 예측 CSV와 같은 파일 ID |
| `measurement_type` | 이번 비교에서는 `sidewall` |
| `y_fraction` | `0.20`, `0.35`, `0.50`, `0.65`, `0.80` |
| `target_id` | `S1`~`S5` |
| `answer_width_px` | 사람이 정한 정답 폭. µm만 있으면 비워도 됨 |
| `answer_width_um` | 사람이 정한 정답 폭. px만 있으면 비워도 됨 |
| `left_x_px`, `right_x_px` | 선택 사항. 사람이 지정한 좌우 edge |
| `repeat_group` | 반복 촬영끼리 같은 그룹명 |
| `repeat_id` | 반복 번호 |
| `brightness_group` | 같은 위치의 조명 변화 사진끼리 같은 그룹명 |
| `brightness_level` | 예: `2`, `2.5`, `3`, `3.5` |

폭 값만 입력해도 비교할 수 있다. 다만 대상 대응을 검증하려면 가능할 때
`left_x_px`, `right_x_px` 또는 `center_x_px`도 함께 적는 것이 좋다.

## 비교 결과 CSV

- `method_summary.csv`: 방법별 MAE, RMSE, bias, 성공률, 평균 처리시간
- `image_metrics.csv`: 사진별 MAE와 측정 성공률
- `matched_measurements.csv`: 정답 한 건과 예측 한 건의 대응 및 개별 오차
- `benchmark_summary.json`: 종류별 결과, 반복성 SD, 밝기 변화 범위를 포함한 전체 보고서

## 팀원이 제출할 것

1. 자신의 `방법명_predictions.csv`
2. 확인용 overlay 폴더
3. 실행 중 오류가 있었다면 간단한 설명

사람 정답 CSV와 성능 비교 결과는 평가 담당자가 관리한다.
