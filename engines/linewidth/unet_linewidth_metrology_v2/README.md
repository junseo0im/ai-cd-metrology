# UNet 선폭 측정 모듈 (대시보드 연동용)

이미지 한 장을 받아 line·gap 선폭을 재고, 팀 공통 출력 형식(dict/JSON)으로 돌려준다.
선폭 판정(pn/pt)은 별도 모듈 `judgment/`가 측정 결과에 덧붙인다.
측정 알고리즘은 팀 저장소 `linewidth/methods/unet` 과 같다 (브랜치 `linewidth-v2-line-gap`).

## 설치

```powershell
pip install -r requirements.txt
```

## 바로 써 보기

```powershell
python run_example.py "사진.png" --output outputs --device cpu
```

`outputs/` 에 결과 JSON(측정 + 판정)과 확인용 overlay(초록=line, 파랑=gap, 빨강=pn line, 주황=pt line)가 생긴다.

## 코드에서 쓰기

```python
import sys; sys.path.insert(0, r"<이 폴더 경로>")
from metrology.unet_metrology import MetrologyEngine

engine = MetrologyEngine(device="cpu")      # 모델 로드는 한 번만 (약 1~2초)
result = engine.analyze({
    "image_id": "W1_D07_c",
    "image_path": r"C:\...\사진.png",
    "wafer_id": "W1", "die_id": "7", "site_code": "c",
    "batch_id": "B001", "analysis_run_id": "RUN-0001",
    # "roi": {"x_min": 0, "y_min": 0, "x_max": 1133, "y_max": 628},   # 선택, native px
    # "calibration": {"calibration_id": "optiview-native-0.1592", "um_per_px": 0.1592},  # 선택
})
```

사진 한 장에 CPU 로 약 1초 걸린다. 한 장이 실패해도 예외를 던지지 않고 `status="failed"` 와
`error_code` 를 돌려주므로 배치를 계속 돌리면 된다.

판정까지 하려면:

```python
from judgment.judge import judge
judged = judge(result)            # result 는 그대로 두고 'judgment' 를 붙인 새 dict
for line in judged["judgment"]["lines"]:
    print(line["line_id"], line["label"], line["min_um"], line["max_um"])
```

## 입력

| 필드 | 필수 | 설명 |
|---|---|---|
| `image_id` | O | 이미지 고유 ID |
| `image_path` | O | 원본 PNG 경로 |
| `wafer_id`, `die_id`, `site_code`, `batch_id`, `analysis_run_id` | | 그대로 결과에 실어 보냄 |
| `roi` | | `{"x_min","y_min","x_max","y_max"}` native px. 없으면 사진 전체 |
| `calibration` | | `{"calibration_id", "um_per_px"}`. 없으면 `0.1592 µm/px` |

## 출력

| 필드 | 설명 |
|---|---|
| `image_id` (+ 입력 메타데이터) | |
| `status` | `success` / `failed` |
| `algorithm_version` | 모델 파일 해시·epoch·임계값·경계 기준·Y 위치 |
| `calibration_id` | 적용한 보정 ID |
| `measurements` | 아래 목록 |
| `error_code`, `error_message` | 실패 시 원인 |
| `pixel_scale` | 원본 사진 위에 그릴 때 쓰는 좌표 변환값 |

`measurements` 한 항목:

| 필드 | 설명 |
|---|---|
| `y_fraction`, `y_px` | 측정 줄 (ROI 높이의 25/45/65%) |
| `line_id` | `L01`, `L02`… / `G01`… 왼쪽부터. 세 줄에서 같은 선은 같은 ID |
| `measurement_type` | `line`(밝은 띠) / `gap`(어두운 띠) |
| `width_px`, `width_um` | 폭 (native px, µm) |
| `left_x_px`, `right_x_px`, `center_x_px` | 측정 위치 (native px) |
| `left_sidewall_px`, `right_sidewall_px` | 양옆 빨간 sidewall 띠의 폭 |

`error_code`: `BAD_REQUEST`, `IMAGE_NOT_FOUND`, `IMAGE_READ_FAILED`, `INVALID_ROI`,
`NO_SIDEWALL_DETECTED`, `NO_MEASUREMENT`, `INTERNAL_ERROR`

예시 전체는 `metrology/example_io.json`.

## 판정 (judgment/)

기준값은 `judgment/rules.json` 한 파일에 있다. 기준이 바뀌면 이 파일만 고친다.

| 항목 | 현재 규칙 (`2026-11-v1`) |
|---|---|
| 경계 | 빨간 띠 안쪽 끝 (측정값 그대로, `edge_basis: "inner"`) |
| 대상 | line (밝은 띠) 만. gap 은 판정하지 않음 |
| pn | 그 line 의 Y 3곳 중 가장 좁은 값 ≤ 8 µm |
| pt | 그 line 의 Y 3곳 중 가장 넓은 값 ≥ 12 µm |
| 그 외 | normal |

하지 않는 것: 사진 단위 판정, Die/Row 판정, pd (결함 파트), PD 사진 제외 → 통합 Runner.

`judgment` 출력:

| 필드 | 설명 |
|---|---|
| `status` | `judged` / `not_judged` (측정 실패 등, `reason` 에 이유) |
| `rule_version`, `edge_basis`, `measurement_type` | 적용한 규칙 |
| `lines[]` | line 마다 `line_id`, `label`(pn/pt/normal), `reason`, `min_um`, `max_um`, `mean_um`, `n_rows`, `rows[]` |
| `lines[].near_threshold` | 기준값 ± 0.2 µm 안 (사람끼리 측정 차이 수준). 판정은 그대로, 재확인 권장 표시 |
| `lines[].pitch_ok` | 이웃 line 중심 간격이 20 ± 2 µm 인가. False 면 빨간 띠를 놓쳐 구간이 합쳐졌을 수 있음 |

`pitch_ok` 는 line 과 gap 이 뒤바뀐 경우는 잡지 못한다 (간격이 그대로이기 때문).

노광 사진 15장으로 확인한 결과: 노광 70 의 3장(E70_w1_16_lt_pn, E70_w1_24_c, E70_w3_22_lt)은
모든 line 이 pn, 나머지 12장은 모두 normal, pt 는 없었다.

## 좌표

모든 `*_px` 는 **native px** 이다. 현미경 캡처는 픽셀이 2배로 중복 저장되어 있어서 이를 걷어낸 좌표다.
원본 사진 위에 그릴 때는 `pixel_scale` 로 바꾼다 (`run_example.py` 의 `draw_overlay` 참고).

```text
x_원본 = x × factor_x + offset_x + (factor_x − 1) / 2
y_원본 = y × factor_y + offset_y + (factor_y − 1) / 2
```

## 측정 규칙

- 경계는 빨간 sidewall 띠의 **안쪽 끝**이다. line = 이웃한 두 빨간 띠 사이의 밝은 구간, gap = 어두운 구간.
- 측정 줄: 높이의 25%, 45%, 65%.
- 양옆 3% 를 뺀 영역에서 빨간 띠 두 개 사이에 온전히 들어온 **모든** line·gap 을 잰다.
  사진 테두리에서 잘린 구간은 재지 않는다.
- 평균, pn/pt/pd 판정, PD 사진 제외는 이 모듈이 하지 않는다 (통합 Runner 담당).

## Runner 에서 주의할 점

1. **판정은 안쪽 끝 기준이다.** line·gap 값은 빨간 띠 가운데를 경계로 볼 때보다 약 0.9 µm 작다
   (line + gap ≈ 18.2 µm, 설계 10 + 10 = 20 µm). 기준을 바꾸려면 `rules.json` 의
   `edge_basis` 를 `"center"` 로 바꾸면 된다 (sidewall 폭 절반씩을 더해 판정).
2. **`pitch_ok: false` 인 line 은 판정을 믿지 말 것.** 빨간 띠 하나를 놓쳐 구간이 합쳐진 경우다.
   결함이 큰 사진에서는 line 과 gap 이 뒤바뀌어도 값이 나올 수 있으니 결함 파트 결과와 함께 본다.
3. **촬영 밝기.** 아주 밝게 찍은 사진에서는 값이 커지는 경향이 있다 (최대 약 1 µm).
   촬영 밝기를 일정하게 유지한다.
