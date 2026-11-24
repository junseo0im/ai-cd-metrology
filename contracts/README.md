# 공통 분석 인터페이스 v0.2

**Status: DRAFT — NOT APPROVED · Version: 0.2**

팀 피드백으로 확정할 Defect AI / Metrology 공통 호출 형식 초안이다.
Python 표준 라이브러리의 `Protocol`만 정의하며 엔진 구현·실행이나 입력 검증은 포함하지 않는다.
Dashboard, SQLite, 기존 학습·계측·MLflow 코드의 동작을 바꾸지 않는다.

참고: [기존 Interface Contract v0.1](../docs/architecture/analysis_interface_contract_v0.1.md).
v0.1 원본은 보존한다. 필드·상태·Y 위치가 다른 두 초안의 자동 호환을 가정하지 않으며,
인터페이스 버전은 DB Schema 버전과 별개다.

## 코드와 실행 방식

프로젝트 루트를 Python import 경로로 두고 사용한다.

```python
from contracts.analysis_contract import DefectEngine, MetrologyEngine

# DefectEngine.predict(request: dict[str, Any]) -> dict[str, Any]
# MetrologyEngine.measure(request: dict[str, Any]) -> dict[str, Any]
```

기존 Defect AI와 Metrology 코드는 독립적으로 유지한다. 실제 연동에는 팀 코드의 함수명·반환 구조를
변환하는 Adapter와 호출·결과 검증을 담당하는 Runner가 필요하다. 이번에는 둘 다 구현하지 않는다.
PD 검출·계측 제외, 평균, Die/Row 판정 등의 규칙은 추후 별도 통합 Runner에서 관리한다.

## 클래스·라벨·정책

- Defect AI 클래스 후보: `DEFECT_CLASSES = ("s", "o", "sc", "p", "pd")`.
- 별도 선폭 판정 라벨: `WIDTH_LABELS = ("pn", "pt")`. `pn`, `pt`는 Defect AI 클래스가 아니다.
- 위 튜플 순서를 기존 모델의 숫자 class ID와 같다고 가정하지 않는다. 실제 모델과 매핑 확인이 필요하다.
- ROI 상대 Y 위치: `Y_FRACTIONS = (0.25, 0.45, 0.65)`.
- 실행 상태: `STATUSES = ("SUCCESS", "FAILED", "SKIPPED")`.
  `SUCCESS`는 호출 성공이며 정상 이미지 판정이 아니다.

다음은 전달받은 **정책 참고값**이다. Python 계약이나 실제 엔진에 판정 로직으로 하드코딩하지 않는다.

| 항목 | 정책 정보 |
|---|---|
| 설계 선폭 | 10 µm |
| PN | 8 µm 이하 |
| PT | 12 µm 이상 |
| PD 관련 기준 | 4 µm 이하 — 적용 대상·제외 정책 확인 필요 |

정책은 변경 가능하며, 이번 단계에서 실제 판정·제외를 수행하지 않는다.

## 공통 입력

- `image_id`, `image_path`: 분석 대상 이미지 ID와 실행 환경의 원본 경로.
- `wafer_id`, `die_id`: Wafer·Die 식별자.
- `site_code`: 촬영 위치 `lt`, `c`, `rb`.
- `batch_id`: 소속 Batch 식별자.
- `analysis_run_id`: 분석 요청·실행 식별자.
- 선택 입력 `roi`, `calibration`: 전달 구조·좌표계·적용 조건은 미확정.

## Defect AI 출력

- `image_id`: 입력 이미지와 동일한 ID.
- `status`, `model_version`: 실행 상태와 실제 사용 모델 버전.
- `detections`: `class_code`, `confidence`, `bbox_xyxy`를 가진 검출 목록.
- `error_code`: 오류 코드 또는 `null`. 세부 코드 목록은 미확정.

`class_code`는 `s/o/sc/p/pd` 후보를 사용한다. `bbox_xyxy`는 `[x1, y1, x2, y2]` 형태이며
좌표계·단위·경계 포함 규칙은 합의가 필요하다. 빈 `detections` 자체로 정상 확정하지 않는다.

## Metrology 출력

- `image_id`: 입력 이미지와 동일한 ID.
- `status`, `algorithm_version`: 실행 상태와 실제 계측 알고리즘 버전.
- `calibration_id`: 적용한 Calibration ID 또는 `null`.
- `measurements`: `y_fraction`, `line_id`, `width_px`, `width_um`을 가진 개별 계측 목록.
- `error_code`: 오류 코드 또는 `null`. 세부 코드 목록은 미확정.

ROI Y축 **25%, 45%, 65%**에서 각 위치의 **모든 Line/Finger**를 측정 대상으로 한다.
유효한 Calibration이 없는 µm 값은 임의로 환산하지 않는다. 아래 예시는 `width_um: null`을 사용한다.
선폭 라벨 부여와 PD 제외·집계는 별도 Runner 정책으로 관리한다.

## 가상 형식 예시 — 실제 분석 결과 아님

아래 ID·경로·좌표·confidence·측정값은 구조 설명용 가상 값이다.
실제 이미지, 학습·추론·계측 결과 또는 공식 Calibration을 나타내지 않는다.
`roi`, `calibration`의 객체 형식은 미확정이므로 예시에는 `null`을 사용한다.

### 공통 입력 예시

```json
{
  "image_id": "EXAMPLE_IMAGE",
  "image_path": "EXAMPLE_ONLY/images/example.png",
  "wafer_id": "EXAMPLE_WAFER",
  "die_id": "EXAMPLE_DIE",
  "site_code": "lt",
  "batch_id": "EXAMPLE_BATCH",
  "analysis_run_id": "EXAMPLE_RUN",
  "roi": null,
  "calibration": null
}
```

### Defect AI 출력 예시

```json
{
  "image_id": "EXAMPLE_IMAGE",
  "status": "SUCCESS",
  "model_version": "EXAMPLE_MODEL",
  "detections": [
    {"class_code": "s", "confidence": 0.9, "bbox_xyxy": [10, 20, 30, 50]}
  ],
  "error_code": null
}
```

### Metrology 출력 예시

형식 설명을 위해 Line 1개만 축약했다. 실제 반환 목록에는 각 Y의 모든 Line/Finger 결과를 담는다.

```json
{
  "image_id": "EXAMPLE_IMAGE",
  "status": "SUCCESS",
  "algorithm_version": "EXAMPLE_ALGORITHM",
  "calibration_id": null,
  "measurements": [
    {"y_fraction": 0.25, "line_id": "EXAMPLE_LINE_01", "width_px": 40.0, "width_um": null},
    {"y_fraction": 0.45, "line_id": "EXAMPLE_LINE_01", "width_px": 41.0, "width_um": null},
    {"y_fraction": 0.65, "line_id": "EXAMPLE_LINE_01", "width_px": 40.5, "width_um": null}
  ],
  "error_code": null
}
```

## 팀원 확인 필요 항목

- **클래스 수 불일치:** 전달된 `s/o/sc/p/pd`는 총 **5개**이나 사용자는 **4개**라고 표현했다.
  임의 삭제하지 않았으며 최종 4개인지 5개인지 확인해야 한다.
- 각 클래스 코드의 정확한 의미.
- ROI 전달 형식 및 좌표계.
- bbox 좌표계·단위·경계 포함 규칙.
- Calibration 입력 형식과 적용 조건.
- 실제 팀원 코드의 함수명 및 반환 구조.
- 분석 실패·건너뜀의 세부 상태·오류 코드.
- PD 검출 및 계측 제외 정책, 4 µm 기준의 적용 대상.

위 내용은 팀원 피드백을 받아 확정할 예정이며, 실제 분석 엔진 연동은 아직 수행하지 않는다.
