# Analysis Interface Contract v0.1

**Status: DRAFT — NOT APPROVED**

**Version: 0.1**

- 실제 계측 알고리즘·YOLO 추론 실행 모듈은 현재 대시보드에 미연결이다.
- 본 문서는 팀원 합의 전 변경 가능한 인터페이스 설계 초안이다.
- 실제 계측·YOLO 모듈의 입출력 코드를 확인해 확정한 명세가 아니다.
- 필드 이름, 자료형, 필수 조건, 상태 및 테이블 후보는 변경될 수 있다.
- 문서 버전은 **DB Schema Version 1과 별개**이다.
- 본 문서 추가로 현재 DB Schema나 프로그램의 동작은 변경되지 않는다.
- 아래 향후 구조와 기능은 구현 완료를 의미하지 않는다. 실제 분석 결과를 제시하지 않는다.

## 1. 목적 및 현재 범위

프로젝트: **AI 기반 포토 공정 IDA 패턴 선폭 자동 측정, 설계 오차율 분석 및 결함 검출 시스템**

광학현미경 IDA Pattern 이미지를 등록하고 Wafer/Die/촬영 위치·공정 조건을 관리한 뒤,
향후 독립적인 자동계측과 결함 검출 결과를 저장·조회하기 위한 공통 입출력 후보를 정리한다.
최종적으로 Streamlit에서 Wafer Map, Overlay, 계측값, 설계치 대비 오차 및 결함 검토 정보를
표시하는 것을 목표로 한다.

### 1.1 이미지 및 식별 구조

- Wafer당 3행 × 8열, 총 24개 Die 위치를 사용한다.
- 표시용 Die 코드: 11~18, 21~28, 31~38. DB의 문자열 PK `die_id`와 표시 코드 `die_code`는 구분한다.
- Die 내 촬영 위치는 `lt / c / rb`이며, 기본 촬영 구성은 72장/Wafer이다.
- Repeatability·Lighting 촬영으로 같은 `physical_site_id`에 여러 `image_id`가 연결될 수 있다.
  72장은 이미지 등록 수의 상한이 아니다.
- 파일명 예시 `w1_12_rb_pn.png`는 식별·기존 Metadata 확인용이다.
  파일명의 `legacy_defect_code`, 사람의 정답 Annotation, YOLO 예측 결과는 서로 다른 정보이다.
- 결함 코드가 없거나 파싱에 실패했다는 이유로 정상 이미지라고 판정하지 않는다.

### 1.2 현재 구현과 향후 구현의 경계

| 영역 | 현재 범위 |
|---|---|
| DB-1A | SQLite Schema Version 1의 핵심 7개 테이블 |
| DB-1B | 파일명 파싱, 이미지 Metadata 등록·조회, SHA-256 중복 감지 |
| DB-1C | Batch Preview, 수동 Metadata 보완, 중복 검사, 명시적 승인 후 일괄 등록, Transaction·Rollback |
| Streamlit 데이터 입력 | 이미지 등록·파일명 확인 및 수정의 Mock UI. 실제 SQLite 업로드·등록 연결은 향후 DB-1D 범위 |
| 분석 통합 | 분석 결과 테이블, Inspection Runner, Metrology/Defect Adapter는 향후 구현 대상 |

핵심 테이블은 `process_runs`, `wafers`, `dies`, `pattern_sites`,
`upload_batches`, `acquisition_conditions`, `images`이다.
이전 완료 보고 기준 DB-1A/B/C 자동 테스트는 48개 PASS이며,
이번 문서 작업에서는 재실행하지 않는다.

DB-1C는 테스트 파일 경로를 참조하는 백엔드 검증 단계이다.
운영용 이미지 영구 보관·복사 및 Streamlit 연결까지 완료된 시스템으로 간주하지 않는다.

## 2. 권장 데이터 흐름과 모듈 책임

```text
Streamlit Data Control Center
  → 이미지 저장소 + SQLite
  → Inspection Runner (향후)
      ├─ Metrology Adapter (향후)
      └─ Defect Adapter (향후)
  → Result Repository + SQLite (향후)
  → Streamlit Inspection & Analysis
```

| 모듈 | 책임 후보 | 경계 |
|---|---|---|
| Data Control Center | 이미지 입력, Metadata 확인·수정, Batch 등록 승인 | 알고리즘 구현 및 실제 분석 결과 생성과 분리 |
| 이미지 저장소 + SQLite | 원본 경로, image_id, Wafer/Die/Site, Batch·촬영 조건 관리 | 파일 보관과 Metadata 저장의 정책은 후속 설계 |
| Inspection Runner | 대상·설정 조회, 모듈 호출, 실행 상태 관리, 출력 검증, 저장 요청 | 계측·검출 알고리즘과 분리 |
| Metrology Adapter | 계측 모듈의 입력 변환·호출, Calibration·ROI·Finger 계측 결과의 계약 변환 | 최종 알고리즘이나 ROI 정렬 기능이 구현됐다는 의미가 아님 |
| Defect Adapter | 학습된 모델의 로드·추론 호출, 예측 결과의 계약 변환 | 학습, GT 수정 및 최종 Criticality 판정과 분리 |
| Result Repository | Run·개별 결과·설정 참조·오류 이력 저장과 조회 | 저장 성공과 알고리즘 실행 성공을 구분 |
| Inspection & Analysis | 저장 결과 조회, Wafer Map·Overlay·비교·검토 상태 표시 | 화면 내부에 알고리즘 코드를 직접 구현하지 않음 |

Metrology와 Defect는 Streamlit 없이도 호출 가능한 독립 모듈로 연결하는 것을 권장한다.
두 분석은 독립적으로 성공·실패할 수 있어야 하며, 한 모듈의 실패로 다른 모듈의 성공 결과를
삭제하거나 덮어쓰지 않는다.

별도 API 서버, 작업 큐, Docker 또는 GPU 서버는 현재 필수 요구사항이 아니다.

## 3. 공통 입력 명세 — Draft

아래는 Runner와 Adapter 사이의 입력 후보이다. DB의 모든 Metadata를 전달하지 않고,
각 모듈에 필요한 정보만 전달한다. Required/Conditional의 세부 조건은 실제 연동 시 재검토한다.

| Field | Type | Required | Description |
|---|---|---|---|
| image_id | string | Yes | 등록 이미지 ID. 파일명 대신 결과 연결의 기준으로 사용 |
| image_path | string | Yes | 실행 환경에서 읽을 수 있는 원본 이미지 경로 |
| batch_id | string | Yes | 등록 Batch ID |
| physical_site_id | string | Yes | Die 내 촬영 위치 레코드 ID |
| analysis_type | string | Yes | 단일 호출의 분석 종류 후보: METROLOGY 또는 DEFECT |
| config_id | string/null | Conditional | 실행 설정 식별자. 설정 저장·참조 방식은 미확정 |
| calibration_id | string/null | Metrology | 계측 Calibration 참조. 없는 경우 처리 정책은 후속 합의 |
| model_version_id | string/null | Defect | 사용할 모델 버전 식별자 |

Image → Pattern Site → Die → Wafer의 연결 및 Batch의 Wafer 일치를 조회로 검증하는 것을 권장한다.
추가 Metadata가 필요하면 기존 식별자로 조회하거나 명시적 입력 항목을 합의한다.
경로나 파일명을 기반으로 분석 대상의 ID를 임의 변경하지 않는다.

## 4. Metrology Adapter — Draft

### 4.1 입력 후보

공통 입력 중 필요한 필드에 다음을 추가한다.

| Field | Type 후보 | 의미 |
|---|---|---|
| image_id | string | 결과를 연결할 원본 이미지 |
| image_path | string | 분석 대상 원본 경로 |
| calibration_id | string/null | Calibration 식별자 |
| scale_x_um_per_px | number/null | 원본 이미지 x축의 µm/px |
| scale_y_um_per_px | number/null | 원본 이미지 y축의 µm/px. x축과 같다고 임의 가정하지 않음 |
| measurement_config | object | ROI, 측정 위치 등 계측 설정. 내부 구조는 미확정 |
| algorithm_version | string | 실제 사용 알고리즘 버전 |

Calibration ID와 전달 scale의 일치, 배율·촬영 조건·resize 이력에 따른 적용 가능성은
후속 검증 항목이다. Calibration이 없거나 유효성이 확인되지 않았는데 임의의 µm 값을
만들지 않는다. px-only 실행 허용 여부와 NOT_CONFIGURED 처리 조건은 팀 합의가 필요하다.

### 4.2 출력 후보

| Field | Type 후보 | 의미 |
|---|---|---|
| measurement_run_id | string | 계측 실행별 식별자 |
| image_id | string | 입력 이미지와 동일한 식별자 |
| status | string | 공통 실행 상태 후보 |
| algorithm_version | string | 실행한 알고리즘 버전 |
| calibration_id | string/null | 실제 적용한 Calibration 참조 |
| measurements | array of object | 개별 측정 항목. 실패 항목 표현 정책은 미확정 |
| overlay_path | string/null | Overlay 산출물 참조. 원본 이미지와 별도 보관 |
| error_code | string/null | 실행 수준 오류 코드 |
| error_message | string/null | 실행 수준 오류 설명 |

### 4.3 개별 Measurement 후보

| Field | Type 후보 | 의미 및 미확정 경계 |
|---|---|---|
| point_index | string | P1~P5의 측정 위치 식별자 |
| y_normalized | number | 0~1 상대 Y 위치. 원본/ROI 기준 및 P1~P5의 정확한 비율은 미확정 |
| finger_index | integer/string | Finger 식별 번호. 정렬·번호 부여·Y 간 물리적 대응 규칙은 합의 필요 |
| measurement_type | string | LINE / GAP / EDGE 후보. 최종 물리 정의는 미확정 |
| value_px | number/null | 픽셀 측정값. 측정 불가 시 0으로 대체하지 않음 |
| value_um | number/null | 유효한 Calibration을 적용한 값. 임의 환산 금지 |
| validity_status | string | 개별 측정 유효성. 상태 목록과 판단 기준은 미확정 |
| failure_reason | string/null | 측정 불가·유효성 부족 사유 |

광학현미경 이미지의 Calibration 기반 px→µm 변환, ROI 정렬, 반복 Finger 계측,
Y 상대 위치의 5-point 계측, 가능한 모든 Finger 측정이 **설계 후보**이다.
측정 불가와 실제 패턴 결함은 구분한다.

최종 ROI 알고리즘, Edge 정의·검출 방법, Finger Pairing, LINE/GAP/EDGE의 물리 정의 및
출력 구조, Calibration 매칭, 이상 이미지 처리, Mask 설계치 및 허용오차는 아직 확정하지 않는다.
기존 특정 계측 결과 필드와 이 계약의 필드를 임의로 동일시하거나 기존 알고리즘을 변경하지 않는다.

## 5. Defect Adapter — Draft

### 5.1 입력 후보

| Field | Type 후보 | 의미 |
|---|---|---|
| image_id | string | 원본 이미지 ID |
| image_path | string | 분석 대상 원본 경로 |
| model_version_id | string | 모델 버전 식별자 |
| weight_path | string | 실제 사용할 가중치 경로. 현재 연결된 경로를 가정하지 않음 |
| class_schema_version | string | 클래스 의미·ID 매핑의 버전 |
| inference_config | object | 추론 설정. 상세 필드 및 값은 미확정 |

### 5.2 출력 후보

| Field | Type 후보 | 의미 |
|---|---|---|
| inference_run_id | string | 추론 실행별 식별자 |
| image_id | string | 입력 이미지와 동일한 식별자 |
| status | string | 공통 실행 상태 후보 |
| model_version_id | string | 실제 사용 모델 버전 |
| predictions | array of object | 검출 결과 목록 |
| overlay_path | string/null | 원본과 분리된 예측 Overlay 참조 |
| error_code | string/null | 실행 수준 오류 코드 |
| error_message | string/null | 실행 수준 오류 설명 |

### 5.3 개별 Prediction 후보

| Field | Type 후보 | 의미 |
|---|---|---|
| class_id | integer/string | 사용 모델의 클래스 ID. 모델·schema 버전으로 해석 |
| class_name | string | 해당 ID에 대응하는 클래스 이름 |
| confidence | number | 모델이 반환한 confidence. 최종 결함 확정이나 Criticality가 아님 |
| bbox_xyxy | array of 4 numbers | [x_min, y_min, x_max, y_max] 형식 후보 |
| mask_path | string/null | 선택적 segmentation mask 참조. Detection 모델에서 필수로 만들지 않음 |
| review_status | string/null | 사람 검토 상태. 목록·초기값은 미확정 |

결함 클래스 후보는 Open(`o`), Short(`s`), Scratch(`sc`), Particle(`p`)이다.
목록과 숫자 ID 순서는 실제 가중치·데이터셋·class_schema_version에 따라 확인하며,
본 문서에서 고정 매핑을 확정하지 않는다.

- 예측과 Ground Truth를 구분하며 예측으로 Annotation을 자동 대체하지 않는다.
- 예측 결과와 최종 Criticality 판정은 별도이다. 높은 confidence도 자동 확정 근거가 아니다.
- 추론 성공과 `predictions=[]`는 **검출 결과 없음**이며 **정상 확정**이 아니다.
- bbox는 원본 이미지 픽셀 좌표계를 권장한다. 전처리 resize/letterbox가 있으면 원본 좌표로
  복원하고, mask·Overlay도 같은 원본과 대응하도록 하는 계약을 후속 검토한다.
- bbox 경계 포함 방식, mask 인코딩, confidence 유효 범위의 검사 규칙은 실제 모듈과 합의한다.
- 모델 버전 및 실제 추론 설정을 기록한다. 현재 모델 로드·추론이 연결됐다는 의미는 아니다.

## 6. 공통 실행 상태 및 오류 형식 — Draft

### 6.1 실행 상태 후보

| Status | 의미 후보 |
|---|---|
| PENDING | 실행 요청이 준비되어 대기 중 |
| RUNNING | 해당 분석 실행 중 |
| COMPLETED | 해당 분석 호출이 완료됨 |
| FAILED | 분석 실행 중 오류로 완료하지 못함 |
| SKIPPED | 명시적 사유로 실행을 건너뜀 |
| NOT_CONFIGURED | 필요한 모듈·설정이 준비되지 않아 실행하지 못함 |

COMPLETED는 계측값이 모두 유효하거나 이미지가 정상이라는 뜻이 아니다.
실행 상태, 개별 계측 유효성, 결함 검토 상태, 최종 Criticality를 분리한다.
DB-1C의 VALID/NEEDS_METADATA/DUPLICATE_CANDIDATE/INVALID는 **등록 전 검증 상태**이며,
위 분석 실행 상태와 별개의 개념이다.

### 6.2 오류 형식 후보

| Field | Type 후보 | 원칙 |
|---|---|---|
| status | string | 실행 상태로 실패·미설정·건너뜀을 구분 |
| error_code | string/null | 프로그램이 구분할 안정적 코드. 코드 목록은 후속 합의 |
| error_message | string/null | 오류 원인을 설명하는 메시지. 토큰·비밀번호 등 비밀정보 포함 금지 |

오류는 image_id, analysis_type 및 해당 run_id와 연결한다.
개별 Measurement의 failure_reason과 실행 전체의 error_code/error_message를 구분한다.
실패 시 결과 목록·부분 성공·Overlay를 어느 범위까지 보존할지는 미확정이며,
실패를 성공 값이나 정상 판정으로 대체하지 않는다.
호출 예외를 이 오류 형식으로 변환하는 경계와 저장 실패의 표시·재시도 방식도 후속 합의 대상이다.

## 7. Inspection Runner의 호출·검증·저장 책임 — Draft

1. Image ID/Batch ID로 대상 및 연결 Metadata를 조회한다.
2. 이미지 파일의 접근 가능 여부와 등록된 식별 관계를 확인한다.
3. 선택한 Calibration, 계측 설정 또는 모델·가중치·클래스 schema를 확인한다.
4. 요청한 분석 종류에 해당하는 독립 Adapter를 호출한다.
5. 분석별 실행 식별자와 실행 상태를 관리한다.
6. 반환값의 필수 필드, image_id/run_id 대응, 자료형·단위·좌표계 및 버전 참조를 검사한다.
   세부 유효 범위와 부분 성공 규칙은 계약 합의 후 정한다.
7. Result Repository에 해당 Run·결과·설정 참조의 저장을 요청한다.
8. 개별 분석 실패 사유와 저장 실패를 기록하고, 다른 분석의 성공 결과는 보존한다.

Runner는 알고리즘·결함 판정 규칙을 화면 코드에 복제하지 않는다.
결과 저장의 원자성, 재시도·중복 실행 처리와 비동기 실행 여부는 후속 검토 사항이다.
이 절은 책임 제안이며 Runner나 Repository를 구현하는 문서가 아니다.

## 8. 향후 SQLite 확장 및 Streamlit 연결 후보

다음은 **확정 테이블이나 현재 존재하는 Schema가 아닌 설계 후보**이다.

| 후보 | 용도 후보 |
|---|---|
| calibrations | Calibration 식별·적용 조건·scale |
| measurement_configs | 계측 설정 및 버전 |
| measurement_runs | 이미지별 계측 실행 이력 |
| measurements | Run에 속한 개별 측정값·유효성 |
| model_versions | 모델·가중치·클래스 schema 버전 참조 |
| ai_inference_runs | 이미지별 추론 실행 이력 |
| ai_predictions | Run에 속한 개별 예측 |
| processing_jobs | 실행 요청·진행 관리 |
| processing_events | 실행·오류 이력 |
| inspection_rules | 향후 팀이 합의할 판정 규칙 |
| inspection_decisions | 예측·계측 결과와 분리된 최종 판정 이력 |

향후 결과 Run은 기존 images.image_id와 연결하고, 세부 결과는 각 Run과 연결하는 것을
권장한다. 실제 PK/FK, 자료형, 제약, 테이블 분리 및 Migration은 후속 설계 대상이다.

Streamlit은 Runner 실행 요청과 Repository 조회를 통해 Wafer/Die/Site별 결과·Overlay·검토
상태를 표시하는 방향을 권장한다. 설계치 대비 오차와 Criticality는 기준이 확정된 뒤 별도로
연결한다. DB-1D의 등록 UI 연결과 분석 실행 연결은 서로 다른 단계이다.

현재 7개 테이블 및 Schema Version 1은 변경하지 않는다.
문서 버전 0.1은 DB 스키마 변경이나 Migration 실행을 의미하지 않는다.

## 9. 버전 및 데이터 관리 원칙

- 원본 이미지는 보존하고 결과·Overlay는 별도 산출물로 관리한다.
- image_id로 이미지를 연결하고 분석 실행마다 별도 run_id를 부여하는 것을 권장한다.
- 알고리즘·모델 버전, 분석 설정 및 Calibration 참조를 기록한다.
- 동일 이미지 재분석 시 이전 Run 및 결과를 보존한다.
- 예측, 정답 Annotation, 사람 검토, 최종 판정을 구분한다.
- Dataset Role은 EVALUATION / TRAINING_CANDIDATE / DEVELOPMENT라는 사용 목적이다.
  DB의 소문자 저장 코드와 UI 표시를 구분하며, YOLO Train/Val/Test Split이나 결함 판정과
  직접 연결된 것으로 가정하지 않는다.
- MLflow는 향후 학습 Run·성능·모델 버전 연결 대상으로, Roboflow는 Annotation·Dataset
  관리 대상으로 고려한다. 이 문서는 API 연결이나 동기화 구현을 포함하지 않는다.

## 10. Open Questions / To Be Confirmed

- 팀원 모듈의 실제 입출력과 공통 계약의 필드 이름·자료형·필수 조건이 일치하는가?
- 원본 이미지 보관·경로 접근·권한·hash 확인 및 Overlay 보관 정책은 무엇인가?
- Run ID 발급 주체, 재실행·중복 요청·저장 원자성·부분 성공·재시도 규칙은 무엇인가?
- Calibration 유효성, 배율·촬영 조건·resize 이력 매칭, x/y scale 및 px-only 실행 정책은 무엇인가?
- P1~P5의 정확한 Y 비율과 좌표 기준, finger_index의 순서 및 Y 간 대응은 어떻게 정하는가?
- 최종 ROI·Edge·Pairing과 LINE/GAP/EDGE의 물리적 정의·단위·출력 구조는 무엇인가?
- 측정 실패와 실제 패턴 결함, 계측 유효성과 정확도·허용오차를 어떻게 구분하는가?
- 실제 가중치와 class_schema_version의 ID/name 매핑, 추론 설정, bbox/mask 좌표·형식은 무엇인가?
- 실행 상태 전이, 오류 코드, 개별 유효성·검토 상태의 목록과 표시 규칙은 무엇인가?
- predictions가 비어 있는 경우와 사람이 정상으로 확인한 경우를 UI에서 어떻게 구분하는가?
- Mask 설계치·허용오차·Criticality 기준은 누가 합의하며 버전과 검토 이력을 어떻게 남기는가?
- 결과 테이블·FK·Index·Migration, Streamlit 등록/분석 연결의 구현 범위와 순서는 어떻게 합의하는가?

모든 항목은 실제 코드 연동 전 검토 대상이다. 본 문서만으로 설계 승인, 분석 모듈 연결,
최종 판정 기준 확정 또는 운영 준비 완료를 선언하지 않는다.

