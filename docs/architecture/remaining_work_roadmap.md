# IDA Inspection Platform — 남은 작업 로드맵 (Remaining Work Roadmap)

- 작성일: 2026-11-15 (KST) · 근거: [implementation_audit.md](implementation_audit.md), [feature_inventory.md](feature_inventory.md)
- 이 문서는 작업 순서를 **제안**하는 문서다. 승인된 계획이 아니며, 임계값·공정값을 새로 정하지 않는다.
- 기존에 합의한 기능(Wafer Batch / Single Image / Repeatability / Lighting, 중복 정책, Batch 상세, Wafer Map, Calibration, 계측, YOLO, SPC, Criticality, MLflow/Roboflow, 리포트)은 하나도 제외하지 않았다.

---

## 1. 팀에서 확정해야 할 기술·실험 기준

코드만으로는 결정할 수 없는 항목이다. 상태는 **확정 / 논의 중 / 미확정 / 근거 부족**으로 표시했다. "논의 중"은 문서·초안·비공식 값은 있으나 승인 기록이 없다는 뜻이다.

| # | 항목 | 상태 | 현재 근거 | 필요한 결정 | 영향을 받는 작업 |
|---|---|---|---|---|---|
| T01 | Wafer/Die/Pattern **논리** 구조(3행×8열, lt/c/rb) | 확정 | `docs/dataset.md`, Schema CHECK 제약 | — | — |
| T02 | Wafer/Die **물리 좌표**(Die pitch, Wafer 내 실제 배치) | 미확정 | Mock 일러스트만. 대시보드 문구에 "실제 geometry 아님" | 실측 배치도 제공 여부 | Wafer Map(4단계) |
| T03 | Mask 설계 Width / Gap / Pitch | 미확정 | 대시보드 "설계 (µm) 미확정" | 설계값과 출처 | 설계 오차(C10) |
| T04 | **측정 항목 정의**(Outer/Inner/Gap vs line/gap/sidewall vs LINE/GAP/EDGE) | 논의 중 | 소스마다 정의가 다름(R09) | 대표 정의 1개와 변환 규칙 | 계측 결과 테이블(6단계) |
| T05 | Pattern/Finger ID 규칙 | 미확정 | Method 2 `sequence_index`는 "물리 finger ID 아님"으로 명시 | 번호 부여·Y 간 대응 규칙 | 계측 결과 저장 |
| T06 | Calibration 방식·허용 기준 | 논의 중 | `calibration_accuracy_protocol.md`(backup 브랜치) 초안, 비공식 scale 2종(0.1592 / 0.0799) | Ruler 촬영 수행, 공식값, 허용오차, X/Y 분리 여부 | 5단계 전체 |
| T07 | Ground Truth 정의 | 논의 중 | 수동 OptiView 프로토콜(Y10/50/90) vs 공통 CSV 정답(5개 Y) | GT 위치·대상·작성자·버전 | GT 오차(C11) |
| T08 | 측정 위치(ROI·5-point) | 논의 중 | 위치별 고정 ROI(PILOT), Y 0.20–0.80 / 계약서 P1–P5 미확정 | 최종 ROI 방식·Y 비율 | 계측 Adapter |
| T09 | 자동계측 Validity 기준 | 논의 중 | 코드 내 조건(5점 중 ≥4 유효, SD ≤0.10 µm 기본값). 승인 근거 문서 없음 | 유효성 상태 목록·기준 | C12 |
| T10 | Narrowing / Thickening 기준 | 논의 중 | taxonomy v1 정성 정의만, 수치 기준 없음 | 판정 기준 | Criticality |
| T11 | 결함 클래스 체계(4종 vs 8종) | 논의 중 | YOLO o/p/s/sc · taxonomy v1 8종 · 파일명 legacy 코드 | 운영 클래스 schema와 버전 | D03, 추론 저장 |
| T12 | Open/Short/Scratch/Particle Critical 기준 | 미확정 | Critical 기준 페이지 Placeholder | 기준과 버전 | 7단계 |
| T13 | Particle 크기 정의 | 논의 중 | `particle_minimum_detection_rule_v2_draft.md`(초안), 분석 산출물 | 크기 측정 방식·단위(µm 환산은 T06 의존) | Criticality |
| T14 | 촬영·노광·조명 Metadata | 논의 중 | 운영 DB: 노광 단위 NULL, 조명값 '10/30/50' 의미 미기록, captured_at 미수집 | 필수 항목·단위·노광 입력 범위 | 1단계 보완 |
| T15 | Repeatability 평가 방법 | 미확정 | CSV 형식(`repeat_group`)만 정의 | 지표(SD/CV 등)·반복 횟수 | E04 |
| T16 | SPC 방식·Spec Limit | 미확정 | 코드·문서 없음 | 관리도 종류·한계값 출처 | 7단계 |
| T17 | Recipe·판정 기준 버전 관리 | 미확정 | `process_runs.recipe_version` 컬럼만 존재(미사용) | 버전 체계 | 판정 이력 |
| T18 | Image ID 연결 규칙 | 논의 중 | DB `IMG-uuid` vs 분석 CSV `image_id=파일명 stem` | Adapter에서의 매핑 방식 | 6·7단계 |
| T19 | 중복 처리 정책(EXCLUDE 기본·HOLD 전체 차단) | 확정 (구현 반영) | DB-1C/1E 구현·테스트 | 결정 이력 DB 저장 여부만 추가 결정 | A12 |
| T20 | Dataset Role ≠ Train/Val/Test | 확정 | 계약서 §9, DB README | — | — |
| T21 | 운영 Schema v2 적용 | 승인 대기 | 복사본 검증 완료 | 적용 승인과 시점 | 2단계 |
| T22 | 운영 모델 후보(어떤 가중치를 대시보드에 연결할지) | 근거 부족 | v1 Test mAP50 0.16–0.27, open AP50=0. v2(Colab) 결과는 로컬에 없음 | 후보 모델·최소 성능 기준 | 7단계 |

---

## 2. 권장 개발 순서

요청에서 받은 기본 순서 1~10을 유지하되, 감사에서 발견된 **선행 조건 세 가지를 추가**했다.

- **0단계 (신규) 보존·정비**: 근거 R01/R02. 대시보드·DB 작업이 Git에 하나도 없고, 분석 소스가 작업 트리에서 빠져 있다. 다른 작업보다 먼저 처리해야 손실 위험이 사라진다.
- **2.5단계 (신규) 입력 데이터 품질 보완**: 근거 R04/R12/R14. 분석 결과를 쌓기 전에 저장 경로와 Metadata 정책을 고쳐야 나중에 Migration 부담이 줄어든다.
- **4.5단계 (신규) 측정 정의 확정 게이트**: 근거 R09/R10. T04·T06·T08이 정해지지 않으면 계측 결과 테이블을 설계할 수 없다.

### 순서 요약

| 단계 | 작업 | 선행 조건 | 대표 검증 항목 |
|---|---|---|---|
| 0 | 보존·정비 | 사용자 승인(commit 범위) | 아래 상세 |
| 1 | TASK 5 결과 반영·검증 | TASK 5 완료 | 78+신규 AppTest, UI-01~06 재확인 |
| 2 | 운영 Schema v2 전환 | 별도 승인(T21), 0단계 | 레코드·해시 보존, 네 입력 유형 |
| 2.5 | 입력 데이터 품질 보완 | 1~2단계 | 상대경로, 노광 입력, staging 정리 |
| 3 | Batch/Wafer/Die/Image 상세 연결 | 2.5단계 | 등록 이미지 기준 탐색 |
| 4 | Calibration DB·페이지 | T06 확정 또는 "미확정 상태" 저장 정책 | µm 미표시 규칙 |
| 4.5 | 측정 정의 확정 게이트 | T03·T04·T05·T07·T08·T09 | 합의 문서 |
| 5 | 자동계측 결과 DB·알고리즘 연결 | 4, 4.5단계 | image_id 매핑, Run 재실행 보존 |
| 6 | YOLO 추론 결과 DB·모델 연결 | T11·T22, 0단계 | bbox 좌표계, 빈 예측 ≠ 정상 |
| 7 | 결과 조회·Overlay·모호 판정 Review | 5, 6단계 | 예측/GT/검토/판정 분리 |
| 8 | SPC·Criticality·공정 분석 | T10·T12·T15·T16, 7단계 | 기준 버전 기록 |
| 9 | MLflow/Roboflow 운영 연동 | 0단계(브랜치 병합) — **0단계 이후 병행 가능** | smoke run, 중복 로깅 방지 |
| 10 | 최종 통합 검증·리포트 | 전 단계 | 72장 실제 Batch E2E |

### 0단계 — 보존·정비 (신규, 최우선)

| 작업 | 이유 | 비고 |
|---|---|---|
| `dashboard/`, `database/`, `docs/architecture/`, `docs/qa/`를 작업 브랜치에 commit | R01: 현재 Git에 기록이 없음 | **사용자 승인 필요.** `.venv`, `.pip-cache`, `runtime_data/`는 제외 |
| 분석 소스 복원 방식 결정: `backup/experiment-history-20261112`의 `src/`, `scripts/`, `tests/`, docs 18개를 작업 트리에 되돌릴지, 별도 브랜치로 유지할지 | R02: editable install이 빈 `src/`를 가리킴 | 복원 시 `canny method/` 사본과의 중복(R17)도 정리 |
| `tools/mlflow_team`(원격 브랜치) 반영 범위 결정 | F01 | 9단계와 연결 |

검증: `git status`가 깨끗한지, `import ai_cd_metrology` 경로, 기존 테스트(DB 79 / Dashboard / canny 10)가 그대로 PASS하는지.

### 1단계 — TASK 5 반영·검증

- 대상: Batch 목록을 페이지네이션 없는 전체 스크롤 표로 전환, UI-02 글자 겹침 해결, 그리고 이번에 수정된 `layout.py / sidebar.py / navigation.py / wafer_map.py / styles.css`.
- 검증: Dashboard AppTest 전체(+`test_task5_ui.py`) 재실행, 브라우저에서 UI-01~UI-06 재확인, 보존 기능 18개(인벤토리 마지막 표) 회귀 확인, 운영 DB 해시 불변.
- 이번 감사에서 ⚠ 표시한 B01·B02·B03·B16·B17·B18·B19·G01·G08 항목의 상태를 확정한다.

### 2단계 — 운영 Schema v2 전환 (별도 승인)

- `database/README.md` "운영 적용은 별도 승인 후" 6단계 절차를 그대로 따른다.
- 검증: 버전 2, Batch 10 / Image 23 보존, 원본 SHA 동일, Single Image 등록, 기존 세 모드 정상 동작.

### 2.5단계 — 입력 데이터 품질 보완 (신규)

| 항목 | 근거 | 내용 |
|---|---|---|
| `storage_path` 상대경로 정책 | R04 | 기존 23건의 변환은 Migration 사안이므로 별도 승인 필요 |
| 노광 입력 범위 | R12, T14 | 고정 목록(50/60/70/80) 외의 실제 값 입력 허용 여부를 팀이 결정 |
| captured_at·조명값 의미 | R14, T14 | 입력 경로 또는 "수집하지 않음" 결정 |
| staging 고아 파일 정리 | R06 | 보존 기간과 정리 도구 |
| 중복 처리 결정 이력 | R13, T19 | 저장할지 결정 |
| rerun 비용 | R05 | 실제 72장 Batch로 측정한 뒤 preview 캐시 여부 판단 |

### 3단계 — 상세페이지 연결 (등록 데이터 기준, 분석 결과 없이 시작 가능)

아래 "4. 향후 상세페이지 구현 순서"에 정리했다.

### 4단계 — Calibration DB·페이지

- 테이블 후보: `calibrations`(계약서 §8). 적용 범위(배율·카메라·resize 이력), X/Y scale 분리, 버전, 승인 상태를 포함한다.
- 원칙: 공식값이 승인되기 전에는 µm 값을 표시하지 않는다. 비공식 0.1592 / 0.0799를 기본값으로 쓰지 않는다.
- 검증: 승인되지 않은 Calibration은 µm 환산 불가, X/Y 각각 적용, 이미지-Calibration 매칭 규칙 테스트.

### 4.5단계 — 측정 정의 확정 게이트 (신규)

- T03·T04·T05·T07·T08·T09를 확정하고 계약서 v0.1을 v0.2로 갱신해 승인받는다.
- 이 단계 없이 5단계로 가면 측정 항목명·Y 위치·finger 번호가 서로 다른 결과가 섞여 저장된다(R09).

### 5단계 — 자동계측 결과 DB·알고리즘 연결

- 테이블 후보: `measurement_configs`, `measurement_runs`, `measurements`.
- Runner가 DB `image_id` → `storage_path`를 조회해 Adapter를 호출한다. 기존 CLI(`canny method/`, `team protocol/`)를 수정하지 않고 감싸는 방식을 권장한다.
- 검증: 측정 불가를 0으로 저장하지 않음, 재실행 시 이전 Run 보존, 한 이미지 실패가 다른 결과에 영향 없음, GT 대비 오차(T07) 재현.

### 6단계 — YOLO 추론 결과 DB·모델 연결

- 테이블 후보: `model_versions`, `ai_inference_runs`, `ai_predictions`.
- 검증: class_schema_version과 ID 매핑, bbox가 원본 좌표계인지(letterbox 복원), `predictions=[]`를 "검출 없음"으로 저장하고 정상으로 판정하지 않음, 가중치 SHA 기록.

### 7단계 — 결과 조회·Overlay·모호 판정 Review

- 예측, GT Annotation, 사람 검토, 최종 판정을 별도로 저장한다(계약서 §9).
- 이 단계에서 Batch 상세의 전환형 Donut(B09)과 Processing Matrix(B10)가 실제 데이터를 갖게 된다.

### 8단계 — SPC·Criticality·공정 분석

- T10·T12·T15·T16이 확정된 뒤에 구현한다. 판정에는 기준 버전을 함께 기록한다.
- Lighting·Repeatability 페이지를 Mock에서 실제 집계로 전환한다.

### 9단계 — MLflow/Roboflow 운영 연동 (0단계 이후 병행 가능)

- MLflow: v0.2 모듈 smoke test → 신규 학습부터 기록한다. YOLO 내장 MLflow 콜백과 동시에 켜지 않는다.
- Roboflow: 현재 수동 export만 사용 중이다. API를 연동할지 먼저 결정한다.
- 대시보드의 모델 성능·모델 선정 페이지 Placeholder를 실제 Run 조회로 바꾼다.

### 10단계 — 최종 통합 검증·리포트

- 실제 72장 Wafer Batch를 E2E로 처리한다: 등록 → Calibration → 계측 → 추론 → 결과 → Map/Report.
- 실제 브라우저에서 전 페이지를 재검증하고, Backup/Restore 리허설을 한다.

---

## 3. 단계별 예상 검증 항목 (요약)

| 영역 | 반드시 확인할 것 |
|---|---|
| 데이터 보존 | 운영 DB 해시·레코드 수, 원본 SHA, Migration 전후 비교 |
| 경계 분리 | Mock 값이 SQLite 화면에 섞이지 않음, 전역 문구가 데이터 소스와 일치 |
| 분석 결과 | 측정 불가 ≠ 0, 빈 예측 ≠ 정상, COMPLETED ≠ 유효, 예측 ≠ GT ≠ 판정 |
| 재현성 | 알고리즘·모델·Calibration·설정 버전이 Run에 기록, 재실행 시 이전 Run 보존 |
| 규모 | 72장 Batch 등록·분석 시간, 메모리, rerun 반응 시간 |
| 회귀 | 인벤토리의 "Mock 이후 추가 기능" 18개 유지 |

---

## 4. 향후 상세페이지 구현 순서

각 페이지는 **등록 데이터만으로 먼저 열고**, 분석 결과는 결과 테이블이 생긴 뒤 칸을 채우는 방식을 권장한다.

| 순서 | 페이지 | 1차(3단계, 등록 데이터) | 2차(5~7단계 이후, 분석 결과) |
|---|---|---|---|
| 1 | **SQLite Batch 상세 보강** | Batch 기본정보(Wafer·모드·Role·노광·조건·이미지 수), 이미지 목록에서 Image 상세로 이동, URL 파라미터로 새로고침 유지 | 이미지별 Processing Matrix, 전체/자동계측/결함검출 **전환형 Donut 1개** |
| 2 | **Image 상세(SQLite)** | 원본, Metadata, 계층(Wafer/Die/Site), 같은 Site의 다른 이미지(Repeatability/Lighting/중복 SHA) | 계측 Overlay, 5-point 값, 예측 bbox, 검토 상태 |
| 3 | **Wafer 상세 / Map(DB 구조)** | DB의 3×8 Die × lt/c/rb 72 Site와 Site별 등록 이미지 수·누락 표시. **Batch Matrix와 분리 유지** | 위치별 측정값·결함·Criticality 색상 |
| 4 | **Die 상세** | Die의 lt/c/rb 이미지 3종 | Die 단위 측정 요약 |
| 5 | Calibration 상세 | 등록된 Calibration 목록·적용 범위 | 적용 이력 |
| 6 | Repeatability / Lighting 상세 | 같은 Site 이미지 그룹 표시 | 통계·비교 그래프 |
| 7 | 결과 리포트 | — | Batch/Wafer 요약 발행 |

I&A의 Mock 72영역 Map(B15)과 독립 Wafer Map(B16)은 3번을 구현할 때 하나의 컴포넌트로 통일하는 것을 권장한다. 이때 Mock 고정 회귀 테스트(R15)도 함께 갱신해야 한다.
