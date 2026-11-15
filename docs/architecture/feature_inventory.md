# IDA Inspection Platform — 기능 인벤토리 (Feature Inventory)

- 감사일: 2026-11-15 (KST) · 근거와 구조 설명은 [implementation_audit.md](implementation_audit.md) 참조
- 상태 정의: **VERIFIED**=IMPLEMENTED_VERIFIED(코드+실행/테스트 근거) · **UNVERIFIED**=IMPLEMENTED_UNVERIFIED(코드 존재, 동작 검증 불충분) · **PARTIAL** · **MOCK** · **DRAFT** · **NOT_IMPL** · **BLOCKED**
- **⚠** = 감사 중 TASK 5 동시 수정으로 현재 상태 **검증 보류**(이전 상태 기준 판정)
- 테스트 근거 약어: **DB**=`database/tests`(79 PASS, 유효) · **UI**=`dashboard/tests` AppTest(78 PASS, ⚠보류) · **BR**=TASK C 실제 Chrome(2026-11-15) · **CM**=`canny method/tests`(10 PASS)
- 우선순위: **P0** 다음 단계의 차단 요인 · **P1** 핵심 흐름 · **P2** 품질·확장 · **P3** 후순위
- "대시보드 동작"은 사용자가 실제 화면에서 쓸 수 있는가를 뜻한다. VERIFIED라도 오프라인 실험(예: YOLO 학습)은 대시보드와 연결되지 않았다.

## 상태별 집계

| 범주 | 항목 | VERIFIED | UNVERIFIED | PARTIAL | MOCK | DRAFT | NOT_IMPL | BLOCKED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. 데이터 입력·등록 | 19 | 14 | 0 | 4 | 1 | 0 | 0 | 0 |
| B. 탐색·상세페이지 | 21 | 4 | 0 | 4 | 7 | 0 | 6 | 0 |
| C. Calibration·자동계측 | 14 | 0 | 4 | 3 | 1 | 4 | 2 | 0 |
| D. AI 결함 검출 | 10 | 2 | 0 | 1 | 1 | 3 | 2 | 1 |
| E. 공정·품질 분석 | 11 | 0 | 0 | 0 | 7 | 0 | 4 | 0 |
| F. 모델·데이터셋 관리 | 9 | 0 | 2 | 2 | 2 | 0 | 2 | 1 |
| G. 시스템 품질·운영 | 12 | 6 | 0 | 3 | 0 | 0 | 2 | 1 |
| **합계** | **96** | **26** | **6** | **17** | **19** | **7** | **18** | **3** |

---

## A. 데이터 입력 및 등록

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| A01 | Wafer Batch 등록(다중 PNG) | `database/batch_registry.py`, `dashboard/components/sqlite_input.py` | VERIFIED | DB, UI, BR(3장 등록) | DCC SQLite | upload_batches, images | 실제 72장 Batch 성능 확인 | P1 |
| A02 | Single Image 등록 | `batch_registry._common_issues`(v2 확인), `sqlite_input` | VERIFIED (격리 v2 한정) | DB `test_single_image_v1_blocked_v2…`, UI, BR(v2 QA DB) | DCC SQLite | v2 `single_image` | **운영 DB v1이라 실사용 불가** → v2 전환 승인 | P1 |
| A03 | Repeatability 등록(동일 Site·반복번호) | `batch_registry._build_preview` | VERIFIED | DB, UI, BR(repeat 1/2) | DCC SQLite | images.repeat_index | 반복번호 운영 규칙 확정 | P2 |
| A04 | Lighting 등록(배율·조명·AE) | `batch_registry._condition_metadata` | VERIFIED | DB, UI, BR | DCC SQLite | acquisition_conditions | 조명 설정값 의미('10','30','50') 정의 | P2 |
| A05 | ZIP 입력(내부 PNG) | `upload_storage.expand_uploads` | VERIFIED | DB, UI, BR | DCC(Wafer Batch만) | images(ZIP 자체 미저장) | — | P3 |
| A06 | 업로드 제한·ZIP 보안(경로탈출·압축비·링크) | `upload_storage.py` 상수·검사 | VERIFIED | DB `test_zip_paths…`, `test_upload_limits…`, BR(unsafe.zip) | DCC | — | — | P3 |
| A07 | 파일명 자동 파싱 | `database/filename_parser.py` | VERIFIED | DB | DCC | images.legacy_defect_code 등 | legacy 코드 사전(ncp 등) 정리 | P3 |
| A08 | Metadata 수동 보완(파일명 확인·수정 탭) | `sqlite_input.render_review` | VERIFIED | DB, UI, BR(unknown.png 보완) | DCC SQLite | metadata_status=manual_confirmed | — | P3 |
| A09 | Wafer/Die/Site 위치 연결·자동 생성 | `batch_registry._prepare_parents` | VERIFIED | DB(24 Die/72 Site fixture) | DCC SQLite | dies, pattern_sites | 실제 72 Site 전체 Batch 확인 | P2 |
| A10 | Dataset Role | DB CHECK, `data_control.py` 라디오 | VERIFIED | DB, UI | DCC | upload_batches.dataset_role | 학습 데이터 반출과의 연계(F08) | P3 |
| A11 | SHA-256 중복 검사(기존 DB + Batch 내부) | `batch_registry._build_preview` | VERIFIED | DB, UI, BR | DCC | idx_images_sha256(비고유) | — | P3 |
| A12 | EXCLUDE / REGISTER_SEPARATELY / HOLD | `batch_registry._build_registration_plan` | VERIFIED | DB `test_selection_*`, UI `test_policy_*`, BR | DCC | (결정 자체는 미저장, R13) | 결정 이력 저장 여부 결정 | P2 |
| A13 | 등록 직전 재검증·Transaction·Rollback | `register_batch`(BEGIN IMMEDIATE, preview 재생성 비교) | VERIFIED (단위 테스트) | DB `test_second_insert_failure…`, UI `test_policy_final_insert_failure…` / BR은 BLOCKED | DCC | 전 테이블 | 브라우저 fault-injection은 미검증 | P3 |
| A14 | 원본 영구 저장(runtime_data/raw) | `upload_storage.NewFiles` | VERIFIED | DB, BR, 운영 23/23 파일 존재 | DCC | images.storage_path | **절대경로 → 상대경로 정책**(R04) | P1 |
| A15 | 임시 staging 정리 | `NewFiles.cleanup`, `_reset_draft` | PARTIAL | DB(실패 시 정리) | — | — | 고아 파일 11개 확인, TTL·정리 도구 | P2 |
| A16 | 등록 상태 관리 | `batch_status`, `metadata_status` | PARTIAL | DB | 목록 표시 | upload_batches, images | 상태 Enum·수명주기 없음(항상 registered) | P2 |
| A17 | 노광·공정 조건 입력 | `sqlite_input` 노광 selectbox | PARTIAL | UI | DCC | process_runs | 고정 목록 50/60/70/80만(R12), recipe·processed_at 입력 없음, 단위 미기록 | P1 |
| A18 | 촬영 시각(captured_at) | Schema 컬럼만 | PARTIAL | DB(형식 검증) | ❌ | images.captured_at(0/23) | 입력 경로 결정 | P2 |
| A19 | Mock 입력 미리보기 | `data_provider.upload_batch_preview` | MOCK | UI | DCC Mock | ❌ | 유지(시연용) | P3 |

## B. 데이터 탐색 및 상세페이지

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| B01 | Data Control Center 4카드·Mock/SQLite 전환 ⚠ | `pages/data_control.py` | VERIFIED | UI, BR | ✅ | SQLite 모드 | TASK 5 후 재검증 | P0 |
| B02 | 카드 확대·복원(입력 보존) ⚠ | `data_control.toggle_card` | VERIFIED | UI, BR | ✅ | — | TASK 5 후 재검증 | P0 |
| B03 | Batch 목록(SQLite) ⚠ | `components/tables.batch_table` | PARTIAL | UI, BR(UI-02 글자 겹침 FAIL) | ✅ | upload_batches | TASK 5(전체 스크롤 표 전환) 결과 검증 | P0 |
| B04 | 전체 이미지 목록·유형 필터·페이지네이션 | `sqlite_input.render_registered_images`, `SQLiteProvider.images` | VERIFIED | UI, BR | ✅ | images JOIN | — | P3 |
| B05 | 이미지 원본·Metadata 조회(선택 1장) | 동일 | VERIFIED | UI, BR | ✅ | images.storage_path | — | P3 |
| B06 | Batch 상세(SQLite) | `pages/batch_detail.py` SQLite 분기 | PARTIAL | UI, BR | ✅ | images | Batch 기본정보(Wafer/모드/Role/노광) 표시, 처리 상태·Matrix | P1 |
| B07 | Batch 상세(Mock) Processing Matrix | `data_provider.processing_matrix` | MOCK | UI, BR | ✅ | ❌ | 실제 처리 테이블 연결 시 대체 | P2 |
| B08 | 상태 Donut(단일) | `charts.status_donut` | MOCK | UI, BR | Mock Batch 상세 | ❌ | — | P3 |
| B09 | 전체/자동계측/결함검출 **전환형 Donut 1개** | — | NOT_IMPL | BR(NOT_IMPLEMENTED) | ❌ | ❌ | 결과 테이블 후 구현 | P1 |
| B10 | 이미지별 Processing Matrix(SQLite) | — | NOT_IMPL | BR(NOT_IMPLEMENTED) | ❌ | ❌(처리 테이블 없음) | processing 상태 모델 선행 | P1 |
| B11 | Batch → Image 상세 이동(SQLite) | — | NOT_IMPL | — | ❌ | — | SQLite 이미지 상세페이지 | P1 |
| B12 | Wafer 상세 | — | NOT_IMPL | — | ❌ | (스키마 계층은 존재) | DB 계층 기반 Wafer 화면 | P1 |
| B13 | Die 상세 | — | NOT_IMPL | — | ❌ | (dies 테이블 존재) | Die별 lt/c/rb 이미지 | P2 |
| B14 | 이미지 분석/상세 페이지 | `pages/image_analysis.py` | MOCK | UI | ✅ | ❌ | SQLite 원본 + 결과 Overlay | P1 |
| B15 | I&A Wafer Map 3×8×lt/c/rb(72영역) | `components/inspection_wafer.py` | MOCK | UI, BR(선택·해제·초기화) | ✅ | ❌ | DB 구조·결과 연결 | P2 |
| B16 | 독립 Wafer Map 페이지 ⚠ | `pages/wafer_map.py`, `charts.build_wafer_figure` | MOCK | UI, BR(UI-05 FAIL: P01–P24) | ✅ | ❌ | TASK 5 변경 확인, I&A 구조와 통일 | P2 |
| B17 | 18페이지 Navigation/Router ⚠ | `navigation.py`, `app.py` | PARTIAL | UI, BR(18/18 렌더, UI-06 콜드 URL FAIL) | ✅ | — | 콜드 직접 URL 원인 조사 | P2 |
| B18 | SQLite 상세 직접 링크·새로고침 유지 ⚠ | `apply_batch_link`(Mock만) | NOT_IMPL | — | ❌ | — | query param 기반 SQLite 문맥 | P2 |
| B19 | 전역 Header/Sidebar 데이터 소스 표시 ⚠ | `layout.header`, `sidebar.render` | PARTIAL | BR(UI-04 FAIL) | 혼용 | — | 소스 문맥 전달 | P1 |
| B20 | 전체 Dataset 페이지 | `pages/dataset.py` | MOCK | UI | ✅ | ❌ | SQLite/학습 인벤토리 연결 | P3 |
| B21 | Main Home 상태·Header 검색/알림 | `pages/main_home.py`, `layout.header` | MOCK | UI, BR | ✅ | ❌ | SQLite 수량 반영 | P3 |

## C. Calibration 및 자동계측

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| C01 | Stage Micrometer 기반 Calibration | `docs/calibration_accuracy_protocol.md` (**backup 브랜치에만**) | DRAFT | — | ❌ | ❌ | Ruler 촬영·interval 측정·승인 | P0 |
| C02 | px→µm 변환 코드(X/Y 분리) | `canny method/src/ai_cd_metrology/calibration.py`, `team protocol/team_protocol.py` | UNVERIFIED | CM(로직) | ❌ | ❌ | 공식 scale 확정 후 연결 | P1 |
| C03 | 공식 X/Y Pixel Scale 값 | 비공식 값 2종: 0.1592 µm/native px(상수), 0.0799 µm/px(validation-only) | NOT_IMPL | — | Mock(scale=None) | ❌ | 공식값 확정 전 µm 표시 금지 유지 | P0 |
| C04 | Calibration 적용 범위·버전 저장 | 계약서 §8 `calibrations` 후보 | DRAFT | — | ❌ | ❌ | 테이블 설계·Migration | P1 |
| C05 | Calibration 페이지 | `pages/calibration.py`, `components/calibration.py` | MOCK | UI, BR | ✅ | ❌ | 등록·검증·적용 UI | P1 |
| C06 | ROI 자동 정렬 | `canny method/src/ai_cd_metrology/roi.py`(위치별 고정 비율 ROI, PILOT) | PARTIAL | CM | ❌ | ❌ | 정렬 알고리즘 vs 고정 ROI 결정 | P2 |
| C07 | Finger/Sidewall 검출 | `canny method/.../gradient_canny.py`, `team protocol/unified_measurement.py` | UNVERIFIED | CM(10) | ❌ | ❌ | GT 대비 정확도 검증 | P1 |
| C08 | 5-point 측정 | `team_protocol.Y_FRACTIONS`(0.20/0.35/0.50/0.65/0.80) | UNVERIFIED | CM | ❌ | ❌ | 수동 GT(Y10/50/90)·계약(P1–P5)과 통일 | P1 |
| C09 | Line Width / Gap 계측 | 오프라인 CLI(`run_method2_team_protocol_compare.py`, `unified_measurement.py`) | UNVERIFIED | CM | ❌ | ❌ | Adapter로 DB `image_id` 연결 | P1 |
| C10 | 설계치 대비 오차 | — (I&A "미계산" 표시) | NOT_IMPL | — | Mock 표 | ❌ | Mask 설계치 확정 | P1 |
| C11 | Ground Truth 대비 측정 오차 | `team protocol/CSV_FORMAT.md`(GT·비교 CSV 형식), `data/derived/team_protocol_compare/` | PARTIAL | — | ❌ | ❌ | 비교 프로그램 위치·GT 데이터 확정 | P1 |
| C12 | Measurement Validity | `status ok/error`, usable-line 조건(5점 중 ≥4, SD≤0.10 µm), Method 2 후보 검증 | PARTIAL | CM | ❌ | ❌ | 유효성 상태 목록·기준 합의 | P1 |
| C13 | Repeatability / Reliability 평가 | CSV `repeat_group` 형식 정의 | DRAFT | — | Mock 페이지 | (repeat_index는 저장) | 평가 방법 확정·구현 | P2 |
| C14 | Metrology Adapter·Runner·결과 테이블 | `analysis_interface_contract_v0.1.md` | DRAFT | — | ❌ | ❌ | 계약 승인 → 구현 | P0 |

## D. AI 결함 검출

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| D01 | YOLOv8n 학습(v1: baseline/oversample/augmented) | `runs/{baseline,oversampling,augmentation}/…/weights/best.pt`, `results.csv`, 스크립트 사본 | VERIFIED (오프라인 실행) | 실행 산출물. Test mAP50 0.158(baseline)/0.273(oversample), o AP50=0 | ❌ | ❌ | 성능 개선, 운영 후보 모델 선정 | P1 |
| D02 | YOLO 추론(파이프라인/대시보드) | — | NOT_IMPL | — | ❌ | ❌ | Defect Adapter | P1 |
| D03 | 클래스·라벨 정책 | `data.yaml`(o/p/s/sc), taxonomy v1(8종, backup), 파일명 legacy | PARTIAL | — | — | legacy_defect_code만 | 클래스 체계 확정(R11) | P0 |
| D04 | Bounding Box·Confidence 저장 | 계약서 §5 | DRAFT | — | ❌ | ❌ | ai_predictions 설계 | P1 |
| D05 | Particle 크기 정의 | `transfer/…zip` 내 `particle_minimum_detection_rule_v2_draft.md`, `data/derived/particle_analysis_v1` | DRAFT | — | ❌ | ❌ | 규칙 확정 | P2 |
| D06 | Criticality 판정 | `inspection_wafer.PatternCriticality`(UI 어휘만) | MOCK | UI | I&A Mock | ❌ | 기준 확정 후 규칙 엔진 | P2 |
| D07 | 모호 결과 수동 검토 UI | (오프라인 human review CSV만) | NOT_IMPL | — | ❌ | ❌ | Review 상태·화면 | P2 |
| D08 | 모델 버전·추론 이력 | 계약서 §8 `model_versions`, `ai_inference_runs` | DRAFT | — | ❌ | ❌ | 테이블·Adapter | P1 |
| D09 | v2 실험(combined/corrected/general/geometric/rare_diverse/weighted/class_weight, Colab) | 데이터셋 폴더만 로컬 존재 | BLOCKED | 로컬에 run/가중치 없음(Colab 외부) | ❌ | ❌ | Colab 결과 회수 여부 확인 | P2 |
| D10 | 분류기(MobileNetV3/ResNet18) 실험 | `runs/mobilenet`, `runs/resnet` | VERIFIED (오프라인 실행) | 실행 산출물, Macro-F1 ≈0.46(o/s F1=0) | ❌ | ❌ | 파이프라인 역할 결정 | P3 |

## E. 공정·품질 분석

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| E01 | Wafer Map(실제 위치별 결과) | — | NOT_IMPL | BR(NOT_IMPLEMENTED) | Mock만 | ❌ | 결과 테이블 후 | P1 |
| E02 | 공정 조건·노광 비교 | `charts.exposure_chart`(Mock 관측치) | MOCK | UI | I&A | ❌ | 실제 측정값 집계 | P2 |
| E03 | Lighting 비교 분석 | `components/comparison.py` | MOCK | UI | ✅ | ❌(등록은 ✅) | 비교 계산 | P2 |
| E04 | Repeatability 분석 | 동일 | MOCK | UI | ✅ | ❌(등록은 ✅) | 통계 계산 | P2 |
| E05 | SPC Chart | — (코드 없음) | NOT_IMPL | — | ❌ | ❌ | 기준·시계열 축 선행 | P2 |
| E06 | Spec Limit | — | NOT_IMPL | — | ❌ | ❌ | 팀 확정 | P2 |
| E07 | Critical 기준 페이지 | `pages/critical_criteria.py` | MOCK | UI, BR | ✅ | ❌ | 확정 기준 게시 | P2 |
| E08 | 실험용 판정 기준 | (기존 PASS/WARNING/FAIL 폴더 라벨만) | NOT_IMPL | — | ❌ | ❌ | 기준 버전 관리 | P2 |
| E09 | 자동 리포트 | `pages/results_report.py` | MOCK | UI | ✅ | ❌ | 결과 집계·발행 | P3 |
| E10 | Simulation / What-if | `pages/simulation.py` | MOCK | UI, BR | ✅ | ❌ | 범위 결정 | P3 |
| E11 | 실험 환경 및 조건 | `pages/experiment_environment.py` | MOCK | UI, BR | ✅ | ❌ | 실제 내용 입력 | P3 |

## F. 모델·데이터셋 관리

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| F01 | MLflow 공통 모듈 v0.2 | `tools/mlflow_team/*` — **원격 브랜치 `origin/feature/mlflow-team-integration-v0.2`에만 존재**, mlflow 미설치 | UNVERIFIED | — | ❌ | ❌ | 작업 트리 반영 여부·smoke test | P2 |
| F02 | 학습 코드 ↔ MLflow 연결 | YOLO v1 스크립트는 `SETTINGS mlflow=False` | BLOCKED | DagsHub "historical Run 8개"는 외부라 확인 불가 | ❌ | ❌ | 접근 권한으로 확인 | P2 |
| F03 | 대시보드 MLflow 표시 | `components/mlflow.py` | MOCK | UI | ✅ | ❌ | API 조회 연결 | P3 |
| F04 | Roboflow API 연동 | 없음 (수동 export 흔적: `.rf.` 라벨 파일명, `data/incoming/roboflow_current`) | NOT_IMPL | — | ❌ | ❌ | API 사용 여부 결정 | P3 |
| F05 | 학습 데이터셋 버전 | `data/derived/yolov8_detection_*_v1/_v2` + `conversion_summary.json` | PARTIAL | — | ❌ | ❌ | 버전 레지스트리 | P2 |
| F06 | Train/Val/Test split(그룹 기반) | `data/derived/fixed_split_v1`, `defect_split_policy_v1.md`(backup) | UNVERIFIED | 이번 감사에서 재검증 안 함 | ❌ | ❌ | split 무결성 재확인 | P2 |
| F07 | 가중치·재현성 | `best.pt`, seed 42, deterministic, args.yaml | PARTIAL | — | ❌ | ❌ | 절대경로·스크립트 소실(R02, R04) | P1 |
| F08 | Dataset Role ↔ 학습 데이터 반출 | — (의도적으로 독립) | NOT_IMPL | — | ❌ | dataset_role | 연계 필요 여부 결정 | P3 |
| F09 | 모델 성능 페이지 | `pages/model_performance.py` | MOCK | UI | ✅ | ❌ | 평가 결과 연결 | P3 |

## G. 시스템 품질·운영

| ID | 항목 | 실제 코드/파일 | 상태 | 테스트 근거 | UI 연결 | DB 연결 | 남은 작업 | 우선순위 |
|---|---|---|---|---|---|---|---|---|
| G01 | Mock / SQLite 데이터 분리 ⚠ | `data_provider.py` vs `sqlite_provider.py` | VERIFIED (데이터 수준) | UI `test_explicit_provider_paths…`, BR | DCC | ✅ | 전역 문구 혼용은 B19 | P1 |
| G02 | 운영 Schema v1 무결성 | `schema.sql`, `init_db.py` | VERIFIED | DB, 운영 DB quick_check/FK | — | ✅ | — | P3 |
| G03 | Schema v2 Migration 도구 | `migrate_v2.py`, `schema_v2.sql` | VERIFIED (복사본) | DB `test_migrate_v2`, migration_readiness 기록 | — | ✅ | 운영 적용(별도 승인) | P1 |
| G04 | DB Backup/Restore | Migration 내장 backup, readiness 복원 검증 | PARTIAL | 기록 | — | — | 일상 백업·복원 절차 | P2 |
| G05 | 운영 DB 보호 가드 | `_check_connection(allow_operational)`, read-only connect | VERIFIED | DB `test_13_operational_db_guard…` | — | ✅ | — | P3 |
| G06 | DB·분석 자동 테스트 | `database/tests`, `canny method/tests` | VERIFIED | 이번 감사 재실행 79 + 10 PASS | — | — | — | P3 |
| G07 | Dashboard 자동 테스트(AppTest) | `dashboard/tests/test_*.py` (+신규 `test_task5_ui.py`) | BLOCKED | 78 PASS였으나 실행 중 소스 변경 → 보류 | — | — | TASK 5 후 재실행 | P0 |
| G08 | 실제 브라우저 검증 ⚠ | `docs/qa/dashboard_browser_audit.md` | VERIFIED (TASK 5 이전) | BR 86항목 | ✅ | 격리 DB | TASK 5 범위 재검증 | P0 |
| G09 | 파일·데이터 보안 | ZIP 검증, DB 경로 숨김(`IDA_SHOW_DB_PATH`) | PARTIAL | DB | — | — | 절대경로, 인증 없음 | P2 |
| G10 | 외부 공유·권한 관리 | — | NOT_IMPL | — | 비활성 버튼 | — | 공유 범위 결정 | P3 |
| G11 | 로그·처리 이력 | — (logging 미사용) | NOT_IMPL | — | — | ❌ | processing_events | P2 |
| G12 | 소스 버전관리·패키징 | Git: 대시보드/DB 미추적, `src/`·`scripts/`·`tests/` 소스 부재 | PARTIAL | git status, manifest | — | — | **보존 commit(승인 필요)·소스 복원** | P0 |

---

## Mock 이후 추가 기능 보존 여부

TASK 지시서 §3의 확인 대상 18개. "보존"은 현재 코드에 존재하고 이전 검증을 통과했다는 뜻이며, ⚠ 표시는 TASK 5 수정 대상 파일과 겹쳐 현재 상태 재확인이 필요하다는 뜻이다.

| 기능 | 보존 | 근거 | 비고 |
|---|---|---|---|
| Wafer Batch | ✅ | A01 | |
| Single Image | ✅ (격리 v2) | A02 | 운영 v1에서는 정책상 차단 |
| Repeatability | ✅ | A03 | 분석 페이지는 Mock |
| Lighting | ✅ | A04 | 분석 페이지는 Mock |
| 단일 PNG / 다중 PNG / ZIP | ✅ | A01, A05 | ZIP은 Wafer Batch만 |
| 파일명 자동 파싱 | ✅ | A07 | |
| Metadata 수동 보완 | ✅ | A08 | |
| Wafer/Die/Pattern 위치 연결 | ✅ | A09 | 화면 표시(Map)는 미연결 |
| Dataset Role | ✅ | A10 | |
| SHA-256 중복 검사 | ✅ | A11 | |
| EXCLUDE / REGISTER_SEPARATELY / HOLD | ✅ | A12 | 결정 이력 미저장 |
| 등록 직전 재검증·Transaction·Rollback | ✅ | A13 | 브라우저 fault-injection 미검증 |
| SQLite 원본 저장·조회 | ✅ | A14, B05 | 절대경로 |
| 전체 이미지 목록·유형별 필터 | ✅ | B04 | |
| Batch 목록·상세페이지 이동 | ✅ ⚠ | B03, B06 | TASK 5가 목록을 수정 중 |
| 카드 확대·복원 | ✅ ⚠ | B02 | `data_control.py` 수정됨 |
| Mock/SQLite 데이터 분리 | ✅ ⚠ | G01 | 전역 문구 혼용은 미해결(B19) |
| Schema v2 Migration·Single Image 지원 | ✅ | G03, A02 | 운영 미적용 |

18개 모두 감사 시작 시점 코드에 존재했다. TASK 5 이후에도 유지되는지는 G07·G08 재검증으로 확인해야 한다.
