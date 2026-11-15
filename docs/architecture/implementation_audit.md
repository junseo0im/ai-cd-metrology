# IDA Inspection Platform — 구현 현황 감사 (Implementation Audit)

- 감사일: 2026-11-15 16:35–16:45 (KST), 읽기 전용 조사
- 기준: 작업 디렉터리 `C:\Users\cindi\Desktop\ai-cd-metrology`, 브랜치 `feature/streamlit-dashboard` (HEAD `d8fdbab` = `origin/main`)
- 관련 문서: [feature_inventory.md](feature_inventory.md) · [remaining_work_roadmap.md](remaining_work_roadmap.md) · [analysis_interface_contract_v0.1.md](analysis_interface_contract_v0.1.md) · [../qa/dashboard_browser_audit.md](../qa/dashboard_browser_audit.md)
- 이 문서는 코드·DB·Schema를 변경하지 않았다. 운영 DB는 `mode=ro`로만 읽었고, 감사 전후 SHA-256이 동일하다(`6671dba3…51e9`).

> **판단 원칙** — 이 문서는 세 가지를 구분한다.
> ① **코드 존재** ② **자동 테스트 통과** ③ **실제 대시보드에서 작동(브라우저 확인)**.
> 파일이 있다는 이유만으로 "구현 완료"로 보지 않았다.

---

## 0. 감사 중 발견된 동시 수정 — 검증 보류 범위

감사 도중 Codex의 TASK 5 수정이 진행되었다. 시작 시점 해시 스냅샷과 종료 시점을 비교한 결과:

| 시각 | 변경 파일 |
|---|---|
| 16:39:45 | `dashboard/components/tables.py`, `components/styles.css`, `components/layout.py`, `pages/data_control.py`, `navigation.py` |
| 16:40:01 | `dashboard/components/sidebar.py`, `pages/wafer_map.py` |
| 16:41:27 | `dashboard/tests/test_data_control_unified.py`, `test_data_control_ux.py`, **신규** `test_task5_ui.py` |

- 위 파일에 대한 이 문서의 코드 관찰은 **TASK 5 이전 상태**(16:39 이전에 읽은 내용) 기준이다. 해당 항목은 문서 전반에서 **⚠ 검증 보류**로 표시했다.
- Dashboard 자동 테스트(78개 PASS)는 16:39:33–16:40:15에 실행되어 **수정 전후 상태가 섞였으므로 확정 근거로 쓰지 않는다**.
- `database/`, `canny method/`, `team protocol/`은 감사 중 변경되지 않았다 → 해당 테스트 결과는 유효하다.

**운영 DB 실시간 사용:** 이 문서의 운영 DB 수치(Batch 10 / Image 23, SHA `6671dba3…`)는 16:35–16:45 시점의 값이다. 감사 종료 직후 **16:48:37에 실행 중인 대시보드(8502)를 통해 새 Wafer Batch 1개가 등록되었다**(이미지 2장 `3-7_s.png`, `3-7(2)_p.png`, `manual_confirmed`). 이후 상태는 v1 / Batch 11 / Image 25다. 이 등록은 감사 작업이 아니며, 감사는 운영 DB를 `mode=ro`로만 읽었다. 표준 파일명이 아닌 실제 이미지가 수동 Metadata로 등록되고 있다는 점은 A08 기능이 실사용 중이라는 근거이기도 하다.

---

## 1. 실제 조사 범위

| 영역 | 경로 | 결과 |
|---|---|---|
| Dashboard | `dashboard/` (app, navigation, 18 pages, 14 components, 2 providers, mock_data, tests) | 전체 정독 |
| DB | `database/` (schema v1/v2, init_db, migrate_v2, filename_parser, image_registry, batch_registry, upload_storage, tests) | 핵심 로직 정독 |
| 운영 DB | `runtime_data/database/ida.sqlite3` | 읽기 전용 조회 |
| 분석 코드(추적됨) | `canny method/`, `team protocol/` | 구조·상수·CLI 확인 |
| 실험 산출물 | `runs/`, `data/derived/`, `data/metadata/`, `outputs/pretrained/`, `transfer/` | 목록·요약 지표 확인 (재학습/추론 안 함) |
| 문서 | `README.md`, `docs/*.md`, `docs/architecture/`, `docs/qa/`, `database/README.md`, `dashboard/README.md` | 확인 |
| Git 이력 | `backup/experiment-history-20261112`, `phase-a-skeleton`, `origin/feature/mlflow-team-integration-v0.2`, `origin/unet-method` | `git ls-tree`/`git show`로 읽기만 |
| **없음** | `tools/mlflow_team/` (작업 트리) | 작업 트리에 없음. 원격 브랜치 `origin/feature/mlflow-team-integration-v0.2`에만 존재 |

실행한 검증: DB 단위 테스트(79), Dashboard 단위/AppTest(78, ⚠ 보류), `canny method` pytest(10). 브라우저 자동 탐색은 지시에 따라 재실행하지 않고 TASK C 결과를 참조했다.

---

## 2. 현재 시스템 구조

```text
                        ┌──────────────────────── Streamlit (dashboard/app.py, 18 pages) ───────────────────────┐
                        │  Main Home · Data Control Center(A) · Inspection & Analysis(B) · 공통/정보 페이지          │
                        │                                                                                         │
 data_provider.py ──────┤  Mock 화면 전부 (Batch 상세 Mock, Wafer Map, I&A, Image 분석, Repeatability, Lighting,     │
 (mock_data/*.json,     │  Report, Calibration, Dataset, 모델/설정/Placeholder)                                    │
  SVG 일러스트)          │                                                                                         │
                        │  Data Control Center 'SQLite' 모드 · SQLite Batch 상세                                     │
 sqlite_provider.py ────┤        └─ components/sqlite_input.py                                                     │
                        └──────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                       │ (등록·조회만)
                     database/upload_storage.py ── batch_registry.py ── image_registry.py ── filename_parser.py
                                                       │
                     runtime_data/database/ida.sqlite3 (Schema v1, 7 tables)   runtime_data/raw/<uuid>/<uuid>/원본.png
                                                       ✕ (연결 없음)
   오프라인 분석(독립 실행):  canny method/ (Method 2 gradient/sidewall) · team protocol/ (공통 CSV·5-point·0.1592 µm/px)
                              runs/ (YOLOv8n v1 ×3, MobileNet ×2, ResNet18 ×2 가중치) · data/derived/* (데이터셋·분석 산출물)
   원격/브랜치에만 존재:       tools/mlflow_team (MLflow v0.2) · src/ai_cd_metrology + scripts/ 48개 + tests/ 11개 + docs 18개
```

핵심 관찰:

1. **등록 파이프라인(DB-1A~1E)만 실제로 동작**한다. 분석(Calibration·계측·YOLO)과 대시보드/DB 사이에는 코드 연결이 없다.
2. 분석 알고리즘과 모델은 **오프라인 스크립트·실험 산출물**로 존재하며, 대시보드에서 호출하거나 결과를 DB에 저장하는 경로는 없다.
3. 대시보드 18개 페이지 중 실제 DB 데이터를 표시하는 곳은 **Data Control Center(SQLite 모드)와 SQLite Batch 상세 2곳**뿐이다.

---

## 3. 실제 데이터 흐름 — 연결/단절

| 단계 | 상태 | 근거 |
|---|---|---|
| Upload (PNG/다중/ZIP) | ✅ 연결 | `upload_storage.stage_uploads`, `sqlite_input._registration_body` · DB 테스트 + TASK C 브라우저 PASS |
| Preview | ✅ 연결 | `batch_registry.preview_batch` (읽기 전용, `saved_to_database: False`) |
| Metadata Validation | ✅ 연결 | 파일명 파싱 + 수동 보완 + Wafer 일치·Repeatability·Lighting 검증 |
| Image Registry | ✅ 연결 | `register_selection` → `register_batch` (BEGIN IMMEDIATE, 재검증, Rollback) |
| DB / Raw Storage | ✅ 연결 | `images.storage_path` → `runtime_data/raw/…` (운영 DB 23장 원본 모두 존재 확인) |
| **Calibration** | ❌ **단절** | DB 테이블 없음. 대시보드는 `CAL-DEMO-20X`(scale=None) Mock. 오프라인 코드에 비공식 상수만 존재 |
| **Metrology** | ❌ **단절** | `canny method/`, `team protocol/` CLI가 **파일 경로/폴더**를 입력으로 받음. DB `image_id`를 쓰지 않음(파일명 stem 사용) |
| **YOLO** | ❌ **단절** | 가중치 `runs/*/weights/best.pt` 존재. 추론 코드·Adapter 없음 |
| **Result Storage** | ❌ **단절** | 결과 테이블 0개 (계약서 §8 후보만 존재) |
| **Dashboard 결과 표시** | ❌ **단절** | 분석 페이지 전부 `data_provider`(Mock JSON) 사용 |

**단절 지점은 "DB/Raw Storage → Calibration" 바로 다음부터 전부**이다. 다음 연결을 위해서는 최소한 ① 결과 테이블 ② Runner/Adapter ③ 파일명 기반 ID ↔ DB `image_id` 매핑이 필요하다.

---

## 4. DB 구조 분석 (운영 DB 실측 포함)

### 4.1 테이블과 관계 (Schema v1, `PRAGMA user_version=1`)

```text
process_runs ─1:N─ wafers ─1:N─ dies ─1:N─ pattern_sites ─1:N─ images
                     └──1:N── upload_batches ──1:N──────────────┘
acquisition_conditions ─1:N─ images (nullable)
```

| 테이블 | PK | 주요 FK / 제약 | 운영 행 수 |
|---|---|---|---:|
| process_runs | process_run_id | exposure_value NOT NULL, exposure_unit/recipe/processed_at nullable | 1 |
| wafers | wafer_id | → process_runs | 1 (W1) |
| dies | die_id (`W1/D11`) | → wafers, row 1–3 / col 1–8, die_code=행열 | 9 |
| pattern_sites | physical_site_id (`W1/D11/c`) | → dies, position ∈ lt/c/rb, UNIQUE(die,position) | 17 |
| upload_batches | batch_id (`BATCH-uuid`) | → wafers, upload_mode CHECK, dataset_role CHECK | 10 |
| acquisition_conditions | condition_id (`COND-uuid`) | magnification>0, auto_exposure NULL/0/1, JSON 검증 | 3 |
| images | image_id (`IMG-uuid`) | → batches, → sites, → conditions(nullable), sha256 비고유 인덱스 | 23 |

- 모든 FK `ON UPDATE/DELETE RESTRICT`. Wafer 일관성 trigger 5개.
- 운영 DB: `quick_check=ok`, FK 위반 0, 원본 23/23 존재, SHA 고유값 17/23(중복 6건 별도 등록 보존).
- 운영 데이터 특성: `exposure_unit` NULL, `captured_at` 0/23, `metadata_status` 전부 `filename_parsed`, 조명 설정 문자열 `'10','30','50'`(의미 미문서화), 배율 20.

### 4.2 Batch와 Wafer의 구분

- **Batch = 등록 작업 단위**, **Wafer = 물리적 대상**. `upload_batches.wafer_id` FK로 **Batch 1개는 Wafer 1개에 속함**(여러 Wafer 혼합 Batch 불가).
- Batch 표시명(`W1 · Batch 01`)은 `sqlite_provider.batches()`에서 created_at 순으로 **매번 계산**되는 별칭이며 DB에 저장되지 않는다.

### 4.3 Image ID와 원본 경로

- `image_id`는 등록 시 `IMG-` + UUID4 hex로 자동 발급. 파일명은 `original_filename`에 보존.
- `storage_path`는 **절대 Windows 경로**(23/23). 프로젝트 이동·다른 PC 공유 시 깨진다(위험 R04).
- 원본 bytes/파일명/SHA는 보존되며 재인코딩하지 않는다.

### 4.4 Upload Mode / Dataset Role / 중복 정책

| 구분 | 값 | 비고 |
|---|---|---|
| upload_mode (v1) | wafer_batch, repeatability, lighting | v2에서 single_image 추가 |
| dataset_role | evaluation(기본), training_candidate, development | Train/Val/Test split과 독립 |
| 중복 | SHA-256 비고유 인덱스, 등록 시 EXCLUDE(기본)/REGISTER_SEPARATELY/HOLD | **결정 이력은 DB에 저장되지 않음**: EXCLUDE·HOLD 대상은 기록 없이 사라지고, 별도 등록 이미지도 원본과의 관계를 저장하지 않음(조회 시 SHA로 재계산) |

### 4.5 분석 관련 테이블 유무

| 필요 테이블 | 존재 | 비고 |
|---|---|---|
| Calibration 결과 | ❌ | 계약서 후보 `calibrations` |
| 자동계측 결과 | ❌ | 후보 `measurement_configs / measurement_runs / measurements` |
| YOLO 추론 결과 | ❌ | 후보 `model_versions / ai_inference_runs / ai_predictions` |
| 처리 이력·작업 상태 | ❌ | 후보 `processing_jobs / processing_events`. 현재 `batch_status`는 항상 `registered` |
| 판정 기준·판정 이력 | ❌ | 후보 `inspection_rules / inspection_decisions` |

### 4.6 Schema v1 → v2 변경점

- `upload_batches.upload_mode` CHECK에 `single_image` 추가, `user_version=2`. 나머지 6개 테이블·인덱스·trigger 동일.
- `migrate_v2.py`: 쓰기 잠금 → SQLite backup → `upload_batches` 재구성 → PK/FK/레코드/trigger/index 검증 → 실패 시 Rollback.
- 검증: DB 테스트(`test_migrate_v2.py`) PASS, `runtime_data/migration_readiness/20261115T053143Z/`에서 운영 복사본 Migration·복원 검증 기록 존재.
- **운영 DB는 v1 그대로**(별도 승인 대기). v1에서는 Single Image 등록이 UI에서 차단된다(정상 정책).

---

## 5. Provider·Dashboard 연결 관계

| 구성 | 데이터 출처 | 쓰기 | 비고 |
|---|---|---|---|
| `data_provider.py` | `mock_data/platform.json`(Batch 5 / Image 78), `inspection_wafer.json`, `control_overview.json`, `filename_review.json`, SVG | 없음 | 모든 분석 페이지의 유일한 출처 |
| `sqlite_provider.py` | 운영 DB + `runtime_data/raw` + `uploads_tmp` | 등록만 | `IDA_DB_PATH / IDA_STORAGE_ROOT / IDA_STAGING_ROOT`로 격리 가능 |
| `navigation.prepare_page_context` ⚠ | Mock Batch ID | 세션 | `batch_detail` 진입 시 SQLite 모드여도 Mock Batch ID를 `selected_batch_id`에 설정 → Header가 Mock Batch 문맥 표시(UI-04의 코드 원인 후보) |
| `layout.header` ⚠ | `data.catalog()["notice"]` | — | 데이터 소스와 무관하게 Mock 공지 표시 |
| SQLite Batch 상세 ⚠ | `session_state["selected_sqlite_batch_id"]` | — | URL 파라미터(`?batch_id=`)는 Mock Batch만 지원 → 새로고침/직접 링크 시 SQLite 문맥 소실 |

---

## 6. 페이지별 현황 (실제 코드 기준, 18개)

| # | 페이지 (route) | 주요 목적 | 실제 표시 데이터 출처 | 구현 상태 | Mock 여부 | 연결되지 않은 기능 |
|---|---|---|---|---|---|---|
| 1 | Main Home (`/`) | 허브·이동 | Mock catalog 수량 | 이동 동작, 상태값 Mock | Mock | SQLite 수량 미반영("Mock only" 고정) |
| 2 | Data Control Center (`/data-home`) ⚠ | 입력·축적·Batch·상태 4카드 | Mock 또는 **SQLite** 선택 | **SQLite 등록·조회 실동작** | 병행 | 분석 처리 상태, 진행률 계산 |
| 3 | Inspection & Analysis (`/inspection-home`) | Map·추이·이미지·결과 | Mock | UI 동작(72영역 선택) | Mock | 실제 결과 전부, MLflow |
| 4 | 전체 Dataset (`/datasets`) | 이미지 목록 | Mock 78건 | UI | Mock | SQLite·data/ 인벤토리·split |
| 5 | 모델 성능 (`/models`) | 성능 지표 | 없음 | Placeholder | Mock | MLflow·평가 결과 |
| 6 | 설정 (`/platform-settings`) | 연결 설정 | Mock connections | Placeholder | Mock | 모든 설정 |
| 7 | Calibration (`/data-calibration`) | Scale 관리 | Mock (scale=None) | Placeholder | Mock | 등록·검증·적용 |
| 8 | Batch 상세 (`/data-batch-detail`) | Batch 정보·처리 상태 | Mock **또는** SQLite | **부분** | 혼합 | (SQLite) Processing Matrix, 전환형 Donut, 이미지 상세 링크 |
| 9 | Wafer Map (`/wafer-map`) ⚠ | 위치별 결과 | Mock P01–P24 | UI | Mock | 실제 3×8×(lt/c/rb) 구조, 결과 |
| 10 | 이미지 분석 (`/image-analysis`) | 원본·Overlay | Mock SVG | UI | Mock | 실제 원본(SQLite), Overlay, 분석 실행 |
| 11 | 결과 리포트 (`/reports`) | 리포트 | Mock | Placeholder | Mock | 발행·집계 |
| 12 | Repeatability (`/inspection-repeatability`) | 반복 비교 | Mock | UI | Mock | 통계 계산 (등록은 실동작) |
| 13 | Lighting (`/inspection-lighting`) | 조명 비교 | Mock | UI | Mock | 비교 계산 (등록은 실동작) |
| 14 | 실험 환경 및 조건 | 장비·공정·Mask·Calibration | 없음 + Mock Calibration | Placeholder 6탭 | Mock | 실제 내용 |
| 15 | Simulation | What-if | 없음 | Placeholder | Mock | 계산 |
| 16 | Critical 기준 | 판정 기준 | 없음 | Placeholder 2탭 | Mock | 기준값·판정 |
| 17 | AI 활용 | 전후 비교 | 없음 | Placeholder 2탭 | Mock | 내용 |
| 18 | 모델 선정 기준 | 선정 근거 | 없음 | Placeholder | Mock | MLflow·선정 결과 |

### 6.1 Batch 상세페이지 — 요구 요소별

| 요구 요소 | Mock 모드 | SQLite 모드 |
|---|---|---|
| Batch 기본정보 | Wafer·모드·노광(예시) | 표시명 + 전체 ID만. Wafer/모드/Role/노광은 목록 카드에만 있음 |
| Batch 소속 이미지 처리 상태 | 예시 상태 컬럼 | ❌ (처리 테이블 없음) |
| 전체/자동계측/결함검출 **전환형 Donut 1개** | ❌ 단일 상태 Donut만 (MOCK) | ❌ |
| 이미지별 Processing Matrix | Mock 테이블 | ❌ |
| 이미지 상세페이지 연결 | ❌ | ❌ (Batch 내 이미지 원본·Metadata 조회는 ✅) |

### 6.2 Wafer 상세/Map — 물리 구조

| 요구 요소 | 상태 |
|---|---|
| 물리 Wafer/Die/Pattern 계층 | DB Schema에는 **반영됨**(3행×8열, lt/c/rb). 화면에서 DB 구조를 그리는 페이지는 없음 |
| 3×8 Die + lt/c/rb 표현 | I&A 카드에만 Mock으로 존재(72영역, Die당 이미지 1장 공유). 독립 Wafer Map은 이전 P01–P24 표현 ⚠ |
| 위치별 이미지 | ❌ (SQLite 이미지 ↔ Map 연결 없음) |
| 위치별 측정값·결함 결과 | ❌ |

**Batch Processing Matrix(이미지×처리단계 표)는 물리 Wafer Map이 아니다.** 두 화면은 다른 목적의 별도 페이지로 유지해야 한다.

---

## 7. 구현 상태 요약

상세 항목은 [feature_inventory.md](feature_inventory.md) (96개 항목). 요약:

| 상태 | 개수 | 대표 |
|---|---:|---|
| IMPLEMENTED_VERIFIED | 26 | 등록 파이프라인 전반, 중복 정책, Rollback, v2 Migration 도구, 운영 DB 보호, YOLO/분류기 오프라인 학습 실행 |
| IMPLEMENTED_UNVERIFIED | 6 | Method 2/sidewall 계측 코드, 5-point, px→µm 변환, MLflow v0.2 모듈, split |
| PARTIAL | 17 | SQLite Batch 상세, 등록 상태 관리, 노광 입력, ROI, Validity, 클래스 정책, Backup/Restore, 보안 |
| MOCK_ONLY | 19 | 분석 페이지 전부, Calibration 페이지, Criticality, Donut, 리포트 |
| DRAFT_ONLY | 7 | 분석 계약서(Adapter/Runner/결과 테이블), Calibration 프로토콜, Particle 크기 규칙 |
| NOT_IMPLEMENTED | 18 | 결과 저장, 추론 연결, Wafer/Die/Image 상세(SQLite), SPC, Spec, 전환형 Donut, 권한 |
| BLOCKED | 3 | v2 Colab 학습 산출물, DagsHub MLflow 이력, Dashboard 테스트(동시 수정) |

---

## 8. 구조적 위험요소·기술부채

**[확인]** = 코드/파일/DB로 직접 확인. **[추정]** = 코드 근거는 있으나 실제 영향 미측정.

| ID | 영향 | 구분 | 내용 | 근거 |
|---|---|---|---|---|
| R01 | **높음** | 확인 | `dashboard/`, `database/`, `docs/architecture/`, `docs/qa/` 전체가 **Git 미추적(untracked)**. 현재 브랜치 HEAD는 `origin/main`과 동일 → 대시보드·DB 작업 전체가 작업 트리에만 존재 | `git status` |
| R02 | **높음** | 확인 | 작업 트리에서 분석 소스 소실: `src/ai_cd_metrology/`·`scripts/`(48개)·`tests/`(11개)는 `__pycache__`만 남음, docs 18개 부재, `pyproject.toml` 부재. `.venv`의 editable install이 빈 `src`를 가리켜 `import ai_cd_metrology`가 빈 namespace로 로드됨. **원본은 `backup/experiment-history-20261112`(로컬·원격)에 보존** | 2026-11-12 `project_file_manifest.csv` 대비, `git ls-tree` |
| R03 | 높음 | 확인 | TASK 5 동시 수정 중 → 현재 Dashboard 상태 미확정 | §0 |
| R04 | 중 | 확인 | `images.storage_path` 절대경로(23/23), YOLO `data.yaml`/`args.yaml`도 절대경로 → 이동·공유 시 깨짐 | DB 조회, data.yaml |
| R05 | 중 | 추정 | Streamlit rerun마다 `provider.preview()`가 staging 이미지 전체를 다시 읽고 SHA-256 계산, `connect()`가 호출마다 v1 전체 Schema 비교 + `quick_check` + `foreign_key_check` 수행(1회 rerun에 여러 번). 72장 Batch ≈ 64 MB(실측 평균 0.9 MB/장)를 매 상호작용마다 재해시. 실제 72장 업로드 성능은 미측정 | `sqlite_input.py` `_registration_body`, `sqlite_provider.connect`, `init_db._validate` |
| R06 | 중 | 확인 | 고아 staging 파일: `runtime_data/uploads_tmp/`에 11개(12 MB, 00:44부터) 잔존. TTL/정리 정책 없음 | 파일 목록 |
| R07 | 중 | 확인 ⚠ | Mock/SQLite 문맥 혼용: Header 공지·Sidebar "MOCK DATA ONLY"·Batch 문맥이 SQLite 모드에서도 Mock (UI-04). 카드 수치 자체는 섞이지 않음 | `layout.header`, `navigation.prepare_page_context` (TASK 5 이전) |
| R08 | 중 | 확인 ⚠ | SQLite Batch 상세가 세션 상태에만 의존 → 새로고침/링크 공유 시 Mock 화면으로 전환 | `batch_detail.py`, `apply_batch_link` |
| R09 | 중 | 확인 | **측정 정의가 3갈래**: Method 2 `Outer/Inner/Gap(E1–E6)` · team protocol `line/gap/sidewall`(Y 0.20–0.80 5점) · 계약서 `LINE/GAP/EDGE`(P1–P5) · 수동 GT `Y10/Y50/Y90`. 대시보드 Mock은 Outer/Inner/Gap과 Line/Gap/Edge를 혼용 | 각 소스 |
| R10 | 중 | 확인 | **Scale 값 2종**: `0.1592 µm/native px`(team protocol 하드코딩, 출처 미문서화) vs `0.0799 µm/px`(OptiView validation-only, x축만). 0.0799×2≈0.1598로 2배 픽셀 복제 가설과 정합하나 **공식 calibration 아님** | `team_protocol.py`, `data/derived/calibration_validation/` |
| R11 | 중 | 확인 | 클래스 체계 불일치: taxonomy v1 8종(normal/open/short/narrowing/thickening/deformation/particle/scratch) vs YOLO 4종(o/p/s/sc) vs 파일명 legacy(p/pd/pn/s/nc/ncp…) | backup 브랜치 문서, data.yaml |
| R12 | 중 | 확인 | SQLite 노광 입력이 고정 목록 `[None,50,60,70,80]` + 기존값만 허용 → 실제 다른 노광값의 신규 공정 등록 불가 | `sqlite_input.py` 노광 selectbox |
| R13 | 중 | 확인 | 중복 처리 결정(EXCLUDE/HOLD/별도 등록 사유)과 업로드 이력이 DB에 남지 않음 → 감사 추적 불가 | `_prepare_parents`, `register_batch` |
| R14 | 낮음-중 | 확인 | `captured_at` 입력 경로 없음(0/23) → 향후 시계열/SPC 축 부재 | DB, UI grep |
| R15 | 낮음-중 | 확인 | Mock 전용 회귀 테스트가 이전 구조를 고정(예: `test_wafer_grid_has_24_unique_patterns_inside_circle`) → Wafer Map 교체 시 테스트 수정 필요 | `dashboard/tests` |
| R16 | 낮음 | 확인 | 처리 상태 모델 부재: `batch_status`는 항상 `registered`, `metadata_status`는 자유 TEXT | Schema, 운영 DB |
| R17 | 낮음 | 확인 | 중복 구현: `team_protocol.py` 2벌(`team protocol/`, `canny method/team protocol/`), `ai_cd_metrology` 2벌(canny method 사본 vs backup 원본), 학습 스크립트가 `runs/*`에 사본으로 분산 | 파일 목록 |
| R18 | 낮음 | 확인 | 인증·권한 없음(`DEMO USER` 비활성 버튼). 로컬 전용 전제. 외부 공유 시 원본 이미지 노출 | `layout.header` |
| R19 | 낮음 | 확인 | 애플리케이션 로그·처리 이벤트 기록 없음(`logging` 미사용) | grep |

### 테스트 사각지대

- 실제 72장(또는 ZIP) 연구 이미지 Batch 성능·메모리 (TASK C도 합성 이미지만 사용)
- 브라우저 수준 부분 INSERT 실패 Rollback (TASK C BLOCKED; 단위 테스트로는 검증됨)
- 다중 사용자/동시 등록, 프로세스 강제 종료 후 복구
- SQLite Batch 상세의 새로고침/직접 링크
- 분석 코드(canny method 10개)의 **정확도**(GT 대비)는 테스트 범위 밖

---

## 9. 검증 근거 요약

| 근거 | 결과 | 유효성 |
|---|---|---|
| `python -m unittest discover -s database/tests` | **79 PASS** | 유효 (감사 중 미변경) |
| `python -m unittest discover -s dashboard/tests` | 78 PASS | ⚠ **검증 보류** (실행 중 소스 변경) |
| `canny method` pytest | **10 PASS** | 유효 |
| TASK C 실제 Chrome 탐색 (2026-11-15 15:33) | 86항목: PASS 71 / FAIL 6 / BLOCKED 1 / NOT_IMPLEMENTED 8 | TASK 5 이전 상태 기준 |
| 운영 DB 읽기 | v1, Batch 10 / Image 23, integrity ok | 유효 (감사 전후 해시 동일) |
| Migration 준비 기록 (05:31Z) | 복사본 v2 전환·복원 검증 | 기록 확인만(재실행 안 함) |
