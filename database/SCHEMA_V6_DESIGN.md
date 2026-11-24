# Schema v6 설계 — 다중 Wafer 단일 Batch (TASK 18-B-10B-1)

상태: **운영 적용 완료(TASK 18-B-10B-3/4, 2026-11-23)**. 운영 DB v5 → v6 마이그레이션, 백업 `runtime_data/database/ida.pre_task18b10b_v6_20261123T095414130111Z_641b4e25.sqlite3`.
등록·조회·분석 요청 코드는 v5/v6 버전 분기로 동작하며, v6 DB에서만 다중 Wafer Batch가 활성화된다.
마이그레이션: `database/migrate_v6.py` · 검증: `database/tests/test_migrate_v6.py`, `dashboard/tests/test_multi_wafer_v6.py`

## 1. 현재 v5 관계 (코드 확인)

```
process_runs ◄── wafers.process_run_id (NOT NULL, Wafer 기본 공정)
     ▲
     └── upload_batches.process_run_id (NULL = Wafer 공정 상속, v5)
upload_batches.wafer_id (NOT NULL)  ─►  wafers
upload_batches.peb_temperature_c    (값은 초(s), 열 이름만 °C)
images.batch_id ─► upload_batches
images.physical_site_id ─► pattern_sites.die_id ─► dies.wafer_id ─► wafers
analysis_batch_runs.batch_id ─► upload_batches ; analysis_image_runs.image_id ─► images
```

- 이미지의 실제 Wafer는 `images → pattern_sites → dies.wafer_id`로 **이미지별로 직접 식별 가능**하다.
- 공정(노광·PEB)은 Batch 단위(`COALESCE(b.process_run_id, w.process_run_id)`, `b.peb_temperature_c`)로만 결정된다.
- 같은 Wafer가 여러 Batch·조건에 나오면 Batch 단위 공정으로 구분된다(현 Wafer별 조건표 방식).
- 분석 결과 폴더: `runs/batch-<uuid>/images/<sha256(image_id)>/analysis/` — Wafer와 무관하다.

## 2. "Batch 1 = Wafer 1" 가정 위치 (충돌 지점)

| 위치 | 내용 |
|---|---|
| 트리거 `images_same_wafer_insert/update` | 이미지 Site의 Wafer ≠ `upload_batches.wafer_id`이면 INSERT 차단 |
| `batch_registry._build_preview` | "Batch and image Wafer mismatch" INVALID |
| `batch_registry._prepare_parents` / `_batch_process` | Die·Site 생성과 공정 결정을 `batch["wafer_id"]` 하나로 처리 |
| `migrate_v5.validate_v5` 등 | `b.wafer_id<>d.wafer_id`를 위반으로 판정 |
| `image_registry.get_image` | `wafer_id`로 **Batch의 Wafer**(`b.wafer_id`)를 반환 → 분석 저장 검증이 이를 사용 |
| `request_mapper.prepare_analysis_inputs` | Site Wafer ≠ Batch Wafer → `BATCH_SITE_WAFER_MISMATCH`; 노광·PEB·Calibration wafer_id를 Batch에서 가져옴 |
| `analysis_repository` | `get_image()`의 wafer_id로 결과 identity 검증 |
| `sqlite_provider.batches/images/wafers/latest_analysis` | `b.wafer_id`·Batch 공정 기준 조인, `u.wafer_id` 필터 |
| `inspection_wafer.wafer_context`, `sqlite_input.condition_accumulation`, Wafer 상세 | Batch 행의 `wafer_id`로 Wafer 필터·집계 |

`BatchRunner`, `run_dashboard_analysis`는 요청마다 wafer/노광/PEB를 받으므로 **변경 불필요**.

## 3. 후보 비교

| 기준 | A. Batch-Wafer 관계 테이블 | B. 기존 관계만 활용(이미지→Site→Die) | C. 기존 테이블 컬럼 추가(images에 공정) |
|---|---|---|---|
| v5 데이터 호환 | 1 Batch당 1행 백필, 기존 행 불변 | 불변 | images 행에 NULL 컬럼 추가 |
| 단일 Batch 여러 Wafer | 지원 | 트리거 교체 필요(=스키마 변경) | 트리거 교체 필요 |
| Wafer별 독립 노광·PEB | **지원(행 단위)** | **불가**(Batch 단위 1개) | 이미지 단위로 가능하나 Wafer 내 불일치 위험 |
| 기존 1-Wafer 처리 | 동일 값 미러 | 동일 | 동일 |
| FK 무결성 | (batch, wafer) PK + FK + 멤버십 트리거 | 기존 | 이미지 216행에 공정 중복 저장 |
| Runner 변경 | 없음(mapper만) | 없음 | 없음 |
| I&A 조회 변경 | 조인 1개 교체 | 공정 구분 불가 | 이미지 조인 |
| 마이그레이션 위험 | 낮음(추가 테이블·트리거 교체, 재작성 없음) | — | 낮음~중간 |
| 확장성 | Wafer별 recipe/메모 등 추가 용이 | 낮음 | 이미지 단위로 과분할 |

**권장: A** — 이미지의 Wafer는 B의 기존 관계(Site→Die)로 식별하고, Wafer별 공정만 새 관계 테이블에 둔다.

## 4. 권장 Schema v6

```sql
CREATE TABLE batch_wafers (
  batch_id TEXT NOT NULL REFERENCES upload_batches(batch_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
  wafer_id TEXT NOT NULL REFERENCES wafers(wafer_id)        ON UPDATE RESTRICT ON DELETE RESTRICT,
  process_run_id TEXT REFERENCES process_runs(process_run_id) ON UPDATE RESTRICT ON DELETE RESTRICT, -- NULL=Wafer 공정 상속
  peb_time_s REAL CHECK(peb_time_s IS NULL OR typeof(peb_time_s) IN ('real','integer') ...),         -- 초(s), NULL=미입력
  PRIMARY KEY (batch_id, wafer_id));
CREATE INDEX idx_batch_wafers_wafer ON batch_wafers(wafer_id);
-- 교체: images_same_wafer_insert/update  →  images_batch_wafer_insert/update
--   (이미지 Site의 Wafer가 그 Batch의 멤버 Wafer여야 함)
-- 추가: batch_wafers_keep_images (이미지가 있는 멤버십 삭제 금지),
--       batch_wafers_immutable_keys (batch_id/wafer_id 변경 금지)
```

- `upload_batches.wafer_id`는 테이블 재작성 없이 유지하며 **대표(첫) Wafer** 의미로만 쓴다. 조회에 쓰지 않는다.
- 다중 Wafer Batch는 `upload_batches.process_run_id`·`peb_temperature_c`를 NULL로 두고 `batch_wafers`만 공정 출처로 쓴다(검증 규칙).
- 단일 Wafer Batch는 `batch_wafers` 값과 `upload_batches` 값이 같아야 한다(검증 규칙). 기존 26개 Batch는 이 규칙을 그대로 만족한다.
- 이미지별 유효 공정:
  `images → pattern_sites → dies(d.wafer_id) → batch_wafers(bw.batch_id=i.batch_id AND bw.wafer_id=d.wafer_id) → process_runs(COALESCE(bw.process_run_id, w.process_run_id))`, PEB = `bw.peb_time_s`.

## 5. 보존 방법 (마이그레이션 절차)

한 트랜잭션(`BEGIN IMMEDIATE`) 안에서 처리하고, 중간에 하나라도 실패하면 전체 롤백한다.
- 마이그레이션 전: `validate_v5`, 그리고 기존 7개 표·분석 5개 표·이미지별 유효 공정(v5 경로)의 스냅샷.
- 적용: DDL과 백필.
- 마이그레이션 후: `validate_v6`. 기존 표 전체, 분석 표 전체, 이미지별 유효 공정(v6 경로)이 스냅샷과 같아야 한다. 바뀐 스키마 객체는 v5 단일 Wafer 트리거 2개 교체와 신규 객체뿐이어야 한다.
- 이미 v6인 DB에서 다시 실행하면 검증만 하고 아무것도 쓰지 않는다.
- 운영 DB 경로는 `allow_operational=True` 없이 거부한다.
- 운영 적용 시: Online Backup으로 사본 생성 → 사본 검증 → 승인 후 적용 → SHA·건수 비교.
- Batch·Image·Wafer ID, 파일 경로·SHA, 분석 이력, 채택 상태는 변경하지 않는다.

## 6. 업로드 흐름 (PNG 216장 · ZIP 3개 공통)

1. 업로드 후 `expand_uploads`가 PNG/ZIP을 검증하고 펼친다. 여러 ZIP과 ZIP별 출처는 이미 지원된다. 한도는 512장·512MB·ZIP 128MB이다.
2. 파일명을 파싱해 Wafer·Die·위치 후보를 이미지별로 만든다. 파싱에 실패하면 `NEEDS_METADATA`로 두고, 기존 이미지별 수정 UI에서 보완한다.
3. ZIP 이름에 W 토큰이 있고 내부 파일 Wafer와 다르면 `REVIEW_REQUIRED`(충돌 표시)로 둔다. ZIP 이름만으로 Wafer를 확정하지 않는다.
4. Wafer별로 묶어 "Wafer별 공정조건" 표를 만든다. 감지된 Wafer마다 한 행이고, 노광(필수 숫자)과 PEB(s, 선택)를 받는다.
   - 공통값 적용은 대상 Wafer 목록을 보여주고 **빈 칸에만** 채운다. 이미 값이 있으면 덮어쓰지 않는다.
   - 노광 단위는 입력받지 않고 추정하지 않는다. 기존 Wafer 기록과 값이 같을 때만 그 기록의 단위를 재사용한다.
5. Preview에 다음을 표시한다.
   - Batch 전체 수와 Wafer별 수(합계 일치)
   - Wafer별 노광·PEB
   - READY/REVIEW_REQUIRED/BLOCKED 수
   - SHA 중복과 동일 위치 추가 촬영의 구분, 충돌 사유
6. 기존 승인·중복 처리 정책 그대로 한 트랜잭션에 등록한다. 쓰는 순서는 다음과 같다.
   - `upload_batches` (대표 Wafer, 다중이면 공정 NULL)
   - Wafer별 `process_runs` (기존 Wafer 기록과 다를 때만)
   - `batch_wafers`
   - 이미지 Wafer별 `dies`/`pattern_sites`
   - `images`

## 7. 분석 영향

- BatchRunner는 변경하지 않는다. Batch를 선택하면 그 Batch의 모든 READY 이미지가 대상이 되고, 실행 1회로 처리된다.
- `request_mapper`: 이미지별 `d.wafer_id`와 `batch_wafers`의 노광·PEB를 요청에 넣는다. Calibration 조회도 이미지 Wafer 기준으로 한다.
- `get_image`: `wafer_id`를 `d.wafer_id`로 바꾼다. 그러면 `analysis_repository`의 identity 검증이 이미지 Wafer 기준이 된다.
- 결과 폴더, PD 제외, 부분 실패, 진행률은 이미지 단위라 변경하지 않는다.
- I&A의 Wafer Map, Die 판정, 공정 그래프는 이미 이미지 프레임의 `wafer_id`·노광·PEB로 동작한다. Provider의 이미지 조회 조인만 v6 경로로 바꾼다.

## 8. 2단계 수정 대상

`database/migrate_v6.py`(확정), `database/migrate_v5.py`(ANALYSIS_VERSIONS·검증 분기),
`database/batch_registry.py`, `database/image_registry.py`, `database/analysis_repository.py`,
`inspection/request_mapper.py`, `dashboard/sqlite_provider.py`, `dashboard/components/sqlite_input.py`,
`dashboard/components/tables.py`, `dashboard/components/sqlite_batch_detail.py`, `dashboard/pages/batch_detail.py`,
`dashboard/components/sqlite_wafer_detail.py`, `dashboard/pages/wafer_detail.py`,
`dashboard/components/inspection_wafer.py`, `dashboard/components/analysis_management.py`, 관련 테스트.
변경 없음: `inspection/batch_runner.py`, `inspection/run_dashboard_analysis.py`, 모델·Calibration.
