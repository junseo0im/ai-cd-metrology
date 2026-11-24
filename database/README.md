# IDA DB-1A — SQLite core schema v1

팀 공통 DB 구조를 재현·검토하기 위한 7개 핵심 테이블입니다.
실제 이미지, 실험값, Calibration, 계측/AI 판정, Dashboard 연결은 포함하지 않습니다.

## 실행

프로젝트 루트에서 Python 3.11+ 표준 라이브러리만 사용합니다.
SQLite의 내장 JSON 함수 `json_valid`가 필요합니다.
검증 환경: Python 3.11.9 / SQLite 3.45.1.

```powershell
python -B -m unittest discover -s database/tests -v
python -B database/init_db.py
```

기본 출력은 `runtime_data/database/ida.sqlite3`입니다. 경로는 script 위치를 기준으로
계산하므로 팀원 PC의 절대 경로를 하드코딩하지 않습니다.
다른 검토 경로가 필요하면 `--db <path>`를 사용합니다.

`database/`의 네 소스 파일은 공유 대상입니다. 실제 DB와 테스트 임시 파일은
`runtime_data/` 아래에 두고 Git에서 제외합니다.
운영 DB에는 초기 예시 레코드나 Die/Site 자동 seed를 삽입하지 않습니다.

## 테이블과 관계

| 테이블 | 역할 / 연결 |
|---|---|
| process_runs | 공정 실행 / 노광 값과 단위, recipe, 수행 시각 |
| wafers | process_runs 참조 |
| dies | wafers 참조 / 행 1~3, 열 1~8, Die 코드 |
| pattern_sites | dies 참조 / lt, c, rb |
| upload_batches | wafers 참조 / 업로드 모드, Dataset Role |
| acquisition_conditions | 배율·조명·카메라 조건 |
| images | upload_batches + pattern_sites + nullable acquisition_conditions 참조 |

관계:
`process_runs → wafers → dies → pattern_sites → images`
및 `wafers → upload_batches → images`, `acquisition_conditions → images`.

## 무결성 정책

- 모든 TEXT PK는 NOT NULL/빈 문자열 금지이며 ID를 자동 생성·변환하지 않습니다.
- FK는 RESTRICT 정책입니다. 묵시적 ID 변경/연쇄 삭제는 없습니다.
- Wafer 내 Die 코드와 좌표를 각각 UNIQUE로 관리합니다.
- Die 코드는 행·열과 일치해야 합니다: 11~18, 21~28, 31~38.
- Wafer당 허용 가능한 위치 공간은 24개입니다. 단계적 등록을 허용하므로
  테이블 자체가 모든 Wafer에 24행을 자동 생성하거나 최소 개수를 강제하지는 않습니다.
  테스트에서는 24 Die / 72 Site 전체 구성을 검증합니다.
- Site는 lt/c/rb만 허용하며 Die별 중복 위치를 거부합니다.
- 이미지 INSERT/연결 UPDATE에 더해 Batch.wafer_id, Die.wafer_id,
  Site.die_id 변경까지 5개 trigger로 동일 Wafer 조건을 보장합니다.
- Site별 이미지 수 및 Wafer별 이미지 수에는 상한이 없습니다.
- SHA-256은 64자리 hex이며 **비고유 인덱스**입니다. 중복을 삭제·병합하지 않습니다.
- width/height는 양의 정수입니다. condition_id와 repeat_index는 NULL을 허용합니다.
  반복 번호의 운영 규칙은 아직 정하지 않고 정수 여부만 검사합니다.
- 파일명에 결함 코드가 없으면 legacy_defect_code는 NULL로 유지할 수 있습니다.
  정상 판정이나 defect class를 자동 생성하지 않습니다.
- auto_exposure는 NULL(정보 없음), 0(OFF), 1(ON)을 구분합니다.
- magnification은 양수이며 20/40으로 제한하지 않습니다.
- camera_setting_json은 NULL 또는 유효한 JSON 문자열입니다.
- batch_status / metadata_status는 필수 비어 있지 않은 TEXT입니다.
  확정되지 않은 상태 Enum이나 AI/QC 판정 기준은 만들지 않습니다.

## Dataset Role

DB 값은 다음 소문자 코드만 허용합니다.

| DB dataset_role | 향후 UI 어댑터에서 대응할 기존 UI 코드 |
|---|---|
| evaluation (기본값) | EVALUATION |
| training_candidate | TRAINING_CANDIDATE |
| development | DEVELOPMENT |

upload_mode는 `wafer_batch / repeatability / lighting`입니다.
Dataset Role은 Train/Validation/Test split이 아닙니다. 이번 구현은 Dashboard를
수정하거나 실제 코드 변환/DB 저장을 연결하지 않습니다.

## 시각 및 connection

시각은 UTC의 단일 ISO 8601 형식 `YYYY-MM-DDTHH:MM:SSZ`를 사용합니다.
created_at은 해당 형식의 현재 UTC가 기본값입니다.
processed_at/captured_at은 NULL 또는 동일 형식이며 잘못된 날짜를 거부합니다.
다른 timezone/소수초 입력이 있다면 향후 입력 경계에서 이 형식으로 변환해야 합니다.

SQLite의 foreign_keys 설정은 connection마다 필요합니다.
프로젝트의 `connect_database(path, readonly=False)`는 항상 이를 활성화하고
기존 파일만 엽니다. 다른 SQL 클라이언트에서도 `PRAGMA foreign_keys=ON`을 설정하세요.

## 버전과 기존 DB 보존

- 버전은 별도 테이블 없이 `PRAGMA user_version=1`로 관리합니다.
- 새로운 파일만 배타적으로 생성하고 스키마를 transaction 안에서 적용합니다.
- 기존 DB는 **read-only connection**으로 검증합니다. 초기화 재실행 시 INSERT,
  스키마 재적용 또는 데이터 변경을 하지 않습니다.
- user_version과 기준 schema.sql의 테이블·인덱스·trigger·view 정의를 비교합니다.
  SQL 공백 정규화 외에는 엄격히 비교하므로 별도 추가 객체도 호환성 오류입니다.
- SQLite quick_check, foreign_key_check, 기존 이미지의 Wafer 관계도 확인합니다.
- 다른 버전/구조/손상 데이터가 있으면 오류와 exit code 1을 반환합니다.
  자동 Migration/복구/강제 재생성은 없습니다. 기존 DB는 보존합니다.
- 신규 생성 중 오류가 발생한 파일도 자동 삭제하지 않습니다. 원인을 확인해야 합니다.

## 직접 조회

프로젝트 루트에서 아래 명령은 읽기 전용으로 각 테이블의 행 수를 출력합니다.

```powershell
python -B -c "from database.init_db import connect_database, TABLE_NAMES; c=connect_database('runtime_data/database/ida.sqlite3', readonly=True); print({t:c.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in TABLE_NAMES}); c.close()"
```

테이블 목록:

```powershell
python -B -c "from database.init_db import connect_database; c=connect_database('runtime_data/database/ida.sqlite3', readonly=True); print(c.execute('SELECT name FROM sqlite_master WHERE type=? ORDER BY name', ('table',)).fetchall()); c.close()"
```

## 테스트 범위

17개 unittest는 `runtime_data/test_tmp/`에 새 임시 DB를 만들고 해당 임시 디렉터리만
정리합니다. 운영 ida.sqlite3는 읽거나 변경하지 않습니다.
Wafer 1 / Die 24 / Site 72 fixture와 교차 Wafer 거부용 별도 fixture는 테스트 전용입니다.
재실행 보존 및 호환성 오류 테스트는 임시 DB의 SHA-256 불변도 확인합니다.

이미지 업로드·파일명 파싱·dataset 이동·Streamlit 연결·모델 실행은 구현하지 않습니다.

## DB-1B — 임시 PNG 등록 검증

DB-1A의 schema.sql / init_db.py / schema version 1은 변경하지 않습니다.
DB-1B는 표준 라이브러리만 사용하며 운영 DB와 실제 이미지 dataset에 레코드를 추가하지 않습니다.

- filename_parser.py: parse_filename(basename) → ParsedFilename
- image_registry.py: register_image(connection, path, ...) / get_image(connection, image_id)
- tests/test_image_registry.py: 테스트마다 새 임시 DB와 작은 PNG 1~3장 생성

파일명은 w<숫자>_<두 자리 Die>_<lt|c|rb>[_defect].png입니다.
Die는 11~18 / 21~28 / 31~38만 허용합니다. 예를 들어
w1_12_rb_pn.png는 W1 / 12 / 행1 / 열2 / rb / legacy pn으로 파싱합니다.
결함 suffix는 word 문자(Unicode 포함)·밑줄·하이픈을 허용하며 고정 결함 목록은 없습니다.
미표기는 None이며 정상 판정으로 해석하지 않습니다. Wafer 숫자와 legacy code의 원문은 보존합니다.
파서는 basename만 받으며 파일명 오류는 FilenameParseError를 발생시킵니다.

등록 전제:
- 호출자가 초기화된 임시 DB connection과 명시적인 image_id / batch_id /
  physical_site_id / metadata_status를 전달합니다. condition_id는 선택 사항입니다.
- 상위 Process/Wafer/Die/Site/Batch/Condition은 Fixture에서 먼저 생성하고 commit해야 합니다.
- 미완료 caller transaction은 자동 commit/rollback하지 않고 등록 전에 거부합니다.
- filename의 W<숫자>가 해당 Fixture의 wafer_id와 정확히 일치해야 합니다.
- PNG signature / IHDR / IHDR CRC / 양의 크기를 확인하고 전체 원본 bytes의 SHA-256을 구합니다.
  픽셀 디코딩이나 IDAT/IEND 전체 검증은 하지 않는 최소 header 검사입니다.
- 파일은 제자리에서 읽기만 합니다. 테스트 storage_path는 원본의 절대 경로이며
  실제 서비스의 영구 저장 위치/상대 경로 정책을 확정한 것이 아닙니다.

등록은 BEGIN IMMEDIATE → 상위 관계 확인 → SHA 중복 조회 → INSERT → 조회 → commit 순서입니다.
진행 중 오류가 있으면 해당 등록 transaction을 rollback합니다.
성공 반환값은 저장된 이미지 필드와 wafer_id / die_id / die_code / 행·열 / position_code,
그리고 반환 전용 duplicate_sha256 / duplicate_image_ids입니다.
get_image는 저장된 레코드와 계층 identity를 조회하며, 없는 ID는 None을 반환합니다.

같은 SHA-256도 서로 다른 Image ID로 보존합니다. 기존 Image ID 목록으로 중복을 알리고,
원본 파일이나 기존 행을 삭제·병합·덮어쓰지 않습니다.
같은 Image ID를 다시 삽입하면 오류로 처리하고 기존 행을 유지합니다.
운영 runtime_data/database/ida.sqlite3 경로에 대한 register_image 호출은 보호 장치로 거부합니다.

검증 명령 (운영 DB와 실제 이미지 사용 없음):

    python -B -m unittest discover -s database/tests -v

DB-1A 17개 + DB-1B 14개 = 31개 테스트입니다.
INSERT 이후 readback 오류를 발생시켜 실제 rollback과 재시도 가능 여부도 확인합니다.

실제 Batch 등록 전에는 파일명 Wafer 코드와 DB ID 매핑, 상위 레코드 식별자,
영구 storage_path 정책, 촬영 조건/시각/반복 번호, 중복 처리 및 Batch 실패 정책,
운영 DB 등록 승인 범위를 먼저 확정해야 합니다.


## DB-1D — Streamlit 등록 연결

Data Control Center의 데이터 소스 기본값은 Mock입니다. SQLite를 선택하면 기존 두 탭에서
PNG 1~3장 업로드 → 기존 Batch Preview → 수동 Metadata 보완 → 중복 후보별 확인 →
명시적 승인 → 원본 보관 및 Batch 등록 → Batch/이미지 조회를 진행할 수 있습니다.
Mock 분석 결과는 실제 SQLite 등록 결과에 합치지 않습니다.

- 기본 DB: runtime_data/database/ida.sqlite3 (기존 DB-1A로 준비).
- 임시 업로드: runtime_data/uploads_tmp/UUID/UUID/원본파일명.
- 승인 후 보관: runtime_data/raw/UUID/UUID/원본파일명.
- 원본 bytes·파일명·SHA-256을 유지합니다. 동일 파일명은 독립 경로를 사용합니다.
- Batch/Image/새 Process ID는 자동 발급하지만 Wafer 및 실제 노광 조건은 사용자가 확인합니다.
  노광 단위는 미상일 때 NULL이며, 촬영 장비 조건은 임의 생성하지 않습니다.
- 파일명 수정 탭은 Metadata만 보완하며 파일을 rename하지 않습니다.
- 기본 DB-1B/C의 운영 DB 보호는 유지합니다. DB-1D Provider는 명시적 승인과 관리 저장소를
  사용하는 등록 경로에서만 운영 등록을 요청합니다.
- 정상적인 파일 쓰기/DB INSERT/Commit 실패 시 Transaction을 Rollback하고 이번 등록이
  생성한 파일만 정리합니다. 기존 파일을 덮어쓰거나 재귀 삭제하지 않습니다.
- 파일시스템과 SQLite의 분산 Transaction은 아닙니다. 프로세스 강제 종료 후 고아 파일 복구,
  세션 종료 후 임시 파일 보존 기간 및 다중 사용자 운영 정책은 후속 검토가 필요합니다.

격리 검증 시 아래 환경변수 세 개를 모두 임시 경로로 지정합니다. 테스트 코드도 같은 원칙을
사용하며 운영 DB나 연구 이미지에 테스트 데이터를 쓰지 않습니다.

- IDA_DB_PATH: 명시적으로 초기화한 테스트 SQLite 파일
- IDA_STORAGE_ROOT: 테스트 영구 보관 폴더
- IDA_STAGING_ROOT: 테스트 임시 업로드 폴더

database/upload_storage.py는 저장·정리만, 기존 batch_registry.py는 Preview·승인·등록을,
dashboard/sqlite_provider.py는 UI와 DB 사이의 등록·조회 경계를 담당합니다.
자동계측, YOLO, Calibration, Criticality, 분석 결과 테이블 및 Draft 인터페이스 구현은 포함하지 않습니다.

 
## DB-1E — v1 유지 / 명시적 v2 경로

기존 schema.sql 및 init_db.py는 Version 1 전용으로 보존합니다.
schema_v2.sql은 같은 7개 테이블을 유지하며 upload_batches.upload_mode에 single_image만
추가합니다. Dashboard/Provider는 버전을 읽어 호환 경로를 선택하지만 Migration을 실행하지 않습니다.
운영 DB가 v1이면 기존 세 모드는 사용 가능하고 Single Image 등록은 차단됩니다.

새 임시 v2 DB 생성:

    python -m database.migrate_v2 --db <새로운_테스트_DB.sqlite3> --create

기존 v1의 별도 테스트 복사본 Migration:

    python -m database.migrate_v2 --db <테스트_복사본.sqlite3> --backup <새_백업.sqlite3>

도구는 v1 -> v2만 지원하며, 유효한 v2에 재실행하면 읽기 전용 검증 후 종료합니다.
Migration은 쓰기 잠금을 얻고 SQLite backup으로 원본 v1 상태를 보관한 뒤 Batch 테이블을
재구성합니다. PK/FK/모든 레코드/기존 Trigger·Index를 검증하고 실패하면 Transaction을
Rollback합니다. 기존 백업은 덮어쓰지 않으며 같은 v1 상태인 백업만 재시도 시 재사용합니다.

### 운영 적용은 별도 승인 후

이번 개발에서 runtime_data/database/ida.sqlite3는 Migration하지 않았습니다.

1. 사용자/팀의 별도 운영 적용 승인.
2. Dashboard와 다른 DB 쓰기 작업 중지.
3. SQLite backup으로 운영 DB 백업 및 복원 가능성 확인. 원본 저장소도 보존.
4. 백업 복사본에서 Migration·레코드/FK/Trigger/Index 검증.
5. 승인된 운영 경로와 고유 백업 경로를 명시하여 적용:

       python -m database.migrate_v2 --db runtime_data/database/ida.sqlite3 --backup <고유_백업.sqlite3> --allow-operational

6. 버전·레코드·원본 해시 확인 후 Dashboard 재시작 및 네 입력 유형 확인.
   실패 시 운영을 재개하지 않고 v1 백업으로 복원/검토.

### 입력/조회

- Wafer Batch: PNG 여러 장 또는 표준 단일 디스크 ZIP. 72개 촬영 위치의 누락/추가 촬영은 안내이며
  이미지 수를 72장으로 고정하지 않습니다.
- Single Image: v2에서만 PNG 1장, 정식 single_image 저장. 기존 1장짜리 wafer_batch를 변경하지 않습니다.
- Repeatability: 같은 Site, 양의 정수이면서 서로 다른 repeat_index를 확인·수정합니다.
- Lighting: 실제 배율과 조명 설정을 입력합니다. Auto Exposure 미상/0/1을 구분하고 기존 조건 테이블을 사용합니다.
- 일반 등록/노광 확인 체크박스 없이 등록 버튼 클릭을 명시적 승인으로 전달합니다.
  새로운 공정의 미입력 노광값은 임의 기본값으로 채우지 않습니다.
- EXCLUDE / REGISTER_SEPARATELY / HOLD, 최종 대상 재검증, 전체 Rollback은 유지합니다.
- 오류 수정 목록은 기본적으로 Metadata 보완 대상만 표시하며 전체 이미지 보기로 확장할 수 있습니다.
- 전체 실제 이미지는 유형별·5/10/25장 단위로 조회하고, 파일은 사용자가 선택한 한 장만 읽어 표시합니다.
  Batch 표시명은 별칭이며 내부 ID를 변경하지 않습니다. 기존 Batch 상세 경로를 재사용합니다.
- 절대 DB 경로/내부 오류 상세는 기본 숨김입니다. 로컬 관리자만 IDA_SHOW_DB_PATH=1로 명시적으로 활성화할 수 있습니다.

### 업로드 제한과 보안

upload_storage.py의 변경 가능한 상수 기본값:
이미지 최대 512장, 파일당 32 MiB, 전체 512 MiB, ZIP 원본 128 MiB,
ZIP 항목 2,048개, 최대 압축 비율 200, PNG 디코딩 스트림 256 MiB.
ZIP64/분할 ZIP/암호화 ZIP/비PNG 항목/중복 archive 경로/링크는 거부합니다.
중앙 directory 개수도 ZipInfo 대량 할당 전에 검사합니다. extractall은 사용하지 않습니다.
서로 다른 archive 경로의 같은 PNG basename은 UUID 경로로 안전하게 분리하고 출처를 임시 Metadata에 남깁니다.
경로 탈출·CRC·손상 PNG/ZIP·실제 읽은 크기를 검증하며 PNG 재인코딩은 하지 않습니다.
검증 실패 시 운영 DB 등록은 없고 이번 작업이 만든 임시/영구 파일만 정리합니다.

프로세스 강제 종료 후 고아 파일 복구 및 세션 종료 시 보관 기간 정책은 여전히 후속 운영 검토 대상입니다.

## 사용자 지정 Batch 이름 (Schema v1/v2 공통)

- 새 컬럼이나 Migration 없이 기존 upload_batches.notes에 선택 이름을 저장합니다.
- JSON 형식 표식 ida.batch_display_name.v1의 batch_name을 사용하며,
  기존 notes 문자열/NULL은 original_notes에 그대로 보존합니다. 이름 삭제 시 원래 notes를 복원합니다.
  기존 메모를 이름으로 추정하거나 기존 레코드를 자동 변환하지 않습니다.
- 등록 시 기존 Batch Transaction에 notes를 포함합니다. 등록 후 이름 변경은 Batch 상세의
  명시적 저장 버튼으로 notes만 Transaction UPDATE합니다. 내부 Batch ID·자동 번호·이미지는 바뀌지 않습니다.
- 목록은 기존 자동 표시명에 이름을 함께 표시하고 긴 이름만 축약합니다. 전체 이름은 Tooltip/상세에서 확인합니다.
- 운영 DB에 자동 Migration/이름 입력/기존 레코드 변경을 수행하지 않습니다. 이 기능의 검증은 임시 DB에서만 수행합니다.


## TASK 14 — Schema v3 PEB temperature

- v1/v2 SQL은 보존합니다. v3는 `upload_batches.peb_temperature_c REAL NULL` 하나만 추가합니다.
- PEB는 Batch 공통 공정 Metadata입니다. 같은 Wafer의 다른 Batch는 서로 다른 PEB를 가질 수 있습니다.
- 노광은 기존 `wafers.process_run_id → process_runs` 관계와 단위를 유지합니다.
- 과거 Batch의 PEB는 NULL이며 보간·기본 온도·자동 backfill을 하지 않습니다.
- 숫자는 유한값만 허용하며, 시간은 이번 필드와 합치지 않습니다. PEB 시간 입력은 미구현입니다.
- 기존 Batch 이름/notes, PK/FK/Index/Trigger, 7개 테이블의 기존 필드·레코드를 보존합니다.
- Migration은 Dashboard 시작 시 자동 실행하지 않습니다. 모든 DB writer를 중지한 뒤 명시적으로 실행합니다.

```powershell
.\dashboard\.venv\Scripts\python.exe -B -m database.migrate_v3 --db runtime_data/database/ida.sqlite3 --backup runtime_data/database/ida.pre_peb_v3.sqlite3 --allow-operational
```

- 적용 전에 v2 스키마를 확인하고 SQLite backup API로 별도 백업을 생성·검증합니다.
- 단일 Transaction에서 ALTER/버전 변경/전체 기존 필드 비교/무결성 검사를 수행합니다. 실패 시 Rollback합니다.
- 재실행은 v3 검증만 하며 기존 PEB를 변경하지 않습니다. 백업 파일은 자동 덮어쓰지 않습니다.
- 복구가 필요하면 모든 writer를 중지하고 새 등록이 있었는지 확인한 후, 실패 DB도 별도 보존하고 검증된 v2 백업을 원래 경로에 복원합니다.
  v3 적용 후 추가 등록된 데이터가 있다면 과거 백업을 바로 덮어쓰지 말고 먼저 보존·대조해야 합니다.
- `initialize_database_v3(path)`는 새 테스트/개발 DB 생성 API이며 기존 v2 DB를 자동 변경하지 않습니다.
- SQLite 등록·조회는 v1/v2도 계속 지원하되 PEB 입력은 v3에서만 활성화됩니다.
