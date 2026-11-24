# IDA Inspection Platform — Streamlit UI Phase 1.1

> 현재 모델 성능 페이지의 MLflow 탭은 실제 서버 기록을 읽기 전용으로 조회합니다. 아래 Phase 설명은 당시 구현 이력이며, 최신 범위는 TASK 15-B 절을 확인하세요.


Phase 1.3: I&A 하단은 이미지/Batch/계측·설계 비교의 3열 통합 카드입니다.
공정 그래프/MLflow 탭과 Simulation Placeholder를 추가했습니다.
Data Control의 수량은 Mock catalog에서 집계하며, 누적 이력과 진행률은 UI 예시입니다.
실제 Calibration·설계 비교·오차 계산·MLflow·Simulation은 미연결입니다.
Phase 1.3 한정 검사는 `dashboard/tests/browser_phase13.py`이며 I&A/Data Control 캡처만
`dashboard/.validation/phase13/`에 저장합니다.

Phase 1.2: Main Home의 실험 환경 카드/Placeholder 페이지를 추가했습니다.
I&A Home의 Map만 8행×3열 Die(11–83), 각 Die의 lt/c/rb로 변경했습니다.
임시 Criticality는 별도 Mock fixture이며 판정 로직이 아닙니다. 세 영역은 해당 Die의
기존 예시 이미지를 공유합니다. 실제 위치별 이미지 연결은 미구현입니다.
같은 영역 재클릭 또는 초기화 시 선택을 해제하고 기본 예시 이미지를 표시하며 Overlay 설정은 유지합니다.
별도 Wafer Map 페이지의 기존 3×8 예시는 변경하지 않았습니다.
Phase 1.2 한정 브라우저 검사는 프로젝트 루트에서
`.\dashboard\.venv\Scripts\python.exe dashboard\tests\browser_phase12.py`로 실행합니다.
캡처 4장과 결과는 `dashboard/.validation/phase12/`에 저장합니다.

화면 골격과 페이지 이동만 구현합니다. **실제 데이터·계측·AI 성능 화면이 아닙니다.**
모든 표/차트/수치는 MOCK / 예시로 표시하며, 이미지는 직접 작성한 SVG illustration입니다.
실제 data/, src/, runs/, tools/, transfer/를 읽거나 import하지 않습니다.
DB·YOLO·Roboflow·MLflow 연결, 학습, 추론, 전처리, calibration 실행은 없습니다.

## 팀원 로컬 실행 (Windows PowerShell)

검증 환경: **Python 3.11.9**, 새 가상환경에 `dashboard/requirements.txt`만 설치,
빈 v6 DB · 8501 실행 · 기본 화면 7개 확인 (2026-11-24).
운영 DB·원본·분석 결과는 Git에 없으며(`runtime_data/`는 제외), 아래처럼 각자 폴더를 지정해 실행합니다.

```powershell
# 1) 프로젝트 루트에서 가상환경 + 의존성
python -m venv dashboard\.venv
.\dashboard\.venv\Scripts\python.exe -m pip install -r dashboard\requirements.txt

# 2) 빈 v6 개발용 DB (v3 초기화 → v4 → v5 → v6). 경로는 필수이며 기존 파일은 덮어쓰지 않습니다.
$dev = "$PWD\dev_data"     # 원하는 개인 폴더 (Git에 올리지 마세요)
.\dashboard\.venv\Scripts\python.exe -B -m database.create_dev_db --db "$dev\dev.sqlite3"

# 3) 데이터 경로를 모두 개인 폴더로 지정 (운영 저장·삭제 스위치는 설정하지 않음 = 꺼짐)
$env:IDA_DB_PATH = "$dev\dev.sqlite3"
$env:IDA_STORAGE_ROOT = "$dev\raw"                  # 등록 원본
$env:IDA_STAGING_ROOT = "$dev\uploads_tmp"          # 업로드 임시
$env:IDA_ANALYSIS_OUTPUT_ROOT = "$dev\results"      # 분석 결과
$env:IDA_RECORDS_DIR = "$dev\records"               # 실험 환경 / Critical 기준 기록
$env:IDA_DELETION_LOG_ROOT = "$dev\deletion_logs"
$env:IDA_DELETION_QUARANTINE = "$dev\quarantine"
Remove-Item Env:IDA_ANALYSIS_OPERATIONAL_SAVE, Env:IDA_DATASET_DELETE_ENABLED -ErrorAction SilentlyContinue

# 4) 8501 실행 (테마 설정을 읽도록 dashboard 폴더에서 실행)
Set-Location dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
```

브라우저: http://localhost:8501 · 종료: 실행한 터미널에서 Ctrl+C. 단일 진입점은 dashboard/app.py입니다.

- 비밀 설정(선택): `dashboard/.streamlit/secrets.toml.example`을 같은 폴더의 `secrets.toml`로 복사해 채웁니다
  (ADMIN 해시는 `python -B -m dashboard.auth --setup`으로 생성). Roboflow 앱은
  `dashboard/database/roboflow_app/.env.example` → `.env`. 실제 `secrets.toml`·`.env`는 Git에서 제외됩니다.
- 팀원 PC에서 되지 않는 기능: 분석 실행(모델 환경 `IDA_ANALYSIS_PYTHON`·가중치·`runtime_data/config`가 Git에 없음),
  Roboflow 탭(별도 Gradio 환경·각자의 API 키 필요). 조회·등록·화면 수정·Word 리포트는 동작합니다.

검증한 주 dependency:
- Streamlit 1.64.0 (st.Page / st.navigation / st.components.v2)
- Pandas 3.0.5 · Plotly 6.9.0
- python-docx 1.2.0 · lxml 6.1.3 · matplotlib 3.11.2 (결과 리포트 Word 내보내기)

## 구조

- app.py: 공통 Header + navigation 등록 + 현재 Page 실행
- navigation.py: 라우트/공통 메뉴/수동 이동
- data_provider.py: read-only Mock 제공 계층
- mock_data/platform.json: 모든 예시 record, scenario, 처리 상태
- mock_data/images/: SVG 예시 이미지/Overlay (실제 이미지 아님)
- components/: 공통 layout, sidebar, chart, image, comparison
- pages/: Main, Data Control, I&A, 비교/세부/Placeholder 페이지
- tests/test_dashboard.py: 데이터 관계, 페이지 로딩 및 이동 검사

UI 컴포넌트에서 예시 record를 생성하지 않습니다. 공급 계층을 통해 저장된 Mock만 읽습니다.
사용자 Metadata 입력은 미리보기용 session_state에만 유지하며 catalog/파일/DB에 저장하지 않습니다.

## 페이지

| 화면 | 범위 |
|---|---|
| Main Home | A/B 이동 카드, 공통 바로가기, 예시 시스템 요약 |
| Data Control Center | 2×2 입력/흐름/Batch 목록/확장 Placeholder |
| Batch 상세 | 선택 Batch의 Processing Matrix와 상태 donut |
| Calibration | scale 미등록 metadata 예시 + 상세 기능 Placeholder |
| Inspection & Analysis | 2×2 Mock Map/노광 비교/이미지·Overlay와 Batch 요약/연동 Placeholder |
| Repeatability | 50:50 이미지 비교(R1/R2/R3) / 예시 그래프·identity |
| Lighting | 50:50 이미지 비교(lg/lm/lb) / 예시 그래프·identity |
| Wafer Map | 실제 배치가 아닌 Mock 위치 선택 + 예시 이미지 |
| 이미지 분석 | 예시 원본/Overlay 및 identity + 실행 기능 Placeholder |
| 결과 리포트 | 예시 요약/identity + 발행 기능 Placeholder |
| 전체 Dataset | Mock catalog 모드 필터 (실제 inventory 아님) |
| 모델 성능 | Placeholder, 실제 P/R/mAP/F1 미표시 |
| 설정 | 단일 ADMIN 로그인/로그아웃, GUEST 조회 및 기록표 편집 권한 안내 |

A/B 페이지는 상단 전용 메뉴와 구분선 아래 공통 메뉴를 표시합니다.
현재 페이지는 native page-link 및 sidebar 현재 페이지 영역에서 표시합니다.
Batch ID 테이블 링크로 상세 이동하며 I&A의 세 모드 버튼으로 레이아웃을 수동 전환합니다.
Map point 선택 또는 Image ID selectbox로 예시 이미지 선택을 연결합니다.

## 검증

모든 쓰기 위치는 dashboard 내부입니다. 필요하면 TEMP/TMP도 안쪽으로 지정합니다.

```powershell
$env:TEMP = (Resolve-Path dashboard\.tmp).Path
$env:TMP = $env:TEMP
.\dashboard\.venv\Scripts\python.exe -m compileall dashboard
.\dashboard\.venv\Scripts\python.exe -m unittest discover -s dashboard\tests -v
```

compileall dashboard는 dashboard/.venv도 순회합니다.
빠른 소스 검증만 원하면 `-x "[/\\]\.venv[/\\]"`를 추가할 수 있습니다.
Streamlit AppTest는 실제 browser 대신 서버 측 페이지/위젯 동작을 검사합니다.
브라우저 렌더링 검증 여부는 작업 결과에서 별도로 보고합니다.

## 다음 DB 연결을 위한 계약 (이번에는 구현하지 않음)

현재 예시 identity는 서로 다른 필드로 보존합니다.
- batch_id: 입력/처리 묶음. mode와 생성 시각, acquisition context
- wafer_id: 실제 wafer identity
- image_id: image URI/object key, SHA-256, width/height, 촬영 시각
- physical_site_id: 동일 물리 위치 식별자. wafer/die/pattern_position 및 실제 좌표 정의 필요
- calibration_id: nullable FK. objective, scale_x/y_um_per_px, 단위, 유효 기간, 승인 상태
- measurement_run_id: image와 알고리즘 실행 연결. 방법/version/parameter snapshot/runtime/status
- model_version: versioned model record. artifact/run identity와 평가 데이터 provenance

추가로 exposure, repeat index(R1/R2/R3), lighting condition(lg/lm/lb),
processing stage/history, measurement/defect result, review/quality 상태가 필요합니다.
예시 JSON의 *_um_example은 실제 계측 schema가 아닙니다. 공식 calibration 환산을 하지 않았습니다.
Mock site_row/site_col은 화면용 순서이며 실제 wafer 좌표로 사용하면 안 됩니다.
Defect/measurement와 acquisition/review 상태의 구분, 결과 집계 단위는 DB 연결 전에 합의해야 합니다.

향후 data_provider의 read 인터페이스 뒤에 실제 repository를 연결할 수 있습니다.
이번 단계에서는 DB client, API key, 모델 실행 호출을 추가하지 않았습니다.

## Phase 1.1 디자인/동작 수정

- native Streamlit toolbar(기본 60px)를 숨기지 않습니다. 본문 padding-top을 3.8rem으로 두어
  사용자 Header가 그 아래 정상 흐름에서 시작합니다. 고정 Header를 추가하지 않았습니다.
- Header는 로고/플랫폼명/현재 페이지, 비활성 검색·알림, 현재 GUEST/ADMIN 사용자 메뉴, 현재 Batch/Wafer를 표시합니다.
- Sidebar는 A/B 메뉴 → 현재 페이지 카드 → 구분선 → 공통 메뉴 순서입니다.
- `.streamlit/config.toml`에 Light/Dark native widget 테마를 분리했습니다.
  root config를 생성하지 않기 위해 **dashboard를 cwd로 실행**하는 명령을 권장합니다.
- 우상단 native ⋮ 메뉴의 Light / Dark로 전환합니다. 사용자 CSS는 native `color-scheme`과
  `light-dark()`를 사용하므로 서버 rerun 전에도 전체 사용자 UI가 같은 모드로 바뀝니다.
  Plotly는 `theme="streamlit"`으로 축/범례/Tooltip을 native 테마에 맞춥니다.
- `components/styles.css`는 한 곳에서 관리합니다. 카드/표/입력창을 페이지별 CSS로 덮어쓰지 않습니다.
- 1920×1080에서 A/B의 2×2 카드를 compact 높이로 배치하고 필요한 내용은 카드 내부에서 스크롤합니다.
  1366×768에서는 페이지 스크롤을 허용합니다. 콘텐츠를 잘라 숨기지 않습니다.
- Batch 표 안의 밑줄 친 Batch ID 셀을 클릭하면 선택 ID를 session context에 반영한 뒤
  native `st.switch_page`로 이동합니다. 전체 새로고침이 없어 테마를 유지하며,
  표 아래에 중복 이동 버튼을 만들지 않습니다. 직접 URL의 `batch_id`도 검증 후 반영합니다.
- 모든 Wafer Batch는 원형 외곽 안 3×8 / P01–P24입니다. 이 좌표는 **Mock UI layout**이며
  실제 제작·계측 위치가 아닙니다. 원래 image_id/physical_site_id는 보존했습니다.
- 공정 그래프는 Mock observation에서 이전/현재 평균·표준편차·n을 재집계합니다.
  실제 측정값을 새로 계산하는 기능이 아닙니다.
- I&A Batch 결과 표에 Line/Gap/Edge 예시 평균, class 개수, status를 표시합니다.
  **Line/Edge는 이번 UI용 예시 명칭이며 기존 Method 2 physical definition을 변경/매핑하지 않습니다.**
- R1/R2/R3 및 lg/lm/lb의 50:50 비교를 유지하고 Overlay 비교 선택을 추가했습니다.

브라우저 검증(선택, 기존 Chrome 또는 지정한 Chromium 필요):

```powershell
# 프로젝트 루트에서 dev 의존성만 dashboard 환경에 설치
.\dashboard\.venv\Scripts\python.exe -m pip install --cache-dir dashboard\.pip-cache -r dashboard\requirements-dev.txt
.\dashboard\.venv\Scripts\python.exe dashboard\tests\browser_smoke.py
```

스크린샷/좌표 검사 결과는 dashboard/.validation/screenshots 및 browser_results.json에 저장됩니다.
일반 unit test는 Playwright나 Chrome 없이 실행할 수 있습니다.


## TASK 10 — 실험·Critical 기록 편집

이 두 페이지의 기록은 분석 설정이 아니며 Defect/Metrology/Runner가 읽지 않습니다.
기존 별도 Calibration 상세페이지는 변경하지 않았습니다.

- 저장: `runtime_data/config/experiment_conditions.json`, `critical_rules.json`.
- 처음 없을 때만 기본값으로 생성하며 기존 JSON은 재실행 시 보존합니다.
- 기록 형식 `schema_version=1`은 SQLite Schema 버전과 별개입니다.
- `record_version`과 탭별 revision·수정 시각을 기록합니다. 과거 버전 전체 이력을 보관하는 기능은 아닙니다.
- 수정은 탭별 초안이며 취소는 폐기, 기본값 복원은 초안만 변경합니다. 저장해야 영구 반영됩니다.
- 숫자·상태 검증, 임시 파일 + fsync + os.replace, 문서 잠금과 탭별 변경 토큰으로 저장 실패·동시 수정 시 기존 파일을 보호합니다.
- JSON이 손상되거나 구조가 다르면 자동 덮어쓰기하지 않습니다. 관리자 확인 후 수동 복구가 필요합니다.
- 비정상 종료로 `.json.lock`이 남으면 모든 기록 writer를 중지하고 관리자가 실제 저장 작업이 없는지 확인한 뒤 잠금을 정리합니다.
- `runtime_data/`는 기존 Git 제외 대상입니다. 실제 로컬 기록은 Commit하지 않습니다.

**기록표 편집은 ADMIN 로그인으로 제한됩니다.** GUEST는 로그인 없이 모든 기록을 조회할 수 있습니다.
기존 `IDA_RECORDS_EDIT_ENABLED`는 더 이상 권한을 부여하거나 차단하지 않습니다.
ADMIN 인증 후에는 기록을 수정할 때마다 환경변수 변경이나 서버 재시작이 필요하지 않습니다.

### ADMIN 최초 설정 (로컬 서버 관리자, 한 번)

프로젝트 루트 PowerShell에서 다음을 실행합니다. 사용자명·표시 이름을 입력하고,
12자 이상 비밀번호를 숨김 입력으로 두 번 입력합니다. 기본 비밀번호는 없습니다.

```powershell
# 프로젝트 루트에서
.\dashboard\.venv\Scripts\python.exe -B -m dashboard.auth --setup
cd dashboard
.\.venv\Scripts\python.exe -B -m streamlit run app.py --server.port=8502 --server.address=127.0.0.1
```

- 생성 위치: `dashboard/.streamlit/secrets.toml`. 기존 ADMIN 설정은 자동 덮어쓰지 않습니다.
  다른 secrets 섹션은 보존합니다. 최초 설정 후 실행 중인 서버는 한 번 재시작하세요.
- `[ida_admin]`의 `username`, `password_hash`, `display_name`을 읽습니다.
  평문 대신 PBKDF2-HMAC-SHA256(600,000회·무작위 salt) 해시만 저장하며 비밀번호·해시는 출력하지 않습니다.
  이 파일은 기존 `.gitignore`로 제외되어 있습니다. 파일 접근 권한은 서버 관리자에게만 부여하세요.
- 배포 환경에서는 `IDA_ADMIN_USERNAME`, `IDA_ADMIN_PASSWORD_HASH`, `IDA_ADMIN_DISPLAY_NAME`
  서버 설정도 지원합니다. 환경변수를 사용하면 사용자명과 해시를 모두 설정해야 하며,
  부분/잘못된 설정은 로그인 불가로 처리합니다. 평문 비밀번호 환경변수는 지원하지 않습니다.
- 계정 미설정 시 GUEST 조회는 정상 동작하고 관리자 로그인 안내만 표시합니다.
- 상단 GUEST → 관리자 로그인 또는 설정 페이지에서 로그인합니다.
  로그인 후 상단에는 설정된 표시 이름이 나타나며 로그아웃하면 즉시 조회 전용이 됩니다.
- ADMIN 권한은 서버 메모리의 무작위 세션 토큰과 실제 Streamlit 세션 ID를 대조합니다.
  `role`/사용자명/URL 값만으로 권한을 얻을 수 없습니다. 다른 브라우저로 전파되지 않으며,
  새 WebSocket 세션(새로고침 포함)이나 서버 재시작 시 다시 로그인해야 합니다. 최대 유효기간은 8시간입니다.
- 인증 설정 변경 시 기존 권한은 무효화됩니다. 실패 5회/60초는 계정 전체에 30초 제한을 적용합니다.
  이 제한과 로그인 세션은 서버 프로세스 메모리에만 있으며 서버 재시작 시 초기화됩니다.
- UI와 기록 저장 함수 양쪽에서 ADMIN을 확인하고, 로그아웃/권한 만료 시 미저장 편집 초안을 폐기합니다.
  MEMBER·회원가입·OAuth·사용자 DB는 구현하지 않았습니다.
- 이 권한은 실험/Critical 기록표에만 적용합니다. 기존 이미지 등록 등 다른 기능의 권한 정책은 유지합니다.
  로그인은 두 기록표를 편집할 권한이며 대시보드 전체에 대한 접근 제한을 제공하지 않습니다.

비밀번호/해시는 README·로그·Git·기록 JSON에 복사하지 마세요.
공개 접속 시 HTTPS를 사용하고 secrets 파일은 서버에만 보관합니다.
검증 근거: [Python PBKDF2](https://docs.python.org/3/library/hashlib.html#hashlib.pbkdf2_hmac),
[상수 시간 비교](https://docs.python.org/3/library/hmac.html#hmac.compare_digest),
[Streamlit secrets](https://docs.streamlit.io/develop/api-reference/connections/st.secrets).

`IDA_RECORDS_DIR`로 기록 디렉터리를 별도 지정할 수 있습니다. QA는 임시 경로를 사용하고 운영 기록을 편집하지 않습니다.
실제 Calibration 값·장비 모델명·제작 실측값은 제공되기 전까지 미입력이며,
Row 예외 조건·PD 참고 기준·최종 라벨 우선순위 등은 검토가 필요합니다.


## TASK 14 — 노광/PEB 공정 조건 조회

- Data Entry의 선택 PEB Temperature(°C)는 Batch와 함께 SQLite에 저장됩니다. 과거 값은 미기록입니다.
- I&A는 왼쪽 Wafer Map/선택 이미지·Batch 결과, 오른쪽 전체 공정 분석 패널로 구성됩니다.
- 공정 패널은 Wafer Map 데이터 소스와 일치하며, 범위 기본값은 전체 등록 이미지입니다.
- 노광 탭은 PEB 조건으로, PEB 탭은 기록된 노광값·단위 조합으로 필터합니다. Wafer/Batch 범위는 별도로 선택합니다.
- 다른 PEB/노광을 합친 전체 집계는 혼합 조건 안내를 표시합니다. 미입력 PEB는 별도 상태이며 0°C로 대체하지 않습니다.
- 현재 결과 Repository·Metrology·Defect AI·Calibration은 미연결입니다. SQLite 선폭·오차·표준편차·결함 수·Fail Rate는 미분석입니다.
- 기존 Mock 평균/결함 예시는 명시적으로 예시 표시하며 실제 분석 결과와 혼합하지 않습니다. 새 가상 계측값은 만들지 않습니다.
- `dashboard/process_analysis.py`의 내부 계산 경계는 향후 검증된 µm 결과만 받습니다. 이 입력은 미연결 초안이며 공통 Contract를 변경하지 않습니다.
  촬영 배율/유효 Calibration ID·배율/Calibration 적용 여부/계측 성공/PD 확인을 검증합니다.
  px 결과는 upstream의 검증된 Calibration 변환이 필요하며 여기서 자동 변환하지 않습니다. µm 결과는 다시 스케일링하지 않습니다.
- 유효 Finger 선폭 → 이미지 평균 → 유효 이미지 평균들의 산술평균·표본 표준편차 순서입니다.
  PD 제외는 이미지별이며 같은 Die의 다른 유효 이미지를 임의 제거하지 않습니다.
- 유효 이미지 0개는 데이터 없음, 1개는 표준편차 산출 불가입니다. 설계 선폭 10 µm 대비 Signed/Absolute Error를 정의합니다.
- 기록표의 Calibration/설계값을 분석 설정으로 자동 연결하지 않습니다. Calibration 입력 UI도 추가하지 않습니다.
- Die/Row Fail 영역은 B 방식 기준과 미평가 상태만 표시합니다. 반복 이미지의 공식 선정 기준과 실제 결과 연결은 후속 작업입니다.
  Wafer 자체 최종 PASS/FAIL 기준은 미확정이며, PENDING을 임의 FAIL/PASS로 바꾸지 않습니다.
- I&A에서 제거한 MLflow 탭/계측·설계 비교 함수는 공유 코드에서 삭제하지 않습니다.


## TASK 15-B — 모델 성능 / MLflow 읽기 전용 조회

- Common Navigation → 모델 성능의 기본 탭은 MLflow입니다. Roboflow 탭은 연결 준비 화면입니다.
- 활성 Experiment 전체 목록, 선택 Experiment의 최신 Run 페이지(25/50/100/200개), 동적 Metric 비교와 Run 상세를 제공합니다.
- 비교 범위는 현재 페이지입니다. 전체 Run 수를 추정하지 않으며, 누락 Metric은 0으로 채우지 않습니다.
- Artifact는 폴더별 목록·경로·확장자·크기만 조회합니다. 다운로드, 가중치 로딩, 모델 학습/추론은 없습니다.
- 기존 공통 MLflow Placeholder는 모델 선정 기준 등 다른 화면에서 그대로 유지합니다.
- Python 표준 라이브러리의 읽기 전용 MLflow REST API를 사용하므로 추가 패키지는 필요 없습니다.
  공식 API: https://mlflow.org/docs/latest/api_reference/rest-api.html
- 기본 서버는 기존 DagsHub https://dagshub.com/mnitocea/IDA-Defect-AI-MLflow.mlflow 입니다.
  Experiment ID/Run/Metric 값은 고정하지 않고 실제 서버에서 조회합니다.
- 서버 환경변수 MLFLOW_TRACKING_URI, MLFLOW_TRACKING_TOKEN(Bearer), 또는
  MLFLOW_TRACKING_USERNAME + MLFLOW_TRACKING_PASSWORD(Basic)를 지원합니다.
  DagsHub Basic 인증은 서버에 등록한 사용자명과 토큰을 password로 설정합니다.
- 대안으로 기존 Streamlit secrets의 [mlflow]에 tracking_uri, token 또는 username, password를 설정할 수 있습니다.
  환경변수가 우선하며 인증정보는 서버 측에만 보관합니다. 현재 공개 조회에 별도 인증 설정은 필요하지 않습니다.
  인증정보가 들어간 URL은 허용하지 않고, HTTP redirect에 인증 헤더를 전달하지 않습니다.
- GET은 Run 상세/Artifact 목록, POST는 Experiment/Run 검색에만 사용합니다. 생성·수정·삭제 API는 없습니다.
- 조회 캐시는 60초이며 새로 조회는 해당 화면의 조회 세대를 갱신합니다. 실패 시 경고만 표시합니다.
- IDA_Width_Measurement의 stability/success_rate는 측정 행 기준 0–1 비율입니다.
  이미지 검출 성공률로 표현하지 않으며 모델의 우열을 자동 판정하지 않습니다.
- Roboflow 전달 앱은 import/실행하지 않습니다. iframe·Gradio 서버·공개 공유·프록시는 만들지 않습니다.
- 이 화면의 MLflow 기록은 실제 서버 데이터이며 기존 Mock 분석이나 SQLite 운영 계측 결과와 혼합하지 않습니다.

오프라인 테스트 (PowerShell, 프로젝트 루트):

    .\dashboard\.venv\Scripts\python.exe -B -m unittest dashboard.tests.test_mlflow_provider dashboard.tests.test_model_performance -v
