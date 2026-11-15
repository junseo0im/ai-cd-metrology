# IDA Inspection Platform — Streamlit UI Phase 1.1

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

## Windows PowerShell 실행

프로젝트 루트에서 Python 3.11을 기준으로 실행합니다.
기존 학습 환경과 분리하기 위해 dashboard/.venv를 사용합니다. 이미 있으면 생성은 생략합니다.

```powershell
Set-Location C:\Users\cindi\Desktop\ai-cd-metrology
New-Item -ItemType Directory -Force dashboard\.tmp | Out-Null
$env:TEMP = (Resolve-Path dashboard\.tmp).Path
$env:TMP = $env:TEMP
python -m venv dashboard\.venv
.\dashboard\.venv\Scripts\python.exe -m pip install --cache-dir dashboard\.pip-cache -r dashboard\requirements.txt
Set-Location dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

테마 설정을 자동으로 읽도록 실행 시 작업 경로를 dashboard로 사용합니다.
브라우저: http://localhost:8501
종료: 실행한 터미널에서 Ctrl+C. 단일 진입점은 dashboard/app.py입니다.

검증한 주 dependency:
- Streamlit 1.64.0 (st.Page / st.navigation)
- Pandas 3.0.5
- Plotly 6.9.0

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
| 설정 | 미연결 상태, 인증/저장 기능 없음 |

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
- Header는 로고/플랫폼명/현재 페이지, 비활성 검색·알림·DEMO 사용자, 현재 Batch/Wafer를 표시합니다.
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
