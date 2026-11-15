# IDA Dashboard — TASK C 실제 브라우저 자동 탐색·통합 검증

검증일: 2026-11-15 (KST). 중단 이후 기존 결과·스크린샷을 재사용하여 보고서를 완성했다. 앱/UI/DB 코드는 수정하지 않았다.

## 1. 결론 및 집계

실제 Chrome 자동화 성공. 등록된 18개 페이지 모두 탐색했다. 기본 등록·조회 기능과 추가 구현 기능을 보존한 상태이며, 표시·진입 관련 FAIL 6건은 수정하지 않고 남겼다.

| 구분 | 수 |
|---|---:|
| 탐색 페이지 | 18 |
| 검증 항목 | 86 |
| PASS | 71 |
| FAIL | 6 |
| BLOCKED | 1 |
| NOT_IMPLEMENTED | 8 |

집계 단위는 페이지 렌더링 54건(18×3 화면 조건)과 기능·시각 검증 32건이다. 페이지 렌더링 PASS가 그 페이지의 모든 미래 기능 구현 완료를 의미하지 않는다.

## 2. 실제 도구·URL·DB 격리

- Playwright Python 1.65.1 + Google Chrome 157.0.8103.78 (headless 실제 브라우저). AppTest가 아니다.
- 연결형 브라우저 MCP는 빈 inventory였고 Computer Use native pipe도 사용할 수 없어, 이미 설치된 Playwright/Chrome으로 진행했다. 신규 설치 없음.
- 운영 URL http://127.0.0.1:8502 : health와 Main Home만 읽기 전용 확인. 운영 서버는 유지했다.
- 격리 v1 URL http://127.0.0.1:8503 : 새 QA DB 및 별도 raw/staging 저장소.
- 격리 v2 URL http://127.0.0.1:8504 : 기존 schema_v2.sql로 새 빈 QA DB 생성. 기존 DB Migration은 실행하지 않았다.
- 8503/8504 QA 서버는 검증 후 종료했다. 공개 Tunnel, 외부 서비스 이미지 업로드 없음.
- 합성 96×64 PNG와 QA ZIP만 업로드했다. 실제 연구 이미지를 업로드·이동·삭제·변경하지 않았다.
- 격리 최종 결과: v1 Batch 5개/이미지 10개, v2 Batch 1개/이미지 1개. 두 DB 모두 integrity=ok, FK 위반 0.
- 운영 DB 전후: Schema v1 / Batch 10개 / 이미지 23개. DB·등록 원본·기존 코드 SHA-256 동일.

원시 검증 기록(로컬 Git ignore 대상): runtime_data/browser_audit/20261115T060232Z/browser_results.json 및 각 browser_*.log. 초기 선택자 진단은 selector_attempt_*.json으로 분리했다.
초기 자동화 실패(상대 링크, 검색형 Dropdown, Overlay 라벨, 중첩 details 선택)는 QA 선택자를 보정한 후 재검증했으며 제품 오류 6건에 포함하지 않았다.

## 3. 탐색한 전체 페이지와 실제 캡처

1920×1080 Light/Dark 및 1366×768 Dark를 확인했다. 1366×768 Light와 그보다 작은 모바일 폭은 별도 검증하지 않았다. 문서 가로 overflow 검사는 통과했지만 카드 내부 표시 문제는 별도 FAIL로 구분했다.

| 페이지 | Route | Light 1920 | Dark 1920 | Dark 1366 |
|---|---|---|---|---|
| Main Home | / | [캡처](screenshots/taskc_20261115T060232Z/main_home_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/main_home_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/main_home_dark_1366.png) |
| Data Control Center | /data-home | [캡처](screenshots/taskc_20261115T060232Z/data_control_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/data_control_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/data_control_dark_1366.png) |
| Inspection & Analysis | /inspection-home | [캡처](screenshots/taskc_20261115T060232Z/inspection_analysis_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/inspection_analysis_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/inspection_analysis_dark_1366.png) |
| 전체 Dataset | /datasets | [캡처](screenshots/taskc_20261115T060232Z/dataset_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/dataset_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/dataset_dark_1366.png) |
| 모델 성능 | /models | [캡처](screenshots/taskc_20261115T060232Z/model_performance_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/model_performance_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/model_performance_dark_1366.png) |
| 설정 | /platform-settings | [캡처](screenshots/taskc_20261115T060232Z/settings_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/settings_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/settings_dark_1366.png) |
| Calibration | /data-calibration | [캡처](screenshots/taskc_20261115T060232Z/calibration_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/calibration_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/calibration_dark_1366.png) |
| Batch 상세 | /data-batch-detail | [캡처](screenshots/taskc_20261115T060232Z/batch_detail_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/batch_detail_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/batch_detail_dark_1366.png) |
| Wafer Map | /wafer-map | [캡처](screenshots/taskc_20261115T060232Z/wafer_map_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/wafer_map_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/wafer_map_dark_1366.png) |
| 이미지 분석 | /image-analysis | [캡처](screenshots/taskc_20261115T060232Z/image_analysis_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/image_analysis_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/image_analysis_dark_1366.png) |
| 결과 리포트 | /reports | [캡처](screenshots/taskc_20261115T060232Z/results_report_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/results_report_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/results_report_dark_1366.png) |
| Repeatability | /inspection-repeatability | [캡처](screenshots/taskc_20261115T060232Z/repeatability_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/repeatability_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/repeatability_dark_1366.png) |
| Lighting | /inspection-lighting | [캡처](screenshots/taskc_20261115T060232Z/lighting_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/lighting_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/lighting_dark_1366.png) |
| 실험 환경 및 조건 | /experiment-environment | [캡처](screenshots/taskc_20261115T060232Z/experiment_environment_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/experiment_environment_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/experiment_environment_dark_1366.png) |
| Simulation | /simulation-preview | [캡처](screenshots/taskc_20261115T060232Z/simulation_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/simulation_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/simulation_dark_1366.png) |
| Critical 기준 | /critical-criteria | [캡처](screenshots/taskc_20261115T060232Z/critical_criteria_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/critical_criteria_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/critical_criteria_dark_1366.png) |
| AI 활용 | /ai-usage | [캡처](screenshots/taskc_20261115T060232Z/ai_usage_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/ai_usage_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/ai_usage_dark_1366.png) |
| 모델 선정 기준 | /model-selection-criteria | [캡처](screenshots/taskc_20261115T060232Z/model_selection_light_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/model_selection_dark_1920.png) | [캡처](screenshots/taskc_20261115T060232Z/model_selection_dark_1366.png) |

요약 시트: [Light 전체](screenshots/taskc_20261115T060232Z/overview_light.png) / [Dark 전체](screenshots/taskc_20261115T060232Z/overview_dark.png) / [작은 화면 전체](screenshots/taskc_20261115T060232Z/overview_small.png).

상세 캡처 폴더: docs/qa/screenshots/taskc_20261115T060232Z/. 초기 선택자 진단 캡처도 보존했으므로 failure_ 파일명만으로 제품 오류를 판단하지 말고 아래 문제 표를 기준으로 한다.

## 4. 기능별 결과

| 검증 항목 | 결과 | 실제 확인 내용 | 증거 |
|---|---|---|---|
| Mock/SQLite 공통 2×2 | PASS | 동일 카드 키·배치·크기와 두 입력 탭 확인. | [캡처1](screenshots/taskc_20261115T060232Z/dcc_mock_light_final.png) / [캡처2](screenshots/taskc_20261115T060232Z/dcc_sqlite_v1_empty.png) |
| v1 Single Image 제한 | PASS | Schema v1의 등록 비활성·v2 필요 안내는 정상 정책. | [캡처1](screenshots/taskc_20261115T060232Z/single_v1_blocked.png) |
| 다중 PNG·파일명 수동 보완·등록 | PASS | 합성 PNG 3장 중 unknown.png를 수동 보완한 후 버튼 한 번으로 3장 등록. | [캡처1](screenshots/taskc_20261115T060232Z/wafer_registered.png) |
| SHA-256·EXCLUDE/HOLD/별도 등록 | PASS | EXCLUDE 기본, HOLD 차단, REGISTER_SEPARATELY 1장 등록. 운영 데이터는 사용하지 않음. | [캡처1](screenshots/taskc_20261115T060232Z/duplicate_registered_separately.png) |
| ZIP 내부 PNG 등록 | PASS | ZIP의 PNG 2장 등록; ZIP 자체는 images 레코드로 저장되지 않음. | [캡처1](screenshots/taskc_20261115T060232Z/zip_registered.png) |
| 경로 탈출 ZIP 차단 | PASS | ../outside.png를 가진 ZIP 거부. 등록 건수 동일. | [캡처1](screenshots/taskc_20261115T060232Z/unsafe_zip_blocked.png) |
| 동일 위치 반복 번호 등록 | PASS | 동일 Site 합성 PNG 2장, repeat_index 1/2. | [캡처1](screenshots/taskc_20261115T060232Z/repeatability_registered.png) |
| 배율·조명 조건 등록 | PASS | 합성 PNG 2장, 입력한 배율·lg/lm 조명 조건으로 등록. | [캡처1](screenshots/taskc_20261115T060232Z/lighting_registered.png) |
| 네 카드 확대·복원 | PASS | 선택 카드 전체 너비·자동 높이, 복원 시 2×2. 다른 카드도 유지. | [캡처1](screenshots/taskc_20261115T060232Z/cards_restored.png) |
| v2 Single Image 등록·조회·중복 차단 | PASS | 신규 QA v2 DB에 PNG 1장 등록·조회. 다시 올리면 중복 감지 및 기본 EXCLUDE 차단. | [캡처1](screenshots/taskc_20261115T060232Z/single_v2_registered.png) / [캡처2](screenshots/taskc_20261115T060232Z/single_v2_duplicate_blocked.png) |
| 콜드 직접 URL 진입 모달 | FAIL | QA 신규 서버 첫 /data-home 직접 접속에서 Page not found 모달. 루트에서 Sidebar로 진입하면 정상. 근본 원인은 미확정. | [캡처1](screenshots/taskc_20261115T060232Z/failure_single_v2_full.png) |
| 유형 필터·이미지 페이지네이션·원본·상세 이동 | PASS | Lighting 2 / Repeatability 2 / Wafer Batch 6 조회, 이미지 페이지 2, 원본·Metadata 및 기존 SQLite Batch 상세 확인. | [캡처1](screenshots/taskc_20261115T060232Z/sqlite_batch_detail.png) |
| 환경 6개 / Critical 2개 / AI 2개 탭 | PASS | 각 탭을 실제 클릭하고 캡처. Placeholder를 실제 실험 결과로 판단하지 않음. | [캡처1](screenshots/taskc_20261115T060232Z/experiment_environment_tab_1.png) / [캡처2](screenshots/taskc_20261115T060232Z/experiment_environment_tab_2.png) |
| Calibration·MLflow 콘텐츠 재사용 | PASS | Calibration 페이지/환경 탭, I&A MLflow/모델 선정 영역의 콘텐츠 동일. | [캡처1](screenshots/taskc_20261115T060232Z/shared_model_selection.png) |
| Batch Matrix와 Wafer Map 구분 | PASS | Mock Processing Matrix는 이미지별 테이블이며 상태 Donut 1개는 pie chart. 물리 Die Map과 별도. | [캡처1](screenshots/taskc_20261115T060232Z/mock_batch_matrix_and_donut.png) |
| 비활성 등록 버튼 대비 | FAIL | Light 흰 글자/흰 배경 1.00:1, Dark 약 1.08:1. 기능적 비활성화는 정상이나 글자가 보이지 않음. | [캡처1](screenshots/taskc_20261115T060232Z/failure_disabled_register_readability.png) |
| SQLite 전역 데이터 소스 안내 | FAIL | 카드에는 실제 SQLite 수량, 전역 Header/Sidebar에는 MOCK DATA ONLY·DB 미연결·Mock Batch 문구가 남음. | [캡처1](screenshots/taskc_20261115T060232Z/failure_sqlite_global_source_labels.png) |
| 독립 Wafer Map 구조 일관성 | FAIL | I&A는 24 Die×lt/c/rb=72영역, 독립 Wafer Map은 P01~P24와 처리 상태 범례의 이전 Mock 표현. | [캡처1](screenshots/taskc_20261115T060232Z/failure_standalone_wafer_consistency.png) |
| 실제 계측·YOLO 추론 | NOT_IMPLEMENTED | 실제 분석은 미연결. Mock 결과에는 예시 표기. | 원시 JSON/로그 |
| 공식 Calibration | NOT_IMPLEMENTED | 공통 Placeholder 및 예시 metadata만 제공. | 원시 JSON/로그 |
| MLflow·Roboflow 실제 API | NOT_IMPLEMENTED | 미연결 Placeholder, 외부 이미지 업로드 없음. | 원시 JSON/로그 |
| 실제 What-if 계산 | NOT_IMPLEMENTED | 입력/출력 Placeholder, 계산 없음. | 원시 JSON/로그 |
| Batch Donut 전체/계측/결함 전환 | NOT_IMPLEMENTED | Mock의 단일 상태 Donut은 존재하나 전체/자동계측/결함검출 전환은 미구현. | 원시 JSON/로그 |
| SQLite 처리 Matrix·분석 상세 연결 | NOT_IMPLEMENTED | SQLite 이미지 원본/Metadata 조회는 존재. 분석 처리 Matrix와 분석 상세 연계는 미연결. | 원시 JSON/로그 |
| 실제 Wafer 위치별 결과 | NOT_IMPLEMENTED | 실제 계측·결함 결과를 물리 Wafer에 연결하지 않음. | 원시 JSON/로그 |
| 승인된 장비·Critical·모델 기준 | NOT_IMPLEMENTED | 작성 공간/탭만 존재. 승인되지 않은 기준값을 생성하지 않음. | 원시 JSON/로그 |
| 부분 INSERT 실패 후 Rollback | BLOCKED | 강제 중간 INSERT 오류를 주입하지 않았음. UI 차단의 무쓰기 결과와 atomic Rollback 증명을 구분. 과거 단위테스트로 대체하지 않음. | 원시 JSON/로그 |
| Batch 목록 글자 겹침 | FAIL | Wafer/Role·이미지/노광·일시/상태의 두 줄 텍스트 및 다음 Batch와 실제 겹침. | [캡처1](screenshots/taskc_20261115T060232Z/failure_batch_metadata_readability.png) |
| 미등록 입력·Role·HOLD·이미지 선택 보존 | PASS | 미등록 Lighting 배율 20, qa_lg/qa_lm, DEVELOPMENT, HOLD 및 조회 이미지 선택을 네 카드 확대·복원 중 보존. 추가 INSERT 없음. | [캡처1](screenshots/taskc_20261115T060232Z/draft_expanded_input.png) / [캡처2](screenshots/taskc_20261115T060232Z/draft_expanded_batches.png) |
| I&A 24 Die·72영역 선택·해제·초기화·Overlay | PASS | 22/lt 클릭·재클릭 해제, 다른 위치 및 초기화, Overlay 토글 확인. | [캡처1](screenshots/taskc_20261115T060232Z/ia_map_selected_22_lt.png) / [캡처2](screenshots/taskc_20261115T060232Z/ia_map_reset_overlay.png) |
| 입력창 경계 대비 | FAIL | Light 입력 래퍼의 흰 테두리와 흰 배경이 같음. 기존 data-baseweb=input CSS 선택자 매칭 0. | [캡처1](screenshots/taskc_20261115T060232Z/failure_input_field_boundary.png) |
| Main Home 카드·환경 페이지 왕복 | PASS | Data Control/I&A 카드, 실험 환경 상세 진입 및 Main Home 복귀. | [캡처1](screenshots/taskc_20261115T060232Z/main_home_light_1920.png) |

## 5. P0/P1/P2 문제 목록 — 이번 작업에서는 수정하지 않음

P0 발견 없음. P1 2건, P2 4건. 자료 보존 범위에서 확인한 결과이며 모든 잠재 오류가 없다는 의미는 아니다.

| ID | 우선순위 | 예상 / 실제 | 증거 | 최소 검토 대상 |
|---|---|---|---|---|
| UI-01 | P1 | 비활성 등록 버튼도 글자가 읽혀야 함 / Light 흰 글자·흰 배경(1.00:1), Dark 약 1.08:1로 빈 버튼처럼 보임 | [Light](screenshots/taskc_20261115T060232Z/disabled_register_light.png) / [Dark](screenshots/taskc_20261115T060232Z/disabled_register_dark.png) | dashboard/components/styles.css의 비활성 primary 색상 |
| UI-02 | P1 | Batch별 두 줄 Metadata가 분리되어야 함 / 실제 글자·다음 행과 겹침. 예: W1와 EVALUATION 텍스트가 약 15px 중첩 | [겹침 측정](screenshots/taskc_20261115T060232Z/batch_metadata_overlap_measured.png) / [전체 이미지 조회와 함께 표시된 상태](screenshots/taskc_20261115T060232Z/failure_real_query_and_batch_detail.png) | TASK 5: dashboard/components/tables.py 및 Batch 범위 CSS |
| UI-03 | P2 | 입력칸 경계 식별 / Light의 입력 래퍼가 흰 테두리·흰 배경. data-baseweb=input 스타일 선택자 매칭 0 | [입력 경계](screenshots/taskc_20261115T060232Z/native_input_border_light.png) | dashboard/pages/data_control.py의 입력 전용 CSS |
| UI-04 | P2 | SQLite 전역 안내 일치 / 실제 수량을 표시하는 카드와 달리 Header/Sidebar는 MOCK DATA ONLY·DB 미연결·Mock Batch 맥락 | [SQLite 전역 문구](screenshots/taskc_20261115T060232Z/sqlite_source_global_labels.png) | dashboard/components/layout.py, dashboard/components/sidebar.py 및 데이터 소스 맥락 전달 |
| UI-05 | P2 | 두 Wafer 화면의 Die/위치 표현 일치 / I&A는 24 Die·72영역, 독립 Wafer Map은 P01~P24와 이전 처리 상태 범례 | [독립 Map](screenshots/taskc_20261115T060232Z/standalone_wafer_legacy_structure.png) / [I&A 72영역](screenshots/taskc_20261115T060232Z/ia_map_selected_22_lt.png) | dashboard/pages/wafer_map.py 및 기존 Map 컴포넌트 재사용 범위 |
| UI-06 | P2 | 등록된 직접 URL로 진입 / 새 QA 서버 첫 /data-home 요청에서 Page not found 모달. 루트→Sidebar는 정상 | [콜드 진입 모달](screenshots/taskc_20261115T060232Z/failure_single_v2_full.png) | dashboard/app.py, dashboard/navigation.py의 초기 라우팅 조사. 근본 원인은 미확정 |

UI-02의 DOM 좌표 검사는 캡션을 포함한 모든 p 요소를 대상으로 재측정했으며, 실제 캡처와 함께 교차 확인했다. UI-01의 비활성 조건 자체는 올바르게 작동한다. UI-04는 전역 문구/맥락 문제이며 SQLite 카드에 Mock 78장 등의 수치를 섞은 것은 발견되지 않았다.

## 6. 기존 Mock 및 추가 구현 기능 보존

- Mock의 네 카드·공통 Header/Sidebar·테마·탭과 예시 데이터는 유지되었다.
- SQLite는 실제 등록 Metadata와 미연결 분석 상태를 표시한다. registered를 분석 완료로 판정하지 않는다.
- PNG/다중 PNG/ZIP, 파싱·수동 Metadata, 중복 정책, 새 입력 유형으로 전환한 뒤 완료 메시지 정리, 이미지 필터·페이지 조회, 기존 Batch 상세 이동이 실제 브라우저에서 통과했다.
- v1 Single Image 차단은 정상 정책이다. v2 신규 QA DB의 정식 single_image 등록도 통과했다.
- 미등록 Lighting의 입력값·Role·HOLD·선택 파일을 네 카드 확대/복원 중 유지했다. 등록 버튼을 누르지 않은 이 시나리오의 DB 건수는 변하지 않았다.
- 초기 Mock에 없던 기능을 제거 대상으로 취급하지 않았다. 독립 Map의 이전 표현과 미구현 분석 결과만 별도로 구분했다.

## 7. Batch 상세와 Wafer 상세의 구분

- Mock Batch 상세: Batch 선택, 이미지별 Processing Matrix 테이블, 단일 상태 Donut 및 예시 계측 요약이 있다. Matrix는 3×8 물리 Map이 아니다.
- SQLite Batch 상세: 해당 Batch의 이미지·원본·Metadata 조회가 동작한다. 분석 처리 Matrix, 전체/자동계측/결함검출 전환 Donut 및 분석 상세 연결은 아직 미연결/미구현이다.
- I&A Map은 3행×8열, Die별 lt/c/rb 72개 Mock 영역이다. 독립 Wafer Map 페이지는 이전 24개 Pattern Mock 표현이 남아 일관성 문제가 있다.
- 실제 Wafer 위치별 계측/결함 결과는 어느 화면에서도 이번 검증으로 구현되었다고 판단하지 않았다.

## 8. BLOCKED 및 미구현/미검증 범위

- BLOCKED 1건: 브라우저에서 부분 INSERT 이후 실패를 강제하고 atomic Rollback을 관찰하는 fault-injection은 수행하지 않았다. 코드/Schema를 변경하지 않았고, UI 차단·중복 EXCLUDE/HOLD의 무쓰기 확인과 혼동하지 않았다.
- NOT_IMPLEMENTED 8건은 위 기능 표에 명시했다: 실제 계측·YOLO, 공식 Calibration, 실제 MLflow/Roboflow, 실제 Simulation, Donut 전환, SQLite 분석 Matrix/분석 상세 연계, 실제 Wafer 결과, 승인된 기준 내용.
- 페이지 렌더링 미검증 페이지는 없다(18/18). 다만 모든 버튼 조합·모든 파일 형식 변형·장시간 동시 등록을 완전 탐색한 것은 아니다.
- 작은 화면 Light 전체 행렬, 모바일 폭, 실제 72장 연구 이미지 Batch 업로드는 수행하지 않았다.
- 이전 DB 79개 + Dashboard 78개 단위/AppTest 결과는 참고 이력일 뿐, 이번 실제 브라우저 검증의 증거로 대체하지 않았다. 재개 시 이미 완료된 탐색을 반복하지 않았다.

## 9. TASK 5 후속 수정 대상 — 이번 단계 미구현

사용자 추가 요청에 따라 다음을 TASK 5 대상으로 기록한다.

1. Batch 목록의 페이지네이션을 제거한다.
2. Batch 전체를 하나의 스크롤 가능한 표로 전환한다.
3. UI-02 글자 겹침을 함께 해결한다. 표시명·유형·Wafer·Dataset Role·노광·등록 수량·일시의 행 높이와 정렬을 검증한다.
4. 기존 내부 Batch ID와 상세페이지 선택 전달은 유지한다. DB Schema/등록 로직을 바꿀 사안으로 확대하지 않는다.
5. 대상은 Batch 목록이다. 전체 이미지 조회의 페이지네이션은 별도 기능이며 이번 검증에서 정상 확인했다.

현재 Batch 구현은 페이지당 최대 5개 표이다. QA v1에는 Batch가 5개여서 Batch 2페이지 전환은 실제 브라우저에서 발생시키지 않았다. 이미지 조회의 2페이지 전환은 실제로 확인했다. Batch 페이지네이션의 존재 자체는 현재 버그로 집계하지 않고, 제거·전체 스크롤 전환을 후속 변경 요구로 분리한다.

## 10. 수정 권고 우선순위

1. P1 UI-01: 비활성 primary 버튼 글자 대비를 먼저 보완한다.
2. P1 UI-02: TASK 5의 Batch 전체 스크롤 표 전환과 함께 글자 겹침을 해결한다.
3. P2 UI-03/04: 실제 렌더링 DOM에 맞춘 입력 경계와 SQLite 전역 안내를 정리한다.
4. P2 UI-05: 기존 I&A Die/위치 표현의 재사용 여부를 결정하고 두 Map의 표시를 맞춘다.
5. P2 UI-06: 콜드 직접 URL 진입을 별도로 재현하고 라우팅 초기화 원인을 조사한다. 진단만으로 임의의 라우팅 재작성은 권고하지 않는다.

위 항목은 다음 작업 후보이며 이번 TASK C에서 어떤 수정도 실행하지 않았다. Git Commit/Push/PR/Merge, 운영 Migration, 운영 INSERT/UPDATE/DELETE 없음.

## 11. 보존·종료 확인

- 운영 DB: v1 / Batch 10 / images 23, integrity_check=ok, foreign_key_check=[] 유지.
- 보호 대상 앱·DB·기존 테스트·원본 이미지의 파일 해시 변경/삭제/추가 0건.
- 기존 8502 서버 health=200. 이번에 만든 8503/8504 QA 서버만 종료했다.
- 새 산출물은 이 문서·QA 캡처 및 Git ignore된 runtime_data/browser_audit의 검증 로그/합성 fixture/격리 DB이다.
