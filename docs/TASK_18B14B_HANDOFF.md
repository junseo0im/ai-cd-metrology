# TASK 18-B-14B 인수인계 — Wafer Map 위험도 기준 v0.1 및 UI 통합 개편

작성: 2026-11-24 · 이전 세션이 사용량 한도로 중단된 시점 기준으로 작성, 아래 '최종 상태'에서 마무리 결과를 덧붙임.

## 최종 상태 (이어진 세션에서 마무리)

- 코드 추가 수정 없음(중단 시점에 구현 완료 상태였음).
- 최종 코드 focused test 재실행: 68개 중 실패 5개 = 이전 TASK부터 있던 운영 데이터 원인 실패
  (test_die_assessment 08, test_ia_analysis_result 05/09/10, test_w5_dashboard_results 06). 이번 변경 관련 테스트 전부 통과.
- 임시 서버(운영 DB 읽기 전용 사본, 확인 후 삭제)와 운영 8502에서 Light/Dark 브라우저 확인 통과.
- 이전 세션 종료로 8502가 내려가 있었음 → 진행 중 작업 없음 확인 후 기존 설정으로 재기동, 반영 완료.
- 운영 DB SHA a1ce1326772775ad… 작업 전후 동일.
- 남은 결정: P/SC를 위험도 점수에 넣을지(현재 미포함, WARNING→NORMAL 16 Die의 원인).

## 1. 완료된 수정사항과 파일

| 파일 | 내용 |
|---|---|
| `dashboard/die_assessment.py` | `RISK_POLICY`, `CD_BOUNDS=(10,20,30)`, `PNPT_BOUNDS=(0,2,5)`, `cd_score()`, `pnpt_score()`, `map_risk()` 추가. 기존 `assess_die()`·PASS/FAIL/PENDING·Row 집계는 변경 없음 |
| `dashboard/components/inspection_wafer.py` | `die_snapshot()`이 AI 잠정 색상만 `map_risk()`로 교체(`legacy_display_risk`·`map_risk` 보존, 공식 판정 view 불변). `risk_text()`, `die_hover()`·`die_status()` 새 기준 표시(위치별 줄도 새 기준 값). `criticality_legend()` → 접힘 Expander '위험도 범례'. I&A Map 확대(410px)·첫 이미지 있는 Die 자동 선택·Map 아래 Die 요약/범례. `render_ia_selected_image()` → 원본+Overlay 2칸. `render_site_images()`(상세) → 이미지 먼저, 선택 컨트롤은 '이미지 · 위치 변경' 접기 메뉴. 기본 위치 = 이미지가 있는 첫 위치. 노광 표시는 `dashboard/units.py` 재사용 |
| `dashboard/components/ia_analysis_result.py` | `render_overlay_slot()` 추가(저장 Overlay만, 재추론 없음, 두 Overlay 동시 표시 시 BOTH_NOTE 유지) |
| `dashboard/pages/inspection_analysis.py` | Wafer/Image 탭 2열 50:50 |
| `dashboard/pages/wafer_map.py` | 범례 중복 호출 제거(Map 카드 내부 접힘 범례 사용), 안내 문구 정리 |
| `dashboard/components/styles.css` | 카드 최소 높이, 이미지 확대(최대 440px), 카드 폭 900px 이하 세로 배치, Die 요약 스타일 |
| `dashboard/tests/test_map_risk.py` (신규) | 경계값·점수·O/S/PD +1 중복 방지·재촬영 중복 방지·미평가·P/SC 제외·기존 판정 불변 6개 |
| 테스트 기대값 갱신 | `test_ia_selected_image.py`, `test_wafer_browser.py`, `test_ia_analysis_result.py`(test_04), `test_w5_dashboard_results.py`(test_03): 원본+Overlay 2칸, 첫 Die 자동 선택, 이미지 있는 위치 기본 선택 |

## 2. 데이터 확인 결과 (운영 DB 읽기 전용)

- PN/PT는 RF-DETR 클래스가 아니라 계측 판정 라벨(`line_judgments.label`, Line 단위, pn ≤8 µm / pt ≥12 µm). 운영: normal 4366, pn 1044, pt 2.
- 집계: 위치별 이미지 중 최대(재촬영 중복 방지) → lt/c/rb 합. PD 계측 제외 이미지는 PN/PT 산출 불가(부분 표시).
- Die 264개(Wafer×조건 11개 범위) 기존 → 신규:
  기존 NORMAL 0 / MINOR 23 / WARNING 66 / CRITICAL 63 / UNUSABLE 12 / NOT_EVALUATED 100
  신규 NORMAL 39 / MINOR 51 / WARNING 28 / CRITICAL 46 / UNUSABLE 0 / NOT_EVALUATED 100, 변경 125.
- WARNING→NORMAL 16개는 모두 P/SC 검토 때문에 WARNING이던 Die(P/SC는 새 공식에 없음, 검토 표시는 유지) — 사용자 결정 필요.
- O/S/PD Die: UNUSABLE→CRITICAL 7, →WARNING 4, →MINOR 1, 미평가 10 유지.

## 3. 실행한 테스트

- `test_map_risk` 6 PASS.
- 갱신 후 `test_ia_selected_image`+`test_wafer_browser` 18 PASS, test_04/test_03 PASS.
- 1차 묶음 77개 중 기존 데이터 원인 실패 5개(test_die_assessment 08, test_ia_analysis_result 05/09/10, test_w5_dashboard_results 06 — 이전 TASK부터 실패)
- 마지막 '최종 코드 focused test' 재실행은 세션 종료로 결과 없음 → 다음 세션에서 재실행 필요.

## 4. 운영 반영 / DB

- 운영 8502: 이번 TASK 미반영. 이전 세션 종료와 함께 8502 프로세스가 내려가 있음(재시작 필요).
- 운영 DB SHA: a1ce1326772775ad… (작업 전후 동일, 쓰기 없음).

## 5. 다음 순서

1. focused test 재실행(위 7개 모듈).
2. 임시 서버(운영 DB 읽기 전용 사본)에서 브라우저 재확인(I&A, 상세, Light/Dark, 범례, Die 클릭).
3. 진행 중 작업 없음 확인 → 8502 재시작(.venv, IDA_ANALYSIS_OPERATIONAL_SAVE=1, IDA_DATASET_DELETE_ENABLED=1).
4. 8502 화면 확인, DB SHA 비교, 최종 보고.

## 6. 요구사항 체크리스트

| 항목 | 상태 |
|---|---|
| 위험도 계산(10/20/30%, PN/PT, O/S/PD +1, 상한 3) | 구현·단위 테스트 완료 |
| NOT_EVALUATED/UNUSABLE 분리, PASS/FAIL 불변 | 구현 완료 (AI 기준으로는 UNUSABLE 미표시) |
| 메인/상세 동일 함수·색상 일치 | 구현·브라우저 확인 완료 |
| 위험도 범례(접힘, 기준·이유·고지) | 구현·브라우저 확인 완료 |
| I&A 50:50, Map 확대, 이미지 2개 확대 | 구현·브라우저 확인 완료 |
| 미선택/데이터 없음 시 카드 유지 | 구현 완료, 최종 확인 필요 |
| 상세페이지 이미지 우선, 컨트롤 접기 | 구현·브라우저 확인 완료 |
| 최종 focused test | 미완료 |
| 운영 8502 반영 | 미완료 |
