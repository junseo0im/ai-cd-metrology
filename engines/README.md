# 분석 엔진 (결함 r33 + 선폭 U-Net v2)

대시보드 DCC의 **분석 실행**이 별도 Python 서브프로세스(`inspection.run_dashboard_analysis`)에서 사용하는 엔진입니다.
대시보드는 환경변수 두 개로 이 폴더를 지정하며, 분석 서브프로세스는 그 값을 그대로 물려받습니다.

| 환경변수 | 값 (프로젝트 루트 기준) | 내용 |
|---|---|---|
| `IDA_TEAM_DEFECT_DIR` | `engines\defect\defect_adapter_r33` | 팀원 r33 규칙(`aoi_pipeline.py`, `defect_rules.py`: 원본 그대로) + 로컬 ONNX 추론(`defect_inference.py`) + 대시보드 연동 어댑터(`defect_adapter.py`) |
| `IDA_TEAM_UNET_DIR` | `engines\linewidth\unet_linewidth_metrology_v2` | 팀원 선폭 v2 (원본 그대로) |
| `IDA_ANALYSIS_PYTHON` | 분석 환경의 `python.exe` | 아래 2) |

환경변수를 주지 않으면 코드 기본값(`runtime_data\external_incoming\defect_adapter`, `...\unet_linewidth_metrology`, 이전 엔진)이 쓰입니다.

## 동작 요약 (2026-11-25 검증 기준)

- 결함: 회전 보정(rotate+crop, 0.15°~15°) → 흐림 판정(흐리면 모델 미호출, 이미지 전체 PD) → 로컬 ONNX → r33 규칙.
  박스는 **회전·크롭된 이미지 좌표계**(`DESKEWED_IMAGE`)로 저장되고, 오버레이도 같은 회전·크롭 이미지 위에 그립니다.
- 출처: `MODEL_POSTPROCESSED`(모델), `HEURISTIC`(흐림·저대비 PD), `RULE`(규칙이 추가한 S/PD, S에서 분리된 P).
  팀원 원래 출처는 `team_source`, S에서 분리된 P의 원래 S는 `derived_from_index`로 함께 저장됩니다.
- PD 정책: 모델 PD가 있으면 `MODEL_PD_EXCLUDED`, 모델 PD 없이 흐림·저대비·RULE PD만 있으면 `POLICY_REVIEW_REQUIRED`.
  **PD가 있으면 해당 이미지 전체의 선폭·간격 계측을 실행하지 않습니다.** PD가 없을 때만 선폭 v2를 호출합니다.
- I&A '조건별 결함 개수 비율': S와 그 S에서 분리된 P는 1건(S 기준)으로 셉니다. 독립 검출 P는 기존대로 1건.
  규칙이 추가한 S에서 P가 분리된 경우는 현재 기준(모델 검출만 집계)대로 0건입니다.
- 선폭 입력(원본 이미지·ROI·Calibration)은 결함 쪽 회전·크롭과 무관합니다.

## 준비 (Windows PowerShell, 프로젝트 루트)

대시보드 자체 설치·개인 dev DB는 `dashboard\README.md`의 "팀원 로컬 실행"을 먼저 따릅니다.

```powershell
# 1) Git 밖 설정 2개: 저장소의 사본을 runtime_data\config 로 복사 (분석 코드가 이 경로를 읽음)
New-Item -ItemType Directory -Force runtime_data\config | Out-Null
Copy-Item engines\config\defect_local_model.json, engines\config\calibration_profiles.json runtime_data\config\

# 2) 분석 환경 (대시보드 .venv 와 별도, Python 3.11)
python -m venv engines\.venv
.\engines\.venv\Scripts\python.exe -m pip install -r engines\requirements-analysis.txt
```

### 3) 가중치 2개 (Git 제외, 별도 준비)

| 파일 | 배치 경로 | SHA-256 | 받는 곳 |
|---|---|---|---|
| `weights.onnx` (123,304,434 bytes) | `runtime_data\models\defect\rfdetr_seg_small_v2_t1\weights.onnx` | `858962c228d5eb419e5a58dea699045b030d3e09c796bcd0c3fc960794733e52` | 저장소에 없음(GitHub 100 MB 제한). 담당자(mnitocea)에게 파일로 받습니다. 같은 폴더에 `engines\config\rfdetr_seg_small_v2_t1.manifest.json`을 `manifest.json`으로 복사해 두면 출처 확인에 쓸 수 있습니다. |
| `best_unet_manual.pt` (1,991,379 bytes) | `engines\linewidth\unet_linewidth_metrology_v2\methods\unet\models\` | `58a815efaf910c26db5ba8d875eafdec42e7a764d409a981b94a4abc44fd4c49` | 원격 브랜치 `unet-method` (명령은 그 폴더의 README.md) |

확인: `Get-FileHash <파일> -Algorithm SHA256`. 로컬 ONNX 로더는 시작할 때 SHA-256을 검사하며 다르면 분석이 실패합니다.

### 4) 실행

```powershell
powershell -ExecutionPolicy Bypass -File engines\run_dashboard_with_engines.example.ps1
```

분석을 한 번 실행하면 결과 폴더의 `process.log` 마지막 줄 `ENGINE_MODULES`에 실제로 로드한 엔진 파일 경로가 남습니다
(PD가 있는 이미지만 분석한 경우 선폭 모듈은 호출되지 않아 `null`로 표시됩니다).

## 운영(8502)과의 관계

운영 8502는 같은 내용의 엔진을 `runtime_data\external_incoming\defect_adapter_r33`, `...\unet_linewidth_metrology_v2`에서
실행합니다 (Git 제외 경로, 이 폴더와 파일 내용 동일). 운영 실행 스크립트와 복구 스크립트(`runtime_data\ops\`)는
운영 DB·저장 스위치를 켜는 PC 전용 설정이라 저장소에 올리지 않습니다. 팀원은 위 예시 스크립트를 사용합니다.
