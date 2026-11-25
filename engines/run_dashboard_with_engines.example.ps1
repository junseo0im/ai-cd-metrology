# 팀원 로컬 실행 예시: 대시보드 + 분석 엔진(결함 r33 로컬 ONNX, 선폭 U-Net v2).
# 프로젝트 루트에서 실행:  powershell -ExecutionPolicy Bypass -File engines\run_dashboard_with_engines.example.ps1
# 준비물은 engines\README.md 참고 (분석 환경, runtime_data\config 2개, 가중치 2개, 개인 dev DB).
# 운영 저장·삭제 스위치(IDA_ANALYSIS_OPERATIONAL_SAVE, IDA_DATASET_DELETE_ENABLED)는 설정하지 않습니다 = 꺼짐.
$root = (Get-Location).Path
$dev  = "$root\dev_data"                                    # 개인 데이터 폴더 (Git 제외)

$env:IDA_DB_PATH              = "$dev\dev.sqlite3"
$env:IDA_STORAGE_ROOT         = "$dev\raw"
$env:IDA_STAGING_ROOT         = "$dev\uploads_tmp"
$env:IDA_ANALYSIS_OUTPUT_ROOT = "$dev\results"
$env:IDA_RECORDS_DIR          = "$dev\records"
$env:IDA_DELETION_LOG_ROOT    = "$dev\deletion_logs"
$env:IDA_DELETION_QUARANTINE  = "$dev\quarantine"
Remove-Item Env:IDA_ANALYSIS_OPERATIONAL_SAVE, Env:IDA_DATASET_DELETE_ENABLED -ErrorAction SilentlyContinue

# 분석 서브프로세스 Python과 두 엔진 폴더 (서브프로세스는 이 환경변수를 그대로 물려받습니다)
$env:IDA_ANALYSIS_PYTHON = "$root\engines\.venv\Scripts\python.exe"
$env:IDA_TEAM_DEFECT_DIR = "$root\engines\defect\defect_adapter_r33"
$env:IDA_TEAM_UNET_DIR   = "$root\engines\linewidth\unet_linewidth_metrology_v2"

Set-Location "$root\dashboard"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
