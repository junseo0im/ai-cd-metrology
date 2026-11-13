# IDA 공용 MLflow 기록 모듈 (v0.2)

> **학습 코드 대체용이 아닙니다.** 기존 학습을 그대로 실행하면서 `mlflow_config.py` / `experiment_logger.py`의 로깅 호출만 추가합니다.

## 역할
- `mlflow_config.py`: 팀 공용 저장소와 Experiment, 실행자/데이터셋 버전/Run 이름 형식
- `experiment_logger.py`: Run 관리, Metrics/Parameters/Artifacts, Test 결과는 기존 코드가 실제 평가했을 때만 기록
- `yolo_adapter.py`: Ultralytics `on_fit_epoch_end` callback로 이미 계산된 validation metrics를 기록
- `examples/`: 기존 YOLO/PyTorch 코드에 삽입할 **예시**. 그대로 실행하지 마세요.
- `requirements/base.txt`: 기록 모듈 설치용 패키지. `torch`, `ultralytics`, CUDA를 새로 설치하거나 변경하지 않습니다.

## 1. 설치와 권한
1. DagsHub `mnitocea/IDA-Defect-AI-MLflow` 저장소 협업자로 추가되어 있는지 확인합니다.
2. Kaggle에서는 **Internet ON**, PC에서는 네트워크 연결을 확인합니다.
3. 기존 학습용 가상환경/Notebook에서 `python -m pip install -r requirements/base.txt` (Notebook: `%pip install ...`)를 실행합니다.
4. 개인 DagsHub 계정으로 `dagshub login`하여 인증합니다. **토큰을 코드/Notebook/GitHub에 적지 마세요.** Kaggle이면 Secrets 기능을 사용하세요.
5. 이 폴더 경로를 Python import 검색 경로에 추가합니다. VS Code에서는 프로젝트 폴더에 모듈을 배치, Notebook에서는 `sys.path.insert(0, '/path/to/mlflow_team')`를 사용할 수 있습니다.

## 2. 적용 순서
1. `examples/yolo_integration.py` 또는 `examples/pytorch_integration.py`를 참고합니다.
2. 기존 학습의 architecture/optimizer/데이터/증강/split/epoch/batch/seed/평가 코드는 변경하지 않습니다.
3. 학습 앞에 `RunInfo(...)`, `tracked_run(...)`을 적용합니다.
4. 기존 Epoch에서 계산된 지표만 `log_metrics(...)`로 기록합니다. YOLO는 `attach_yolo_epoch_logger(model)`를 이용할 수 있습니다.
5. 실제 저장된 `best.pt` / `best.pth`가 있다면 **선택적으로** `log_checkpoint(path)`를 호출합니다.
6. 기존 코드가 독립 Test를 평가한 경우에만 `log_test_metrics(...)`를 사용합니다. Validation을 Test로 기록하지 마세요.
7. MLflow UI에서 새 Run, owner, dataset_version, Epoch별 metrics, Artifacts를 확인합니다.

## 3. 매우 중요한 운영 규칙
- **새로운 학습 1회마다 새 Run**을 생성합니다. 기존 8개 historical Run을 덮어쓰지 않습니다.
- Run 이름이 같아도 ID는 다릅니다. 필요하다면 suffix를 지정합니다.
- 같은 데이터셋 이름이라도 split 동일 여부는 별도로 검증해야 합니다. 검증 전에는 `unverified` 그대로 둡니다.
- YOLO 자체 MLflow 연동과 공통 로거를 동시에 활성화하지 마세요(중복/충돌 위험). 예제는 `settings.update({'mlflow': False})`를 사용합니다.
- YOLO callback의 `on_fit_epoch_end`가 마지막 best-model 평가 때 다시 불릴 수 있어 같은 epoch은 무시합니다.
- `log_checkpoint()` 성공은 업로드 요청 성공이며 **다운로드 후 SHA-256 검증까지 수행된 것은 아닙니다** (`uploaded_unverified`).
- 모델 파일과 원본 데이터는 **절대 자동 삭제하지 않습니다**. 업로드 용량·권한·보존 정책 확인 후 업로드하세요.
- 네트워크 실패 시 Epoch 로깅 누락이 생길 수 있습니다. 학습 종료 후 기존 `results.csv` 등을 별도로 보존하고 UI를 확인하세요.
- 예제에 있는 `train_kwargs` 등은 참고용 값입니다. 실제 학습 조건을 예제에 맞춰 변경하면 안 됩니다.

## 4. 접속
- 저장소: https://dagshub.com/mnitocea/IDA-Defect-AI-MLflow
- 공용 MLflow: https://dagshub.com/mnitocea/IDA-Defect-AI-MLflow.mlflow

## 5. 배포 전 검증
- 각 팀원 환경에서 현재 코드/버전과 충돌 없는지 검토
- **모델 학습과 데이터셋 없이 connection smoke test**로 새 Run 생성 확인
- Parameters, Tags, Epoch별 Metrics, 원하는 Artifacts, 인증/권한 확인
- 결과 확인 후 GitHub에 PR로 공유

## 6. GitHub에서 검증된 Commit SHA로 가져오기

기존 학습 저장소를 clone/교체하거나 별도 Git 저장소를 만들 필요가 없습니다.
이 PR에서 검증한 **전체 40자리 Commit SHA**를 복사하여 아래 `COMMIT_SHA`에 넣습니다.
브랜치 이름이나 `main`을 사용하지 마세요. SHA가 고정되면 같은 파일을 다시 받을 수 있습니다.
검증 SHA는 PR의 Commits 및 작업 결과 보고에 명시됩니다.
공통 모듈만 별도 `vendor/mlflow_team_<SHA 앞 12자리>`에 다운로드합니다.

### A. Kaggle / Colab

Kaggle은 Internet ON이 필요합니다. 다음 셀은 **다운로드/import 경로 설정만** 수행하며 학습하지 않습니다.

```python
from pathlib import Path
from urllib.request import Request, urlopen
import os
import re
import sys

REPOSITORY = "junseo0im/ai-cd-metrology"
COMMIT_SHA = "PASTE_VERIFIED_40_CHARACTER_COMMIT_SHA"
if not re.fullmatch(r"[0-9a-fA-F]{40}", COMMIT_SHA):
    raise ValueError("검증된 전체 40자리 Commit SHA를 지정하세요")

COMMON = (Path.cwd() / "vendor" / f"mlflow_team_{COMMIT_SHA[:12]}").resolve()
FILES = [
    "mlflow_config.py", "experiment_logger.py", "yolo_adapter.py",
    "requirements/base.txt", "examples/yolo_integration.py",
    "examples/pytorch_integration.py", "README.md",
]
# 공개 저장소: 토큰 없이 raw URL 사용.
# 비공개 저장소: 읽기 권한이 있는 본인 GitHub 토큰을 Secrets -> 환경변수로 전달.
github_token = os.environ.get("GITHUB_TOKEN")
for name in FILES:
    if github_token:
        url = (f"https://api.github.com/repos/{REPOSITORY}/contents/"
               f"tools/mlflow_team/{name}?ref={COMMIT_SHA}")
        headers = {"Accept": "application/vnd.github.raw+json",
                   "Authorization": "Bearer " + github_token}
    else:
        url = (f"https://raw.githubusercontent.com/{REPOSITORY}/"
               f"{COMMIT_SHA}/tools/mlflow_team/{name}")
        headers = {}
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        content = response.read()
    target = COMMON / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
sys.path.insert(0, str(COMMON))
print("Common modules:", COMMON)
```

선택한 Notebook kernel에 기록용 패키지만 설치합니다.
기존 torch / torchvision / ultralytics / CUDA 버전을 변경하는 명령은 넣지 않습니다.
의존성 충돌이 나오면 기존 학습 환경을 강제로 변경하지 말고 설치 계획을 확인하세요.

```python
import subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "-r", str(COMMON / "requirements" / "base.txt"),
])
```

**DagsHub 인증은 GitHub 인증과 별개**입니다. 팀원이 각자 DagsHub 협업 권한을 받은 뒤 본인 계정으로 로그인합니다.

- 로컬/Colab 터미널에서 `dagshub login`으로 공식 로그인 절차를 수행할 수 있습니다.
- Kaggle은 Secrets의 `DAGSHUB_USER_TOKEN`을 사용합니다. 아래 환경변수 설정은 `dagshub`를 import하기 전에 실행합니다.
- Colab Secrets도 `userdata.get("DAGSHUB_USER_TOKEN")`으로 환경변수에 전달할 수 있습니다.
- 개인 토큰을 코드/URL/출력/Notebook 결과/GitHub에 넣지 마세요.
- 토큰을 반환하는 함수를 Notebook 셀의 마지막 표현식으로 실행하지 마세요.

```python
# Kaggle (해당 플랫폼에서만 실행)
import os
from kaggle_secrets import UserSecretsClient
os.environ["DAGSHUB_USER_TOKEN"] = UserSecretsClient().get_secret("DAGSHUB_USER_TOKEN")

# Colab 대안 (위 Kaggle 코드를 대신하여 해당 플랫폼에서만 실행)
# from google.colab import userdata
# os.environ["DAGSHUB_USER_TOKEN"] = userdata.get("DAGSHUB_USER_TOKEN")
```

기존 모델/데이터/학습 인자를 유지한 채 아래처럼 로깅만 연결합니다.
대문자로 된 값과 `model`, `original_train_kwargs`는 **기존 코드에 이미 있는 실제 값**을 사용합니다.
아래 예시는 그대로 실행하는 학습 스크립트가 아닙니다.

```python
from mlflow_config import RunInfo
from experiment_logger import tracked_run, log_metrics
from yolo_adapter import attach_yolo_epoch_logger

info = RunInfo(
    owner=YOUR_DAGSHUB_USERNAME, environment="colab",  # Kaggle이면 "kaggle"
    model_name=EXISTING_MODEL_NAME, dataset_version=ACTUAL_DATASET_VERSION,
    seed=EXISTING_SEED, task="yolo",
)

# YOLO에만 필요: 이중 Run 방지를 위한 로깅 설정이며 학습 인자는 바꾸지 않습니다.
from ultralytics import settings
settings.update({"mlflow": False})
attach_yolo_epoch_logger(model)
with tracked_run(info, params=EXISTING_SCALAR_PARAMS):
    model.train(**original_train_kwargs)  # 기존 호출/인자를 그대로 사용
```

ResNet/MobileNet은 `task="classification"`과 해당 모델 이름을 사용합니다.
기존 학습 loop를 `tracked_run`으로 감싸고, 이미 계산한 값만
`log_metrics(existing_val_metrics, epoch + 1, prefix="val")`로 기록합니다.
새 metric 계산/평가나 checkpoint 업로드를 자동으로 수행하지 않습니다.
기존 프로젝트의 Test 사용 제한도 그대로 따릅니다.

### B. 로컬 VS Code

1. 기존 프로젝트와 별개로 Python 3.11에서 확인한 공통 모듈입니다.
2. 위 다운로드 코드를 VS Code의 **선택된 Python interpreter**에서 실행합니다.
   `COMMON`은 원한다면 본인 공통 라이브러리 경로로 지정해도 됩니다.
3. `python -m pip install -r "<COMMON 경로>/requirements/base.txt"` 후
   `dagshub login`으로 본인 계정을 인증합니다.
4. 기존 프로젝트 실행 파일 앞에서 공통 폴더만 import 경로에 연결합니다.

```python
from pathlib import Path
import sys
COMMON = Path(r"C:/Users/YOUR_USER/Projects/vendor/mlflow_team_VERIFIED_SHA_PREFIX")
sys.path.insert(0, str(COMMON))
from mlflow_config import RunInfo
from experiment_logger import tracked_run, log_metrics
```

저장소 루트를 import 경로로 사용하는 경우
`from tools.mlflow_team.experiment_logger import tracked_run`도 지원합니다.
별도 `__init__.py` 없이 Python namespace package로 사용할 수 있습니다.
단독 파일 1개만 받지 말고 위 FILES를 모두 받으세요.

**업데이트:** 팀에서 검증한 새 전체 Commit SHA로 다운로드하여 새 폴더를 만들고
`COMMON`만 새 경로로 바꿉니다. 실행 중인 학습을 건드리거나 기존 모듈 폴더를 덮어쓰지 않습니다.
Notebook은 이미 import한 구버전이 남을 수 있으므로 학습이 끝난 뒤 kernel을 재시작하여 새 경로를 적용합니다.

### 공개 / 비공개 저장소

- 현재 팀 GitHub 저장소는 공개이며 Commit SHA를 넣은 raw URL로 토큰 없이 받을 수 있습니다.
- 비공개로 전환되면 raw 익명 다운로드는 404가 날 수 있습니다.
  위 GitHub Contents API 방식과 저장소 Contents 읽기 권한의 본인 토큰을 사용하세요.
- GitHub 토큰은 코드 다운로드 전용이며 DagsHub 서버 로그인에 사용하지 않습니다.
- HTTPS 인증 오류를 해결하려고 SSL 검증을 끄지 마세요.

## 7. 배포 검증 범위와 연결 테스트

- MLflow 3.18.0 / DagsHub 0.7.3, Python 3.11에서 문법·일반 import·namespace import를 확인합니다.
- 공통 모듈의 `RunInfo`, `tracked_run`, `log_metrics`로 모델/데이터 없는 연결 테스트를 수행합니다.
- `dataset_version`은 필수이므로 이 테스트에서만 `not_applicable`을 사용합니다.
- 지정 smoke Run 이름은 호출자 쪽 RunInfo 하위 클래스에서 property를 재정의하여 사용합니다.
  공통 학습 Run 이름 규칙과 기존 training API는 변경하지 않습니다.
- 동일 이름의 기존 Run을 먼저 조회합니다. 이미 있으면 수정/재생성하지 않고 상태와 metric만 확인합니다.
- Kaggle/Colab 실제 학습, YOLO callback 실학습, 대용량 artifact 업로드는 이번 배포 테스트 대상이 아닙니다.
- 실제 연결 테스트 결과와 Run ID, 검증 Commit SHA는 PR 설명 및 작업 결과 보고에서 확인합니다.
