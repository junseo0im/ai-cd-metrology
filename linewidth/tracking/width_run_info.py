"""선폭 측정 Run 설정: 팀 공용 MLflow 모듈(v0.2)을 고치지 않고 선폭 Experiment만 추가한다.

공용 모듈의 EXPERIMENTS에는 아직 선폭 항목이 없다 ("Add width experiment after the
team agrees on its name"). 팀이 정한 이름 IDA_Width_Measurement를 여기서만 지정하고,
공용 모듈에 정식으로 추가되면 이 파일 대신 RunInfo(task="width")를 쓰면 된다.

사용 예:
    import sys; sys.path.insert(0, r"...\\linewidth_tool\\vendor\\mlflow_team_2ac4aa1751de")
    from tracking.width_run_info import WidthRunInfo
    from experiment_logger import tracked_run, log_metrics, log_file
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1] / "vendor" / "mlflow_team_2ac4aa1751de"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from mlflow_config import RunInfo  # noqa: E402

WIDTH_EXPERIMENT = "IDA_Width_Measurement"


@dataclass(frozen=True)
class WidthRunInfo(RunInfo):
    """Measurement runs have no training seed; keep seed=0 unless the method is stochastic."""

    task: str = "width"

    @property
    def experiment(self) -> str:
        return WIDTH_EXPERIMENT
