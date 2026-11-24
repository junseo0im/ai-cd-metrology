from typing import Any, Protocol

# Defect AI가 반환하는 결함 클래스 코드 후보
DEFECT_CLASSES = ("s", "o", "sc", "p", "pd")

# Metrology 결과로부터 부여할 별도 판정 라벨
WIDTH_LABELS = ("pn", "pt")

# ROI 높이 기준 Y 위치
Y_FRACTIONS = (0.25, 0.45, 0.65)

# 엔진 실행 상태
STATUSES = ("SUCCESS", "FAILED", "SKIPPED")


class DefectEngine(Protocol):
    def predict(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class MetrologyEngine(Protocol):
    def measure(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        ...
