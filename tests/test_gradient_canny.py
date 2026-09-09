from pathlib import Path

import numpy as np
import pytest

from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord, MetrologyMethod


def test_method_is_gradient_canny() -> None:
    algorithm = GradientCannyMetrology()

    assert algorithm.method == MetrologyMethod.GRADIENT_CANNY


def test_measure_not_implemented_in_phase_a() -> None:
    algorithm = GradientCannyMetrology()
    image = np.zeros((4, 4), dtype=np.uint8)
    image_record = ImageRecord(
        image_id="synthetic-1",
        file_path=Path("unused.png"),
        wafer_id="w1",
        die_id="1",
        pattern_position="center",
        capture_region="standard",
    )

    with pytest.raises(NotImplementedError):
        algorithm.measure(image, image_record)
