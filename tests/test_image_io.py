from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.schemas import ImageRecord


def _image_record(file_path: Path) -> ImageRecord:
    return ImageRecord(
        image_id="synthetic-1",
        file_path=file_path,
        wafer_id="w1",
        die_id="1",
        pattern_position="center",
        capture_region="standard",
    )


def test_load_image_missing_file_raises(tmp_path: Path) -> None:
    image_record = _image_record(tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError):
        load_image(image_record)


def test_load_image_reads_synthetic_png(tmp_path: Path) -> None:
    image_path = tmp_path / "synthetic.png"
    synthetic = np.full((12, 16), 128, dtype=np.uint8)
    cv2.imwrite(str(image_path), synthetic)

    loaded = load_image(_image_record(image_path))

    assert loaded.shape[:2] == synthetic.shape
    assert loaded.dtype == np.uint8
