"""Local image loading without assumptions about dataset location."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .schemas import ImageRecord


def load_image(
    image_record: ImageRecord,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> NDArray[np.generic]:
    """Load an image referenced by ``image_record``.

    Dataset discovery, Google Drive access, preprocessing, and ROI extraction are
    deliberately outside this function.
    """

    image_path = Path(image_record.file_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), flags)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image

