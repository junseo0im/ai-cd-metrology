import numpy as np
import pytest

from ai_cd_metrology.roi import (
    NormalizedROI,
    crop_normalized_roi,
    normalized_roi_to_pixel_bounds,
)


def test_normalized_roi_converts_to_pixel_crop() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)
    roi = NormalizedROI(x_min=0.10, y_min=0.20, x_max=0.90, y_max=0.80)

    bounds = normalized_roi_to_pixel_bounds(image.shape, roi)
    cropped, crop_bounds = crop_normalized_roi(image, roi)

    assert bounds == (20, 20, 180, 80)
    assert crop_bounds == bounds
    assert cropped.shape == (60, 160)
    np.testing.assert_array_equal(cropped, image[20:80, 20:180])


@pytest.mark.parametrize(
    "values",
    [
        (-0.01, 0.10, 0.90, 0.90),
        (0.10, -0.01, 0.90, 0.90),
        (0.10, 0.10, 1.01, 0.90),
        (0.10, 0.10, 0.90, 1.01),
        (0.50, 0.10, 0.50, 0.90),
        (0.60, 0.10, 0.50, 0.90),
        (0.10, 0.70, 0.90, 0.60),
    ],
)
def test_invalid_normalized_roi_is_rejected(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        NormalizedROI(*values)

