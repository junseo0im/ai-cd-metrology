"""Display-only image conversions and overlays for the dashboard."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from ai_cd_metrology.metrology.gradient_canny import GradientCannyMeasurementDiagnostic
from ai_cd_metrology.schemas import MeasurementResult


def image_to_rgb(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Convert a supported OpenCV image to RGB for Streamlit."""

    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def analysis_overlay(
    image: NDArray[np.generic],
    diagnostic: GradientCannyMeasurementDiagnostic,
    result: MeasurementResult,
) -> NDArray[np.uint8]:
    """Draw ROI, local bands and representative E1-E6 x positions."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()

    x0, y0, x1, y1 = diagnostic.roi_bounds
    thickness = max(2, round(max(canvas.shape[:2]) / 900))
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), thickness)
    cv2.putText(
        canvas,
        "ROI",
        (x0 + 8, min(y1 - 8, y0 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 220, 0),
        thickness,
        cv2.LINE_AA,
    )

    for band in diagnostic.local_bands[:-1]:
        y = band.y_bounds_px[1]
        cv2.line(canvas, (x0, y), (x1 - 1, y), (0, 210, 255), thickness)

    edge_colors = (
        (70, 70, 255),
        (70, 210, 255),
        (90, 220, 90),
        (255, 170, 40),
        (220, 90, 220),
        (255, 255, 255),
    )
    for index, ((edge_x, _), color) in enumerate(
        zip(result.edge_coordinates, edge_colors, strict=False), start=1
    ):
        x = round(edge_x)
        cv2.line(canvas, (x, y0), (x, y1 - 1), color, thickness)
        cv2.putText(
            canvas,
            f"E{index}",
            (max(0, x - 14), min(y1 - 8, y0 + 55)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )

    arrow_y = max(y0 + 20, y1 - 22)
    arrow_start = (x0 + 20, arrow_y)
    arrow_end = (min(x1 - 20, x0 + 170), arrow_y)
    if arrow_end[0] > arrow_start[0]:
        cv2.arrowedLine(canvas, arrow_start, arrow_end, (255, 255, 255), thickness)
        cv2.putText(
            canvas,
            "measurement x",
            (arrow_start[0], max(y0 + 15, arrow_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
