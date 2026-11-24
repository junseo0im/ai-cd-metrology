"""Display units shared by every Dashboard page (TASK18-B-13A.2). Display only — DB values are never changed.

Exposure: the user-confirmed unit is mJ/cm². A row whose unit is NOT recorded (NULL/empty) is shown with it;
an explicitly recorded unit is always shown exactly as stored, so different recorded units never share a label.
PEB stays in seconds (s).
"""
EXPOSURE_DISPLAY_UNIT = "mJ/cm²"


def _missing(value):
    return value is None or value != value or (isinstance(value, str) and not value.strip())


def exposure_unit_text(unit):
    return EXPOSURE_DISPLAY_UNIT if _missing(unit) else str(unit)


def exposure_display(value, unit=None, *, missing="미확인"):
    """'50 mJ/cm²' (unit not recorded) / '50 mJ' (recorded) / `missing` when no value."""
    return missing if _missing(value) else f"{float(value):g} {exposure_unit_text(unit)}"


def stored_unit_text(unit):
    """For rows that show the stored field itself: actual DB state + the display rule."""
    return f"DB 미기록 · 표시 기준 {EXPOSURE_DISPLAY_UNIT}" if _missing(unit) else str(unit)
