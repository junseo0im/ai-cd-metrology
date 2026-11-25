"""Display units shared by every Dashboard page. Display only — DB values are never changed.

Exposure (TASK18-B-18): the exposure conditions 50/60/70/80 are TIMES in seconds (s), like PEB.
Every stored exposure unit is NULL (unrecorded) and is shown as 's'. A legacy 'mJ/cm²'-style text
(none stored today) is the same condition and is also shown as 's', so it never forms a separate
condition label. Any other explicitly recorded unit is shown as stored.
PEB stays in seconds (s).
"""
EXPOSURE_DISPLAY_UNIT = "s"
LEGACY_EXPOSURE_UNITS = {"mj/cm²", "mj/cm2", "mj/cm^2", "mj", "mjcm2"}


def _missing(value):
    return value is None or value != value or (isinstance(value, str) and not value.strip())


def _legacy(unit):
    return isinstance(unit, str) and unit.strip().lower().replace(" ", "") in LEGACY_EXPOSURE_UNITS


def exposure_unit_text(unit):
    return EXPOSURE_DISPLAY_UNIT if _missing(unit) or _legacy(unit) else str(unit)


def exposure_display(value, unit=None, *, missing="미확인"):
    """'50 s' (unit not recorded / legacy mJ text) / `missing` when no value."""
    return missing if _missing(value) else f"{float(value):g} {exposure_unit_text(unit)}"


def stored_unit_text(unit):
    """For rows that show the stored field itself: actual DB state + the display rule."""
    if _missing(unit):
        return f"DB 미기록 · 표시 기준 {EXPOSURE_DISPLAY_UNIT}"
    return f"{unit} (표시 기준 {EXPOSURE_DISPLAY_UNIT})" if _legacy(unit) else str(unit)
