"""Shared line-width protocol values used by the measurement code.

Native Y rows (25/45/65% of native height) and measurement-type names.
Dashboard package copy: the benchmark CSV helpers live in the team repository
(linewidth/team_protocol.py).
"""

from __future__ import annotations

import re


Y_FRACTIONS = (0.25, 0.45, 0.65)
MEASUREMENT_ORDER = ("line", "gap", "sidewall")
# Measurement types are chosen by name only (the old 1/2/3 number-key menu was
# removed).  The dashboard Metrology module always measures line and gap.
MEASUREMENT_KEY_MAP = {
    "line": "line",
    "gap": "gap",
    "sidewall": "sidewall",
}
DEFAULT_MEASUREMENT_TYPES = ("line", "gap")


def normalize_measurement_type(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return MEASUREMENT_KEY_MAP[key]
    except KeyError as error:
        raise ValueError("measurement type must be line, gap or sidewall") from error


def parse_measurement_types(value=None) -> list[str]:
    """Parse 'line gap', 'line,gap' or a list of names; None means line and gap."""
    if value is None:
        return list(DEFAULT_MEASUREMENT_TYPES)
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(parse_measurement_types(item))
    else:
        raw = [token for token in re.split(r"[\s,;+]+", str(value).strip().lower()) if token]
        if not raw:
            raise ValueError("select at least one of line, gap, sidewall")
        tokens = [normalize_measurement_type(token) for token in raw]
    selected = set(tokens)
    return [kind for kind in MEASUREMENT_ORDER if kind in selected]


def measurement_rows(native_height: int) -> list[tuple[float, int]]:
    if native_height <= 0:
        raise ValueError("native_height must be positive")
    return [
        (fraction, round((native_height - 1) * fraction))
        for fraction in Y_FRACTIONS
    ]


