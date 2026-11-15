"""Parse acquisition identity only; absent defect codes never imply NORMAL."""
from dataclasses import dataclass
import re


class FilenameParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedFilename:
    wafer_id: str
    die_code: str
    row_index: int
    column_index: int
    position_code: str
    legacy_defect_code: str | None


_PATTERN = re.compile(
    r"[wW](?P<wafer>[0-9]+)_(?P<die>[0-9]{2})_"
    r"(?P<position>lt|c|rb)(?:_(?P<defect>[\w-]+))?\.(?i:png)"
)


def parse_filename(filename: str) -> ParsedFilename:
    """Accept a basename, not a path. Unknown defect-code words are retained.

    Defect suffixes may contain Unicode word characters, underscores and hyphens.
    Wafer digits are retained verbatim (w01 maps to W01, not W1).
    """
    if not isinstance(filename, str) or "/" in filename or "\\" in filename:
        raise FilenameParseError("Expected a PNG basename, not a path")
    match = _PATTERN.fullmatch(filename)
    if match is None:
        raise FilenameParseError("Expected w<wafer>_<die>_<lt|c|rb>[_defect].png")
    die = match["die"]
    row, column = int(die[0]), int(die[1])
    if row not in range(1, 4) or column not in range(1, 9):
        raise FilenameParseError(f"Invalid die code {die}: row 1..3, column 1..8 required")
    return ParsedFilename(
        wafer_id="W" + match["wafer"], die_code=die,
        row_index=row, column_index=column, position_code=match["position"],
        legacy_defect_code=match["defect"],
    )


def validate_metadata(*, wafer_id: str, die_code: str, position_code: str,
                      legacy_defect_code: str | None = None) -> ParsedFilename:
    """Validate manual identity using the same grammar; never rename a file.

    The temporary validation string is not a storage filename. Original names
    are always retained separately by image_registry.
    """
    if not all(isinstance(value, str) and value for value in
               (wafer_id, die_code, position_code)):
        raise FilenameParseError("Wafer, Die and position must be non-empty strings")
    # Validate each field before composing a grammar probe; underscores must
    # not move part of a Die/position field into another filename component.
    if re.fullmatch(r"[wW][0-9]+", wafer_id) is None:
        raise FilenameParseError("Invalid wafer_id: W<digits> required")
    if re.fullmatch(r"[1-3][1-8]", die_code) is None:
        raise FilenameParseError("Invalid die code: row 1..3, column 1..8 required")
    if position_code not in ("lt", "c", "rb"):
        raise FilenameParseError("Invalid position: lt/c/rb required")
    if legacy_defect_code is not None and (
            not isinstance(legacy_defect_code, str) or not legacy_defect_code):
        raise FilenameParseError("Legacy defect code must be a non-empty string or None")
    suffix = "" if legacy_defect_code is None else "_" + legacy_defect_code
    return parse_filename(f"{wafer_id}_{die_code}_{position_code}{suffix}.png")
