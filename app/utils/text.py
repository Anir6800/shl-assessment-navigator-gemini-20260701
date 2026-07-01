"""Text normalization helpers."""

from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: object) -> str:
    """Collapse repeated whitespace and trim surrounding spaces."""

    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_comma_list(value: object) -> list[str]:
    """Split a comma-delimited string into cleaned list items."""

    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
    text = normalize_whitespace(value)
    if not text:
        return []
    return [part for part in (normalize_whitespace(item) for item in text.split(",")) if part]


def coerce_bool(value: object) -> bool:
    """Interpret common yes/no string values as booleans."""

    if isinstance(value, bool):
        return value
    text = normalize_whitespace(value).lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def join_text(parts: Iterable[object]) -> str:
    """Build a single normalized text field from multiple parts."""

    return normalize_whitespace(" ".join(normalize_whitespace(part) for part in parts if normalize_whitespace(part)))
