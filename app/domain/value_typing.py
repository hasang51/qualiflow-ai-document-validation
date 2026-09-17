"""Generic value-shape typing for industrial certificate fields.

Typing uses label-agnostic date/identifier/standard/classification/product-detail
shapes. It does not encode supplier names or document-specific literals.
"""

from __future__ import annotations

import re
from typing import Any

from app.domain.grade_registry import resolve_grade

_DATE_VALUE_RE = re.compile(
    r"^(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|\d{1,2}[./-]\d{1,2}"
    r")$"
)
_SPEC_LINE_RE = re.compile(r"\b(EN\s*ISO|EN\s+\d{4,}|AWS|ASTM|ISO|DIN|BS)\b", re.IGNORECASE)
_SHORT_CLASSIFICATION_RE = re.compile(r"^[A-Z]\d{1,2}$")
_COMPACT_CLASSIFICATION_RE = re.compile(r"^[A-Z]{1,4}\d{1,3}[A-Z]{0,4}(?:-\d+[A-Z]?)?$", re.IGNORECASE)
_DIMENSION_RE = re.compile(
    r"(?:Ø|DIAM(?:ETER|\.)?|DIA)?\s*(\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?)?\s*(?:mm|cm|m|in)\b)",
    re.IGNORECASE,
)
_DETAIL_SUFFIX_RE = re.compile(
    r"\s*[-–]\s*(?:DIAM|Ø|SIZE)|"
    r"\s+DIAM\.?|"
    r"\s*Ø|"
    r"\s+P\.L\.W\.|"
    r"\s*\(Kg|"
    r"\s+\d+(?:[.,]\d+)?\s*(?:kg|mm)\b|"
    r"\s+(?:spool|reel|coil|package|pack)\b",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"\s*[/,;|]\s*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9#°][A-Za-z0-9#°./\-]*")

IDENTIFIER_DATE_COMPANIONS: dict[str, str] = {
    "order_number": "order_date",
    "certificate_number": "certificate_date",
}
DATE_FIELDS: frozenset[str] = frozenset({"certificate_date", "order_date"})
LIST_FIELDS: frozenset[str] = frozenset({"standards", "classifications"})


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def looks_like_date_value(value: Any) -> bool:
    text = _text(value)
    return bool(text) and bool(_DATE_VALUE_RE.match(text))


def looks_like_identifier_value(value: Any) -> bool:
    text = _text(value)
    if not text or looks_like_date_value(text) or " " in text:
        return False
    canonical = re.sub(r"[^A-Z0-9]", "", text.upper())
    return bool(canonical) and any(ch.isdigit() for ch in canonical) and len(canonical) >= 3


def looks_like_spec_line(value: Any) -> bool:
    return bool(_SPEC_LINE_RE.search(_text(value)))


def _digit_count(value: str) -> int:
    return sum(ch.isdigit() for ch in value)


def looks_like_classification_token(value: Any) -> bool:
    text = _text(value)
    if not text or looks_like_date_value(text) or looks_like_spec_line(text):
        return False
    compact = text.replace(" ", "")
    if " " in text and "/" not in text:
        return False
    if _SHORT_CLASSIFICATION_RE.fullmatch(compact.upper()):
        return True
    if not _COMPACT_CLASSIFICATION_RE.fullmatch(compact):
        return False
    if _digit_count(compact) >= 6:
        return False
    resolution = resolve_grade(text)
    return resolution.status not in {"resolved", "resolved_dual"}


def looks_like_product_detail(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    if looks_like_spec_line(text) or looks_like_classification_token(text):
        return False
    return bool(_DETAIL_SUFFIX_RE.search(text)) or bool(_DIMENSION_RE.search(text))


def split_standard_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend(split_standard_tokens(item))
        return tokens
    text = _text(value)
    if not text:
        return []
    parts = [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
    return parts or [text]


def partition_standard_and_classification(value: Any) -> tuple[list[str], list[str]]:
    standards: list[str] = []
    classifications: list[str] = []
    seen_std: set[str] = set()
    seen_cls: set[str] = set()
    for token in split_standard_tokens(value):
        key = token.casefold()
        if looks_like_classification_token(token) and not looks_like_spec_line(token):
            if key not in seen_cls:
                seen_cls.add(key)
                classifications.append(token)
            continue
        if key not in seen_std:
            seen_std.add(key)
            standards.append(token)
    return standards, classifications


def split_product_identity_and_details(value: Any) -> tuple[str | None, str | None]:
    text = _text(value)
    if not text:
        return None, None
    match = _DETAIL_SUFFIX_RE.search(text)
    if not match:
        return text, None
    identity = text[: match.start()].strip(" -/")
    details = text[match.start() :].strip(" -/")
    return (identity or None), (details or None)


def extract_date_tokens(value: Any) -> list[str]:
    dates: list[str] = []
    for token in _TOKEN_RE.findall(_text(value)):
        if looks_like_date_value(token):
            dates.append(token)
    return dates


def extract_identifier_tokens(value: Any) -> list[str]:
    identifiers: list[str] = []
    for token in _TOKEN_RE.findall(_text(value)):
        cleaned = token.strip(" ,;")
        if looks_like_identifier_value(cleaned):
            identifiers.append(cleaned)
    return identifiers


def split_identifier_and_date(value: Any) -> tuple[str | None, str | None, bool]:
    """Return ``(identifier, date, date_conflict)`` from a mixed labeled span."""

    text = _text(value)
    if not text:
        return None, None, False
    identifiers = extract_identifier_tokens(text)
    dates = extract_date_tokens(text)
    identifier = identifiers[0] if identifiers else None
    if len(dates) > 1:
        return identifier, None, True
    date_value = dates[0] if dates else None
    if identifier is None and looks_like_identifier_value(text.split()[0] if text.split() else ""):
        identifier = text.split()[0].strip(" ,;")
    return identifier, date_value, False


def serialize_dimensions(value: Any) -> str | None:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) or None
    text = _text(value)
    return text or None


def as_dimension_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _text(value)
    return [text] if text else []


__all__ = [
    "DATE_FIELDS",
    "IDENTIFIER_DATE_COMPANIONS",
    "LIST_FIELDS",
    "as_dimension_list",
    "extract_date_tokens",
    "extract_identifier_tokens",
    "looks_like_classification_token",
    "looks_like_date_value",
    "looks_like_identifier_value",
    "looks_like_product_detail",
    "looks_like_spec_line",
    "partition_standard_and_classification",
    "serialize_dimensions",
    "split_identifier_and_date",
    "split_product_identity_and_details",
    "split_standard_tokens",
]
