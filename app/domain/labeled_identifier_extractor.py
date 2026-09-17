"""Conservative extraction of explicitly labeled header identity fields from text."""

from __future__ import annotations

import re
from typing import Any

from app.domain.field_mapping_registry import (
    IDENTIFIER_HEADER_FIELDS,
    IDENTITY_HEADER_FIELDS,
    normalize_header,
    resolve_canonical_field,
)
from app.domain.value_typing import (
    DATE_FIELDS,
    IDENTIFIER_DATE_COMPANIONS,
    LIST_FIELDS,
    looks_like_identifier_value,
    split_identifier_and_date,
    split_standard_tokens,
)

LABELED_TRACEABILITY_FIELDS: tuple[str, ...] = (
    "heat_number",
    "batch_number",
    "lot_number",
    "colata_number",
    "cast_number",
    "charge_number",
)

LABELED_IDENTIFIER_CONFIDENCE = 0.70
_MAX_LABEL_TOKENS = 4
_FOLLOWER_SKIP = frozenset({"DATE", "TYPE", "STATUS", "NAME", "DESCRIPTION", "TIME", "TIMESTAMP"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9#°][A-Za-z0-9#°./\-]*")
_DELIMITER_RE = re.compile(r"[:#=]")
_DELIMITER_REQUIRED_LABELS = frozenset(
    {
        "CERT",
        "PO",
        "ORDER",
        "HEAT",
        "BATCH",
        "LOT",
        "COLATA",
        "CAST",
        "CHARGE",
        "PRODUCT",
        "GRADE",
        "MATERIAL",
        "QUALITY",
        "DESCRIPTION",
        "STANDARD",
        "NORM",
        "CERTIFICATE",
        "CERTIFICATO",
        "ORDINE",
        "PRODOTTO",
        "DIAMETER",
        "DIAM",
        "DIA",
    }
)
_IDENTITY_FIELD_SET = frozenset(IDENTITY_HEADER_FIELDS)
_IDENTIFIER_FIELD_SET = frozenset(IDENTIFIER_HEADER_FIELDS)
_TRACEABILITY_FIELD_SET = frozenset(LABELED_TRACEABILITY_FIELDS)
_SUPPLIER_SKIP_KEYS = frozenset({"supplier_name", "document_type", "product_category"})


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _field_empty(payload: dict[str, Any], field_name: str) -> bool:
    current = payload.get(field_name)
    if field_name in LIST_FIELDS:
        if current is None:
            return True
        if isinstance(current, list):
            return not [item for item in current if str(item).strip()]
        return not str(current).strip()
    return not _text(current)


def _split_list_tokens(value: str) -> list[str]:
    return split_standard_tokens(value)


def _has_conflicting_higher_priority(
    payload: dict[str, Any],
    field_name: str,
    value: str,
) -> bool:
    if field_name not in _TRACEABILITY_FIELD_SET:
        return False
    priority_index = LABELED_TRACEABILITY_FIELDS.index(field_name)
    for higher_field in LABELED_TRACEABILITY_FIELDS[:priority_index]:
        existing = _text(payload.get(higher_field))
        if existing and existing != value:
            return True
    return False


def _label_requires_delimiter(raw_label: str) -> bool:
    return normalize_header(raw_label) in _DELIMITER_REQUIRED_LABELS


def _gap_has_delimiter(text: str, label_end: int, value_start: int | None) -> bool:
    end = value_start if value_start is not None else min(len(text), label_end + 8)
    return bool(_DELIMITER_RE.search(text[label_end:end]))


def _value_from_span(
    text: str,
    tokens: list[re.Match[str]],
    label_end_index: int,
    next_label_start_index: int | None,
    canonical_field: str,
) -> str | None:
    start = tokens[label_end_index].end()
    end = tokens[next_label_start_index].start() if next_label_start_index is not None else len(text)
    raw = text[start:end].strip(" \t\r\n:;#.-/")
    if not raw:
        return None
    if canonical_field in DATE_FIELDS:
        identifier, date_value, date_conflict = split_identifier_and_date(raw)
        if date_conflict:
            return None
        return date_value
    if canonical_field in _IDENTIFIER_FIELD_SET:
        identifier, _date_value, _date_conflict = split_identifier_and_date(raw)
        if identifier:
            return identifier
        first = raw.split()[0].strip(" ,;")
        return first if looks_like_identifier_value(first) else None
    if canonical_field in LIST_FIELDS:
        cleaned = raw.strip(" ,;")
        return cleaned or None
    if canonical_field == "dimensions":
        cleaned = raw.strip(" ,;")
        return cleaned or None
    cleaned = raw.strip(" ,;")
    return cleaned or None


def _raw_span(
    text: str,
    tokens: list[re.Match[str]],
    label_end_index: int,
    next_label_start_index: int | None,
) -> str:
    start = tokens[label_end_index].end()
    end = tokens[next_label_start_index].start() if next_label_start_index is not None else len(text)
    return text[start:end].strip(" \t\r\n:;#.-/")


def extract_labeled_candidates_from_text(text: str) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Return labeled candidates and fields whose labeled values conflict."""

    source = text or ""
    tokens = list(_TOKEN_RE.finditer(source))
    if not tokens:
        return [], set()

    labels: list[tuple[int, int, str, str]] = []
    index = 0
    while index < len(tokens):
        follower = tokens[index + 1].group().upper() if index + 1 < len(tokens) else ""
        chosen: tuple[int, int, str, str] | None = None
        max_window = min(_MAX_LABEL_TOKENS, len(tokens) - index)
        for width in range(max_window, 0, -1):
            raw_label = source[tokens[index].start() : tokens[index + width - 1].end()]
            canonical = resolve_canonical_field(normalize_header(raw_label))
            if canonical not in _IDENTITY_FIELD_SET:
                continue
            if width == 1 and follower in _FOLLOWER_SKIP:
                continue
            next_token_start = tokens[index + width].start() if index + width < len(tokens) else None
            next_token = tokens[index + width].group() if index + width < len(tokens) else ""
            if _label_requires_delimiter(raw_label) and not _gap_has_delimiter(
                source, tokens[index + width - 1].end(), next_token_start
            ):
                if not (
                    canonical in _IDENTIFIER_FIELD_SET and looks_like_identifier_value(next_token)
                ):
                    continue
            chosen = (index, index + width - 1, canonical, raw_label.strip())
            break
        if chosen is None:
            index += 1
            continue
        labels.append(chosen)
        index = chosen[1] + 1

    matches: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    values_by_field: dict[str, set[str]] = {}
    conflicts: set[str] = set()
    for position, (_start_index, end_index, canonical, raw_label) in enumerate(labels):
        next_start = labels[position + 1][0] if position + 1 < len(labels) else None
        raw = _raw_span(source, tokens, end_index, next_start)
        value = _value_from_span(source, tokens, end_index, next_start, canonical)
        if value:
            folded = value.casefold() if canonical in LIST_FIELDS else value.casefold()
            key = (canonical, folded)
            if key not in seen:
                seen.add(key)
                values_by_field.setdefault(canonical, set()).add(folded)
                matches.append((canonical, raw_label, value))
        companion = IDENTIFIER_DATE_COMPANIONS.get(canonical)
        if companion and raw:
            _identifier, date_value, date_conflict = split_identifier_and_date(raw)
            if date_conflict:
                conflicts.add(companion)
            elif date_value:
                date_key = (companion, date_value.casefold())
                if date_key not in seen:
                    seen.add(date_key)
                    values_by_field.setdefault(companion, set()).add(date_value.casefold())
                    matches.append((companion, raw_label, date_value))

    for field_name, values in values_by_field.items():
        if len(values) > 1:
            conflicts.add(field_name)
    return matches, conflicts


def extract_labeled_identifiers_from_text(text: str) -> list[tuple[str, str, str]]:
    """Return unambiguous ``(canonical_field, raw_label, value)`` tuples."""

    matches, conflicts = extract_labeled_candidates_from_text(text)
    return [
        (canonical, raw_label, value)
        for canonical, raw_label, value in matches
        if canonical not in conflicts
    ]


def labeled_identity_field_count(text: str) -> int:
    """Count distinct identity fields evidenced by labels in ``text``."""

    return len({field for field, _label, _value in extract_labeled_identifiers_from_text(text)})


def is_combined_identity_line(value: Any) -> bool:
    """True when a string carries two or more independently labeled identity fields."""

    return labeled_identity_field_count(_text(value)) >= 2


def _ensure_field_confidence(payload: dict[str, Any]) -> dict[str, float]:
    existing = payload.get("field_confidence")
    if isinstance(existing, dict):
        return {str(key): float(value) for key, value in existing.items() if isinstance(value, (int, float))}
    return {}


def _assign_value(payload: dict[str, Any], canonical_field: str, value: str) -> None:
    if canonical_field in LIST_FIELDS:
        payload[canonical_field] = _split_list_tokens(value)
        return
    payload[canonical_field] = value


def _clear_conflicting_field(payload: dict[str, Any], field_name: str) -> None:
    if field_name in LIST_FIELDS:
        payload[field_name] = None
    else:
        payload[field_name] = None
    field_confidence = payload.get("field_confidence")
    if isinstance(field_confidence, dict):
        field_confidence.pop(field_name, None)


def _apply_match(
    payload: dict[str, Any],
    *,
    canonical_field: str,
    raw_label: str,
    value: str,
) -> bool:
    current_empty = _field_empty(payload, canonical_field)
    if not current_empty and not (
        canonical_field in {"product_name", "grade", "dimensions"} and is_combined_identity_line(payload.get(canonical_field))
    ):
        return False
    if _has_conflicting_higher_priority(payload, canonical_field, value):
        return False

    _assign_value(payload, canonical_field, value)
    field_confidence = _ensure_field_confidence(payload)
    field_confidence[canonical_field] = LABELED_IDENTIFIER_CONFIDENCE
    payload["field_confidence"] = field_confidence
    if canonical_field in _TRACEABILITY_FIELD_SET:
        payload.setdefault("traceability_identifier_label", raw_label.upper())
        payload.setdefault("traceability_identifier_type", canonical_field)
        payload.setdefault("traceability_identifier_value", value)
    return True


def _iter_row_text_fields(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key, raw in row.items():
        if str(key).startswith("_") or key in _SUPPLIER_SKIP_KEYS:
            continue
        if isinstance(raw, list):
            joined = " ".join(str(item).strip() for item in raw if str(item).strip())
            if joined:
                texts.append(joined)
            continue
        value = _text(raw)
        if value:
            texts.append(value)
    return texts


def _iter_metadata_text_fields(metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    sources: list[tuple[str, dict[str, Any], str]] = []
    for key, raw in metadata.items():
        if key in _SUPPLIER_SKIP_KEYS or str(key).startswith("_"):
            continue
        if isinstance(raw, list):
            joined = " ".join(str(item).strip() for item in raw if str(item).strip())
            if joined:
                sources.append((str(key), metadata, joined))
            continue
        value = _text(raw)
        if value:
            sources.append((str(key), metadata, value))
    return sources


def apply_labeled_identifiers(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply labeled identifier extraction to metadata and rows.

    Returns ``(tokens, traces)`` for finalization auditing.
    """

    tokens: list[str] = []
    traces: list[dict[str, Any]] = []

    def _apply_text(payload: dict[str, Any], text: str, *, scope: str, source: str | None, row_index: int | None) -> None:
        matches, conflicts = extract_labeled_candidates_from_text(text)
        for field_name in conflicts:
            _clear_conflicting_field(payload, field_name)
            token = f"conflicting_labeled_candidates:{field_name}"
            if token not in tokens:
                tokens.append(token)
            traces.append(
                {
                    "step": "labeled_identifier_conflict",
                    "scope": scope,
                    "source": source,
                    "row_index": row_index,
                    "field": field_name,
                    "status": "ambiguous",
                }
            )
        for canonical_field, raw_label, value in matches:
            if canonical_field in conflicts:
                continue
            if _apply_match(payload, canonical_field=canonical_field, raw_label=raw_label, value=value):
                tokens.append(f"{canonical_field}:from_labeled_text:{normalize_header(raw_label)}")
                traces.append(
                    {
                        "step": "labeled_identifier",
                        "scope": scope,
                        "source": source,
                        "row_index": row_index,
                        "field": canonical_field,
                        "label": raw_label,
                        "value": value,
                        "confidence": LABELED_IDENTIFIER_CONFIDENCE,
                    }
                )

    for source_name, payload, text in _iter_metadata_text_fields(metadata):
        _apply_text(payload, text, scope="metadata", source=source_name, row_index=None)

    for row_index, row in enumerate(rows):
        combined = " ".join(_iter_row_text_fields(row))
        if not combined.strip():
            continue
        _apply_text(row, combined, scope="row", source=None, row_index=row_index)

    return tokens, traces


__all__ = [
    "LABELED_IDENTIFIER_CONFIDENCE",
    "LABELED_TRACEABILITY_FIELDS",
    "apply_labeled_identifiers",
    "extract_labeled_candidates_from_text",
    "extract_labeled_identifiers_from_text",
    "is_combined_identity_line",
    "labeled_identity_field_count",
]
