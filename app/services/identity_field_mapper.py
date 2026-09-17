"""Deterministic semantic separation for certificate identity fields.

Keeps product name, grade, standards, dimensions, and document identifiers
independent. Mapping uses the field registry, value-shape checks, and the
existing grade registry. It does not use supplier names or document-specific
labels, and it does not invent values when evidence is missing or mixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.field_mapping_registry import (
    IDENTIFIER_HEADER_FIELDS,
    resolve_canonical_field,
)
from app.domain.grade_registry import resolve_grade
from app.domain.labeled_identifier_extractor import (
    extract_labeled_identifiers_from_text,
    is_combined_identity_line,
)
from app.domain.value_typing import (
    looks_like_classification_token,
    looks_like_spec_line,
    partition_standard_and_classification,
    split_product_identity_and_details,
)

_DIMENSION_RE = re.compile(
    r"(?:Ø|DIAM(?:ETER|\.)?|DIA)?\s*(\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?)?\s*(?:mm|cm|m|in)\b)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_SPLIT_RE = re.compile(r"\s*[/,;|]\s*")
_DETAIL_SUFFIX_RE = re.compile(
    r"\s*[-–]\s*(?:DIAM|Ø|SIZE)|"
    r"\s+DIAM\.?|"
    r"\s*Ø|"
    r"\s+P\.L\.W\.|"
    r"\s*\(Kg",
    re.IGNORECASE,
)

HEADER_PRODUCT_KEYS = ("product_name", "product_description", "material_name")
HEADER_GRADE_KEYS = ("header_grade", "document_grade", "product_grade", "grade")
HEADER_DIMENSION_KEYS = ("dimensions", "diameter", "size")
HEADER_STANDARD_KEYS = ("standards", "header_standards")
HEADER_CLASSIFICATION_KEYS = ("classifications", "header_classifications")
HEADER_IDENTIFIER_KEYS: dict[str, tuple[str, ...]] = {
    "certificate_number": ("certificate_number",),
    "certificate_date": ("certificate_date",),
    "order_number": ("order_number",),
    "order_date": ("order_date",),
    "heat_number": ("heat_number", "header_heat_number"),
    "batch_number": ("batch_number",),
    "lot_number": ("lot_number",),
    "colata_number": ("colata_number",),
    "cast_number": ("cast_number",),
    "charge_number": ("charge_number",),
}
DOCUMENT_HEADER_IDENTIFIER_FIELDS: tuple[str, ...] = IDENTIFIER_HEADER_FIELDS
SUPPLIER_SKIP_KEYS = ("supplier_name", "manufacturer", "producer", "company")


@dataclass
class IdentityMappingResult:
    rows: list[dict[str, Any]]
    tokens: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)


def _text(value: Any) -> str:
    if isinstance(value, list):
        return ""
    return str(value).strip() if value is not None else ""


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not [item for item in value if str(item).strip()]
    return False


def _metadata_text(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        text = _text(value)
        if text:
            return text
    return None


def _standards_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return _split_standard_tokens(value)
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    values: list[str] = []
    for item in value:
        for token in _split_standard_tokens(str(item)):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(token)
    return values


def _split_standard_tokens(value: str) -> list[str]:
    text = _text(value)
    if not text:
        return []
    parts = [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
    return parts or [text]


def _merge_standards(*groups: Any) -> list[str] | None:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for token in _standards_list(group):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(token)
    return merged or None


def looks_like_dimension(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    if looks_like_classification_token(text) or looks_like_spec_line(text):
        return False
    if not _DIMENSION_RE.search(text):
        return False
    remainder = _DIMENSION_RE.sub("", text)
    remainder = re.sub(r"\b(DIAM(?:ETER|\.)?|DIA|Ø)\b", "", remainder, flags=re.IGNORECASE)
    remainder = remainder.strip(" -/,()")
    return len(remainder) <= 8 or not _WORD_RE.search(remainder)


def extract_dimension_fragment(value: Any) -> str | None:
    match = _DIMENSION_RE.search(_text(value))
    if not match:
        return None
    fragment = match.group(1).strip()
    return fragment.replace(",", ".") if fragment else None


def _strip_detail_suffix(value: str) -> str:
    text = _text(value)
    if not text:
        return text
    match = _DETAIL_SUFFIX_RE.search(text)
    if match:
        return text[: match.start()].strip(" -/")
    return text


def _domain_grade(value: str | None, *, allow_contained: bool) -> str | None:
    text = _text(value)
    if not text:
        return None
    resolution = resolve_grade(text)
    if resolution.status not in {"resolved", "resolved_dual"} or not resolution.canonical:
        return None
    if not allow_contained and "contained" in (resolution.reason or ""):
        return None
    return resolution.canonical


def _assign_spec_tokens(row: dict[str, Any], value: Any) -> None:
    standards, classifications = partition_standard_and_classification(
        _merge_standards(row.get("standards"), row.get("classifications"), value)
    )
    row["standards"] = standards or None
    row["classifications"] = classifications or None


def _tokens_are_classifications(value: str) -> bool:
    tokens = _split_standard_tokens(value)
    return bool(tokens) and all(looks_like_classification_token(token) for token in tokens)


def is_standards_only(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    if _tokens_are_classifications(text):
        return True
    if looks_like_spec_line(text) and _domain_grade(text, allow_contained=False) is None:
        return True
    return False


def _looks_like_product_identity(value: Any) -> bool:
    text = _text(value)
    if not text or is_standards_only(text) or looks_like_classification_token(text):
        return False
    if looks_like_dimension(text):
        return False
    labeled_fields = {field for field, _label, _value in extract_labeled_identifiers_from_text(text)}
    if labeled_fields & set(IDENTIFIER_HEADER_FIELDS):
        return False
    has_separator = " " in text or "/" in text
    if not has_separator:
        return False
    exact = _domain_grade(text, allow_contained=False)
    if exact and not _WORD_RE.search(re.sub(re.escape(exact), "", text, flags=re.IGNORECASE)):
        return False
    return bool(_WORD_RE.search(text))


def _append_token(tokens: list[str], token: str) -> None:
    if token not in tokens:
        tokens.append(token)


def _first_alias_value(payload: dict[str, Any], canonical_field: str) -> Any:
    if canonical_field in payload and not _empty(payload.get(canonical_field)):
        return payload.get(canonical_field)
    for key, value in payload.items():
        if key == canonical_field or str(key).startswith("_"):
            continue
        if str(key) in SUPPLIER_SKIP_KEYS:
            continue
        if resolve_canonical_field(str(key)) == canonical_field and not _empty(value):
            return value
    return payload.get(canonical_field)


def _set_if_empty(row: dict[str, Any], field_name: str, value: Any) -> bool:
    if _empty(row.get(field_name)) and not _empty(value):
        row[field_name] = value
        return True
    return False


def _lift_canonical_aliases(row: dict[str, Any], tokens: list[str], trace: dict[str, Any]) -> None:
    """Precedence: explicit canonical field > normalized alias mapping."""

    for field_name in (*IDENTIFIER_HEADER_FIELDS, "product_name", "product_details", "grade", "dimensions"):
        current = row.get(field_name)
        if not _empty(current):
            continue
        alias_value = _first_alias_value(row, field_name)
        if _empty(alias_value):
            continue
        if field_name == "dimensions":
            row[field_name] = _text(alias_value)
        elif field_name in {"standards", "classifications"}:
            continue
        else:
            row[field_name] = _text(alias_value)
        _append_token(tokens, f"identity:alias_mapped:{field_name}")
        trace[f"{field_name}_from_alias"] = row[field_name]

    for spec_field in ("standards", "classifications"):
        alias_value = _first_alias_value(row, spec_field)
        if not _empty(alias_value):
            _assign_spec_tokens(row, alias_value)
            _append_token(tokens, f"identity:alias_mapped:{spec_field}")


def _reclassify_grade_value(row: dict[str, Any], tokens: list[str], trace: dict[str, Any]) -> None:
    current = _text(row.get("grade"))
    if not current:
        return

    dimension = extract_dimension_fragment(current)
    if dimension and _set_if_empty(row, "dimensions", dimension):
        _append_token(tokens, "identity:dimensions_from_grade")
        trace["dimensions_from_grade"] = dimension

    if is_standards_only(current):
        _assign_spec_tokens(row, current)
        recovered = _domain_grade(current, allow_contained=True)
        row["grade"] = recovered
        _append_token(tokens, "identity:standards_moved_from_grade")
        trace["standards_from_grade"] = current
        if recovered:
            _append_token(tokens, "identity:grade_from_domain_rules")
            trace["grade_from_standards_domain"] = recovered
        return

    if looks_like_dimension(current) and not _looks_like_product_identity(current):
        if _set_if_empty(row, "dimensions", current):
            _append_token(tokens, "identity:dimensions_from_grade")
        row["grade"] = None
        return

    if _looks_like_product_identity(current):
        product, details = split_product_identity_and_details(current)
        product = product or _strip_detail_suffix(current)
        if _set_if_empty(row, "product_name", product):
            _append_token(tokens, "identity:product_name_moved_from_grade")
            trace["product_name_from_grade"] = product
        if details and _set_if_empty(row, "product_details", details):
            _append_token(tokens, "identity:product_details_from_grade")
        recovered = _domain_grade(current, allow_contained=True)
        row["grade"] = recovered
        if recovered:
            _append_token(tokens, "identity:grade_from_product_identity")
            trace["grade_from_product_identity"] = recovered
        return

    mixed_standards = [token for token in _split_standard_tokens(current) if looks_like_classification_token(token)]
    if mixed_standards and _domain_grade(current, allow_contained=True):
        _assign_spec_tokens(row, mixed_standards)
        recovered = _domain_grade(current, allow_contained=True)
        if recovered:
            row["grade"] = recovered
            _append_token(tokens, "identity:grade_separated_from_classifications")


def _reclassify_product_value(row: dict[str, Any], tokens: list[str], trace: dict[str, Any]) -> None:
    current = _text(row.get("product_name"))
    if not current:
        return

    if is_standards_only(current):
        _assign_spec_tokens(row, current)
        row["product_name"] = None
        _append_token(tokens, "identity:standards_moved_from_product_name")
        trace["rejected_product_name"] = current
        recovered = _domain_grade(current, allow_contained=False)
        if recovered and _empty(row.get("grade")):
            row["grade"] = recovered
            _append_token(tokens, "identity:grade_from_domain_rules")
        return

    identity, details = split_product_identity_and_details(current)
    if details:
        if identity:
            row["product_name"] = identity
        if _set_if_empty(row, "product_details", details):
            _append_token(tokens, "identity:product_details_from_product_name")

    if looks_like_dimension(current) and not _looks_like_product_identity(current):
        if _set_if_empty(row, "dimensions", current):
            _append_token(tokens, "identity:dimensions_from_product_name")
        row["product_name"] = None
        return

    if _empty(row.get("grade")):
        recovered = _domain_grade(current, allow_contained=True)
        if recovered:
            row["grade"] = recovered
            _append_token(tokens, "identity:grade_from_product_name")
            trace["grade_from_product_name"] = recovered


def _reclassify_item_id(row: dict[str, Any], tokens: list[str]) -> None:
    current = _text(row.get("item_id"))
    if not current:
        return
    if _tokens_are_classifications(current):
        _assign_spec_tokens(row, current)
        row["item_id"] = None
        _append_token(tokens, "identity:standards_moved_from_item_id")
        return
    if looks_like_dimension(current) and _set_if_empty(row, "dimensions", current):
        row["item_id"] = None
        _append_token(tokens, "identity:dimensions_from_item_id")


def _labeled_field_value(text: str, field_name: str) -> str | None:
    for canonical, _label, value in extract_labeled_identifiers_from_text(text):
        if canonical == field_name and _text(value):
            return _text(value)
    return None


def _usable_product_text(value: Any, *, supplier_name: str | None = None) -> str | None:
    text = _text(value)
    if not text:
        return None
    labeled_product = _labeled_field_value(text, "product_name")
    if labeled_product:
        text = labeled_product
    elif is_combined_identity_line(text) or _labeled_field_value(text, "certificate_number") or _labeled_field_value(
        text, "order_number"
    ):
        return None
    if is_standards_only(text):
        return None
    if looks_like_dimension(text) and not _looks_like_product_identity(text):
        return None
    if supplier_name and text.casefold() == supplier_name.casefold():
        return None
    return text


def _usable_grade_text(value: Any, *, supplier_name: str | None = None) -> str | None:
    text = _text(value)
    if not text:
        return None
    labeled_grade = _labeled_field_value(text, "grade")
    if labeled_grade:
        text = labeled_grade
    elif is_combined_identity_line(text):
        return None
    if supplier_name and text.casefold() == supplier_name.casefold():
        return None
    return text


def _backfill_identifiers_from_metadata(
    row: dict[str, Any],
    metadata: dict[str, Any],
    tokens: list[str],
    trace: dict[str, Any],
) -> None:
    for field_name, keys in HEADER_IDENTIFIER_KEYS.items():
        if not _empty(row.get(field_name)):
            continue
        value = _metadata_text(metadata, keys)
        if not value:
            alias_value = _first_alias_value(metadata, field_name)
            value = _text(alias_value) if not _empty(alias_value) else ""
        if not value or is_combined_identity_line(value):
            continue
        row[field_name] = value
        _append_token(tokens, f"identity:{field_name}_from_metadata")
        trace[f"{field_name}_from_metadata"] = value


def _backfill_from_metadata(
    row: dict[str, Any],
    metadata: dict[str, Any],
    tokens: list[str],
    trace: dict[str, Any],
) -> None:
    supplier_name = _metadata_text(metadata, SUPPLIER_SKIP_KEYS)
    _backfill_identifiers_from_metadata(row, metadata, tokens, trace)

    product = None
    for key in HEADER_PRODUCT_KEYS:
        candidate = _usable_product_text(metadata.get(key), supplier_name=supplier_name)
        if candidate:
            product = candidate
            break
    if product and is_standards_only(product):
        _assign_spec_tokens(row, product)
        _append_token(tokens, "identity:standards_from_metadata_product")
    elif product and not looks_like_dimension(product):
        identity, details = split_product_identity_and_details(product)
        if _set_if_empty(row, "product_name", identity or product):
            _append_token(tokens, "product_name:from_metadata")
            trace["product_name_from_metadata"] = identity or product
        if details and _set_if_empty(row, "product_details", details):
            _append_token(tokens, "identity:product_details_from_metadata")
        if _empty(row.get("grade")):
            recovered = _domain_grade(product, allow_contained=True)
            if recovered:
                row["grade"] = recovered
                _append_token(tokens, "identity:grade_from_metadata_product")

    header_grade = None
    for key in HEADER_GRADE_KEYS:
        candidate = _usable_grade_text(metadata.get(key), supplier_name=supplier_name)
        if candidate:
            header_grade = candidate
            break
    if header_grade:
        if is_standards_only(header_grade):
            _assign_spec_tokens(row, header_grade)
            _append_token(tokens, "identity:standards_from_metadata_grade")
            recovered = _domain_grade(header_grade, allow_contained=False)
            if recovered and _empty(row.get("grade")):
                row["grade"] = recovered
                _append_token(tokens, "identity:grade_from_domain_rules")
        elif _looks_like_product_identity(header_grade):
            if _set_if_empty(row, "product_name", header_grade):
                _append_token(tokens, "product_name:from_metadata_grade")
            recovered = _domain_grade(header_grade, allow_contained=True)
            if recovered and _empty(row.get("grade")):
                row["grade"] = recovered
                _append_token(tokens, "identity:grade_from_metadata_grade")
        elif _empty(row.get("grade")) and not looks_like_classification_token(header_grade):
            row["grade"] = header_grade
            _append_token(tokens, "context_propagation:grade_from_metadata")

    dimensions = _metadata_text(metadata, HEADER_DIMENSION_KEYS)
    if dimensions and not is_combined_identity_line(dimensions) and _set_if_empty(row, "dimensions", dimensions):
        _append_token(tokens, "identity:dimensions_from_metadata")

    header_standards = metadata.get("standards")
    if header_standards is None:
        for key in HEADER_STANDARD_KEYS[1:]:
            if metadata.get(key) is not None:
                header_standards = metadata.get(key)
                break
    header_classifications = metadata.get("classifications")
    if header_classifications is None:
        for key in HEADER_CLASSIFICATION_KEYS[1:]:
            if metadata.get(key) is not None:
                header_classifications = metadata.get(key)
                break
    if header_standards is not None or header_classifications is not None:
        before_std = row.get("standards")
        before_cls = row.get("classifications")
        _assign_spec_tokens(row, _merge_standards(header_standards, header_classifications))
        if row.get("standards") != before_std or row.get("classifications") != before_cls:
            _append_token(tokens, "identity:standards_from_metadata")


def _map_row(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    mapped = dict(row)
    tokens: list[str] = []
    trace: dict[str, Any] = {"strategy": "identity_field_separation"}

    _lift_canonical_aliases(mapped, tokens, trace)
    if mapped.get("product_name") and is_combined_identity_line(mapped.get("product_name")):
        mapped["product_name"] = None
        _append_token(tokens, "identity:rejected_combined_header_product")
    if mapped.get("grade") and is_combined_identity_line(mapped.get("grade")):
        mapped["grade"] = None
        _append_token(tokens, "identity:rejected_combined_header_grade")

    _reclassify_item_id(mapped, tokens)
    _reclassify_product_value(mapped, tokens, trace)
    _reclassify_grade_value(mapped, tokens, trace)
    _backfill_from_metadata(mapped, metadata, tokens, trace)
    _reclassify_product_value(mapped, tokens, trace)
    _reclassify_grade_value(mapped, tokens, trace)

    _assign_spec_tokens(mapped, None)
    product = _text(mapped.get("product_name"))
    grade = _text(mapped.get("grade"))
    if product and is_combined_identity_line(product):
        mapped["product_name"] = None
        product = ""
    if product and grade and product.casefold() == grade.casefold():
        mapped["product_name"] = None
    elif (
        product
        and grade
        and not _looks_like_product_identity(product)
        and _domain_grade(product, allow_contained=False) == grade
    ):
        mapped["product_name"] = None
    if _empty(mapped.get("product_name")):
        mapped["product_name"] = None
    if _empty(mapped.get("product_details")):
        mapped["product_details"] = None
    if _empty(mapped.get("grade")):
        mapped["grade"] = None
    if _empty(mapped.get("dimensions")):
        mapped["dimensions"] = None
    for field_name in ("order_date", "certificate_date", *IDENTIFIER_HEADER_FIELDS):
        if _empty(mapped.get(field_name)):
            mapped[field_name] = None
    return mapped, tokens, trace


def promote_unique_header_fields(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    """Copy a uniquely agreed row identifier onto empty document metadata."""

    tokens: list[str] = []
    for field_name in DOCUMENT_HEADER_IDENTIFIER_FIELDS:
        if not _empty(metadata.get(field_name)):
            continue
        values = {
            _text(row.get(field_name))
            for row in rows
            if isinstance(row, dict) and not _empty(row.get(field_name))
        }
        if len(values) != 1:
            continue
        metadata[field_name] = next(iter(values))
        tokens.append(f"identity:{field_name}_promoted_from_rows")
    return tokens


def apply_identity_field_mapping(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> IdentityMappingResult:
    """Separate product, grade, standards, and dimensions without guessing."""

    meta = metadata if isinstance(metadata, dict) else {}
    mapped_rows: list[dict[str, Any]] = []
    tokens: list[str] = []
    traces: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        mapped, row_tokens, row_trace = _map_row(row, meta)
        mapped_rows.append(mapped)
        tokens.extend(row_tokens)
        row_trace["row_index"] = index
        if row_tokens:
            traces.append(row_trace)
    return IdentityMappingResult(rows=mapped_rows, tokens=tokens, traces=traces)


__all__ = [
    "DOCUMENT_HEADER_IDENTIFIER_FIELDS",
    "HEADER_DIMENSION_KEYS",
    "HEADER_GRADE_KEYS",
    "HEADER_IDENTIFIER_KEYS",
    "HEADER_PRODUCT_KEYS",
    "IdentityMappingResult",
    "apply_identity_field_mapping",
    "extract_dimension_fragment",
    "is_standards_only",
    "looks_like_classification_token",
    "looks_like_dimension",
    "looks_like_spec_line",
    "promote_unique_header_fields",
]
