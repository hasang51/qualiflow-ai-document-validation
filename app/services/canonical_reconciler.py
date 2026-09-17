"""Provenance-aware canonical reconciliation between Stage A/B and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.domain.field_mapping_registry import (
    normalize_header,
    resolve_canonical_field,
)
from app.domain.grade_registry import resolve_grade
from app.domain.labeled_identifier_extractor import extract_labeled_candidates_from_text
from app.domain.value_typing import (
    DATE_FIELDS,
    as_dimension_list,
    looks_like_classification_token,
    looks_like_spec_line,
    partition_standard_and_classification,
    serialize_dimensions,
    split_product_identity_and_details,
    split_standard_tokens,
)

SourceKind = Literal["labeled", "stage_a", "stage_b", "alias"]
SOURCE_RANK: dict[str, int] = {"labeled": 0, "stage_a": 1, "stage_b": 2, "alias": 3}

DOCUMENT_SCALAR_FIELDS: tuple[str, ...] = (
    "certificate_number",
    "certificate_date",
    "order_number",
    "order_date",
    "heat_number",
    "batch_number",
    "lot_number",
    "colata_number",
    "cast_number",
    "charge_number",
)
ROW_SCALAR_FIELDS: tuple[str, ...] = (
    *DOCUMENT_SCALAR_FIELDS,
    "product_name",
    "product_details",
    "grade",
    "dimensions",
)
LIST_FIELDS: tuple[str, ...] = ("standards", "classifications")
CONFLICT_PREFIX = "conflicting_labeled_candidates:"
_SUPPLIER_SKIP_KEYS = frozenset({"supplier_name", "document_type", "product_category"})


@dataclass
class ProvenancedValue:
    value: Any
    source: SourceKind
    label: str | None = None
    page: int | None = None
    confidence: float | None = None
    status: Literal["resolved", "ambiguous", "empty"] = "resolved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "label": self.label,
            "page": self.page,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class CanonicalReconciliationResult:
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    tokens: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    review_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "canonical_reconciliation",
            "tokens": list(self.tokens),
            "traces": list(self.traces),
            "review_tokens": list(self.review_tokens),
        }


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


def _fold(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(sorted(str(item).strip().casefold() for item in value if str(item).strip()))
    return _text(value).casefold()


def _iter_payload_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key, raw in payload.items():
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


def _confidence_for(payload: dict[str, Any], field_name: str) -> float | None:
    field_confidence = payload.get("field_confidence")
    if isinstance(field_confidence, dict) and isinstance(field_confidence.get(field_name), (int, float)):
        return float(field_confidence[field_name])
    return None


def _page_for(payload: dict[str, Any]) -> int | None:
    page = payload.get("source_page")
    return int(page) if isinstance(page, int) else None


def _record_conflict(tokens: list[str], review_tokens: list[str], field_name: str) -> None:
    token = f"{CONFLICT_PREFIX}{field_name}"
    if token not in tokens:
        tokens.append(token)
    if token not in review_tokens:
        review_tokens.append(token)


def _merge_unique(values: list[str] | None, extra: list[str]) -> list[str] | None:
    seen: set[str] = set()
    merged: list[str] = []
    for token in [*(values or []), *extra]:
        key = token.casefold()
        if not token or key in seen:
            continue
        seen.add(key)
        merged.append(token)
    return merged or None


def _alias_candidates(payload: dict[str, Any], field_name: str) -> list[ProvenancedValue]:
    found: list[ProvenancedValue] = []
    for key, raw in payload.items():
        if key == field_name or str(key).startswith("_") or key in _SUPPLIER_SKIP_KEYS:
            continue
        if resolve_canonical_field(str(key)) != field_name:
            continue
        if _empty(raw):
            continue
        found.append(
            ProvenancedValue(
                value=list(split_standard_tokens(raw)) if field_name in LIST_FIELDS else _text(raw),
                source="alias",
                label=str(key),
                page=_page_for(payload),
                confidence=_confidence_for(payload, field_name),
            )
        )
    return found


def _typed_candidate(payload: dict[str, Any], field_name: str, source: SourceKind) -> ProvenancedValue | None:
    if field_name not in payload or _empty(payload.get(field_name)):
        return None
    raw = payload.get(field_name)
    value: Any
    if field_name in LIST_FIELDS:
        value = list(split_standard_tokens(raw))
    elif field_name == "dimensions":
        value = serialize_dimensions(raw)
    else:
        value = _text(raw)
    if _empty(value):
        return None
    return ProvenancedValue(
        value=value,
        source=source,
        label=field_name,
        page=_page_for(payload),
        confidence=_confidence_for(payload, field_name),
    )


def _labeled_from_texts(texts: list[str]) -> tuple[dict[str, list[ProvenancedValue]], set[str]]:
    by_field: dict[str, list[ProvenancedValue]] = {}
    conflicts: set[str] = set()
    for text in texts:
        matches, local_conflicts = extract_labeled_candidates_from_text(text)
        conflicts.update(local_conflicts)
        for canonical, raw_label, value in matches:
            payload_value: Any = list(split_standard_tokens(value)) if canonical in LIST_FIELDS else value
            by_field.setdefault(canonical, []).append(
                ProvenancedValue(
                    value=payload_value,
                    source="labeled",
                    label=raw_label,
                    confidence=0.70,
                )
            )
    for field_name, candidates in by_field.items():
        distinct = {_fold(item.value) for item in candidates if not _empty(item.value)}
        if len(distinct) > 1:
            conflicts.add(field_name)
    return by_field, conflicts


def _labeled_dates(metadata: dict[str, Any]) -> tuple[dict[str, list[ProvenancedValue]], set[str]]:
    by_field: dict[str, list[ProvenancedValue]] = {}
    conflicts: set[str] = set()
    labeled = metadata.get("labeled_dates")
    if not isinstance(labeled, list):
        return by_field, conflicts
    for item in labeled:
        if not isinstance(item, dict):
            continue
        label = _text(item.get("label"))
        value = _text(item.get("value"))
        if not label or not value:
            continue
        canonical = resolve_canonical_field(normalize_header(label))
        if canonical not in DATE_FIELDS:
            continue
        by_field.setdefault(canonical, []).append(
            ProvenancedValue(value=value, source="labeled", label=label, confidence=0.70)
        )
    for field_name, candidates in by_field.items():
        if len({_fold(item.value) for item in candidates}) > 1:
            conflicts.add(field_name)
    return by_field, conflicts


def _choose(
    candidates: list[ProvenancedValue],
    *,
    conflicts: set[str],
    field_name: str,
) -> ProvenancedValue:
    if field_name in conflicts:
        return ProvenancedValue(value=None, source="labeled", status="ambiguous")
    ranked = sorted(candidates, key=lambda item: SOURCE_RANK.get(item.source, 99))
    for candidate in ranked:
        if not _empty(candidate.value):
            return candidate
    return ProvenancedValue(value=None, source="alias", status="empty")


def _partition_row_specs(row: dict[str, Any]) -> None:
    combined = []
    for key in ("standards", "classifications"):
        combined.extend(split_standard_tokens(row.get(key)))
    standards, classifications = partition_standard_and_classification(combined)
    row["standards"] = standards or None
    row["classifications"] = classifications or None


def _separate_product_and_details(row: dict[str, Any]) -> None:
    identity, details = split_product_identity_and_details(row.get("product_name"))
    if identity and identity != _text(row.get("product_name")):
        row["product_name"] = identity
        if _empty(row.get("product_details")) and details:
            row["product_details"] = details
    elif details and _empty(row.get("product_details")):
        row["product_details"] = details
        if identity:
            row["product_name"] = identity
    if looks_like_spec_line(row.get("product_name")) or looks_like_classification_token(row.get("product_name")):
        standards, classifications = partition_standard_and_classification(
            _merge_unique(row.get("standards"), split_standard_tokens(row.get("product_name")))
        )
        row["standards"] = standards or None
        row["classifications"] = _merge_unique(row.get("classifications"), classifications)
        row["product_name"] = None
    product = _text(row.get("product_name"))
    details_text = _text(row.get("product_details"))
    if product and details_text and product.casefold() == details_text.casefold():
        row["product_details"] = None


def _separate_grade_from_classification(row: dict[str, Any]) -> None:
    grade = _text(row.get("grade"))
    if not grade:
        row["grade"] = None
        return
    if looks_like_classification_token(grade) and not looks_like_spec_line(grade):
        resolution = resolve_grade(grade)
        if resolution.status not in {"resolved", "resolved_dual"}:
            row["classifications"] = _merge_unique(row.get("classifications"), [grade])
            row["grade"] = None
            return
    if looks_like_spec_line(grade):
        standards, classifications = partition_standard_and_classification(grade)
        row["standards"] = _merge_unique(row.get("standards"), standards)
        row["classifications"] = _merge_unique(row.get("classifications"), classifications)
        recovered = resolve_grade(grade)
        row["grade"] = recovered.canonical if recovered.status in {"resolved", "resolved_dual"} else None


def _set_field(payload: dict[str, Any], field_name: str, chosen: ProvenancedValue, provenance: dict[str, Any]) -> None:
    if chosen.status == "ambiguous":
        payload[field_name] = None
    elif field_name == "dimensions":
        payload[field_name] = serialize_dimensions(chosen.value)
    elif field_name in LIST_FIELDS:
        payload[field_name] = list(chosen.value) if isinstance(chosen.value, list) else split_standard_tokens(chosen.value)
        if not payload[field_name]:
            payload[field_name] = None
    else:
        payload[field_name] = chosen.value if not _empty(chosen.value) else None
    provenance[field_name] = chosen.to_dict()


def apply_canonical_reconciliation(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> CanonicalReconciliationResult:
    """Resolve canonical certificate semantics with label + shape + provenance."""

    meta = dict(metadata) if isinstance(metadata, dict) else {}
    mapped_rows = [dict(row) for row in rows if isinstance(row, dict)]
    tokens: list[str] = []
    traces: list[dict[str, Any]] = []
    review_tokens: list[str] = []

    meta_texts = _iter_payload_texts(meta)
    labeled_meta, meta_conflicts = _labeled_from_texts(meta_texts)
    date_labeled, date_conflicts = _labeled_dates(meta)
    meta_conflicts.update(date_conflicts)
    for field_name, candidates in date_labeled.items():
        labeled_meta.setdefault(field_name, []).extend(candidates)
    for field_name in meta_conflicts:
        _record_conflict(tokens, review_tokens, field_name)

    meta_provenance: dict[str, Any] = {}
    for field_name in (*DOCUMENT_SCALAR_FIELDS, "product_name", "product_details", *LIST_FIELDS):
        candidates = list(labeled_meta.get(field_name, []))
        typed = _typed_candidate(meta, field_name, "stage_a")
        if typed is not None:
            candidates.append(typed)
        candidates.extend(_alias_candidates(meta, field_name))
        chosen = _choose(candidates, conflicts=meta_conflicts, field_name=field_name)
        _set_field(meta, field_name, chosen, meta_provenance)
        if chosen.status == "resolved" and not _empty(chosen.value):
            tokens.append(f"canonical:{field_name}:{chosen.source}")
    traces.append({"scope": "metadata", "provenance": meta_provenance})

    for index, row in enumerate(mapped_rows):
        texts = [*meta_texts, *_iter_payload_texts(row)]
        labeled_row, row_conflicts = _labeled_from_texts(texts)
        row_conflicts.update(meta_conflicts)
        for field_name in row_conflicts:
            _record_conflict(tokens, review_tokens, field_name)
        provenance: dict[str, Any] = {}
        for field_name in (*ROW_SCALAR_FIELDS, *LIST_FIELDS):
            candidates = list(labeled_row.get(field_name, []))
            typed_row = _typed_candidate(row, field_name, "stage_b")
            typed_meta = _typed_candidate(meta, field_name, "stage_a")
            if typed_row is not None:
                candidates.append(typed_row)
            if typed_meta is not None:
                candidates.append(typed_meta)
            candidates.extend(_alias_candidates(row, field_name))
            candidates.extend(_alias_candidates(meta, field_name))
            chosen = _choose(candidates, conflicts=row_conflicts, field_name=field_name)
            _set_field(row, field_name, chosen, provenance)
            if chosen.status == "resolved" and not _empty(chosen.value):
                tokens.append(f"canonical:{field_name}:{chosen.source}")
        _partition_row_specs(row)
        _separate_product_and_details(row)
        _separate_grade_from_classification(row)
        _partition_row_specs(row)
        if as_dimension_list(row.get("dimensions")):
            row["dimensions"] = serialize_dimensions(row.get("dimensions"))
        row["_field_provenance"] = provenance
        traces.append({"scope": "row", "row_index": index, "provenance": provenance})

    unique_review = []
    for token in review_tokens:
        if token not in unique_review:
            unique_review.append(token)
    return CanonicalReconciliationResult(
        metadata=meta,
        rows=mapped_rows,
        tokens=tokens,
        traces=traces,
        review_tokens=unique_review,
    )


__all__ = [
    "CanonicalReconciliationResult",
    "ProvenancedValue",
    "apply_canonical_reconciliation",
]
