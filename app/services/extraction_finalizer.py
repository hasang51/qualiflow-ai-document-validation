"""Generic post-processing finalization after multimodal extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.field_mapping_registry import normalize_header
from app.domain.labeled_identifier_extractor import apply_labeled_identifiers
from app.schemas.extraction import ExtractedItem, UniversalDocumentExtraction
from app.services.identity_field_mapper import apply_identity_field_mapping
from app.services.traceability import TRACEABILITY_VERIFIED, apply_traceability_remarks

# Certificate date label priority (normalized header form).
_CERTIFICATE_DATE_PRIORITY: tuple[str, ...] = (
    "DATA DATE",
    "DATE",
    "CERTIFICATE DATE",
    "CERT DATE",
    "TEST DATE",
)

_EXCLUDED_DATE_LABELS: frozenset[str] = frozenset(
    normalize_header(label)
    for label in (
        "PO DATE",
        "PURCHASE ORDER DATE",
        "DELIVERY DATE",
        "ANALYSIS TIMESTAMP",
        "UPLOAD TIMESTAMP",
        "ISSUE METADATA",
    )
)

# Grade candidate source priority (normalized header form).
_GRADE_SOURCE_PRIORITY: tuple[str, ...] = (
    "PRODUCT NAME",
    "MATERIAL",
    "GRADE",
    "DESCRIPTION",
)

_METADATA_GRADE_KEYS: tuple[str, ...] = ("header_grade", "product_description")

_ROW_AUTO_ACCEPT_FIELDS: frozenset[str] = frozenset({"grade", "weight_or_length"})
_MECHANICAL_AUTO_ACCEPT_FIELDS: frozenset[str] = frozenset(
    {"yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage"}
)

_AUTO_ACCEPT_CRITICAL_FIELDS: tuple[str, ...] = (
    "grade",
    "weight_or_length",
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "elongation_percentage",
)

_TRACEABILITY_IDENTIFIER_PRIORITY: tuple[str, ...] = (
    "heat_number",
    "batch_number",
    "colata_number",
    "lot_number",
    "cast_number",
    "charge_number",
    "coil_number",
    "traceability_identifier_value",
)

_ITEM_ID_ALIASES: tuple[str, ...] = (
    "item_id",
    "item_identifier",
    "pipe_coil_id",
    "pipe_or_coil_id",
    "pipe_id",
)

_PLACEHOLDER_ITEM_IDS: frozenset[str] = frozenset(
    {"", "-", "—", "n/a", "na", "none", "null"}
)

_GRADE_DETAIL_SEPARATOR_PATTERNS: tuple[str, ...] = (
    r"\s*-\s*DIAM",
    r"\s+DIAM\.?",
    r"\s+DIAM\s",
    r"\s*Ø",
    r"\s+SIZE\b",
    r"\s+KG\b",
    r"\s*\(Kg",
    r"\s+P\.L\.W\.",
)

_SHORT_GRADE_RE = re.compile(r"^(SG\d+|S\d+[A-Z0-9+]*|\d{3,4}[A-Z]?)$", re.IGNORECASE)
_SPEC_LINE_RE = re.compile(r"\b(EN\s*ISO|AWS|ASTM)\b", re.IGNORECASE)
_DATE_VALUE_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$")

_BLOCKING_DEVIATION_MARKERS: tuple[str, ...] = (
    "looks suspicious",
    "unknown grade",
    "ambiguous",
    "non-compliant",
    "manual review required",
    "cannot verify",
)

_CONFIDENCE_ONLY_REASONS: frozenset[str] = frozenset(
    {
        "confidence falls below threshold",
        "confidence_below_threshold",
        "low_confidence",
        "overall_decision_confidence_below_threshold",
    }
)

_HARD_AUTO_ACCEPT_BLOCKER_EXACT: frozenset[str] = frozenset(
    {
        "quality_blocker:severe_scan",
        "unresolved_grade",
        "ambiguous_grade",
        "explicit_unmapped_grade",
        "unresolved_spec",
        "unsupported_spec_family",
        "mechanical_table_alignment_uncertain",
        "traceability_identifier_ocr_uncertain",
        "traceability_identifier_conflict",
        "traceability_unverified",
        "missing_critical_identifier_group",
        "row_count_inconsistent",
        "no_items_extracted",
    }
)

_HARD_AUTO_ACCEPT_BLOCKER_PREFIXES: tuple[str, ...] = (
    "missing_critical_field:",
    "critical_identifier_unverified",
    "header_row_conflict:",
    "validation_conflict:",
)


@dataclass
class FinalizationResult:
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    tokens: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "pre_validation",
            "tokens": list(self.tokens),
            "traces": list(self.traces),
        }


@dataclass
class CanonicalFinalizationResult:
    tokens: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    missing_critical_fields_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "canonical_response",
            "tokens": list(self.tokens),
            "traces": list(self.traces),
            "missing_critical_fields_rate": self.missing_critical_fields_rate,
        }


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_placeholder_item_id(value: Any) -> bool:
    return _text(value).lower() in _PLACEHOLDER_ITEM_IDS


def _normalize_date_value(value: str) -> str:
    text = _text(value)
    match = _DATE_VALUE_RE.match(text)
    if not match:
        return text
    day, month, year = match.groups()
    return f"{int(day):02d}/{int(month):02d}/{year}"


def _is_excluded_date_label(label: str) -> bool:
    normalized = normalize_header(label)
    if normalized in _EXCLUDED_DATE_LABELS:
        return True
    for excluded in _EXCLUDED_DATE_LABELS:
        if len(normalized) > len(excluded) and excluded in normalized:
            return True
    return False


def _collect_labeled_dates(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    labeled = metadata.get("labeled_dates")
    if isinstance(labeled, list):
        for item in labeled:
            if not isinstance(item, dict):
                continue
            label = _text(item.get("label"))
            value = _text(item.get("value"))
            if label and value:
                entries.append((label, normalize_header(label), value))
    return entries


def _resolve_certificate_date(metadata: dict[str, Any]) -> tuple[str | None, list[str], dict[str, Any]]:
    tokens: list[str] = []
    trace: dict[str, Any] = {"step": "certificate_date"}

    labeled_entries = _collect_labeled_dates(metadata)
    eligible: list[tuple[int, str, str, str]] = []
    for raw_label, normalized_label, value in labeled_entries:
        if _is_excluded_date_label(raw_label):
            continue
        priority = next(
            (index for index, preferred in enumerate(_CERTIFICATE_DATE_PRIORITY) if preferred == normalized_label),
            None,
        )
        if priority is not None:
            eligible.append((priority, raw_label, normalized_label, value))

    if eligible:
        eligible.sort(key=lambda item: item[0])
        _, raw_label, normalized_label, value = eligible[0]
        normalized_value = _normalize_date_value(value)
        metadata["certificate_date"] = normalized_value
        tokens.append(f"certificate_date:from_label:{normalized_label}")
        trace.update(
            {
                "source": "labeled_dates",
                "selected_label": raw_label,
                "normalized_label": normalized_label,
                "value": normalized_value,
            }
        )
        return normalized_value, tokens, trace

    existing = _text(metadata.get("certificate_date"))
    if existing:
        normalized_value = _normalize_date_value(existing)
        metadata["certificate_date"] = normalized_value
        tokens.append("certificate_date:from_metadata")
        trace.update({"source": "certificate_date", "value": normalized_value})
        return normalized_value, tokens, trace

    trace["source"] = None
    return None, tokens, trace


def _grade_source_priority(normalized_key: str) -> int | None:
    for index, preferred in enumerate(_GRADE_SOURCE_PRIORITY):
        if normalized_key == preferred or preferred in normalized_key:
            return index
    return None


def _is_spec_line(value: str) -> bool:
    return bool(_SPEC_LINE_RE.search(value))


def _is_short_grade(value: str) -> bool:
    return bool(_SHORT_GRADE_RE.match(_text(value)))


def _cleanup_grade_detail_text(grade: str) -> str:
    text = _text(grade)
    if not text:
        return text
    earliest = len(text)
    for pattern in _GRADE_DETAIL_SEPARATOR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < earliest:
            earliest = match.start()
    if earliest < len(text):
        return text[:earliest].strip(" -/\t")
    return text


def _collect_grade_candidates(
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []

    for key, value in row.items():
        if key.startswith("_") or not _text(value):
            continue
        normalized_key = normalize_header(str(key))
        priority = _grade_source_priority(normalized_key)
        if priority is not None:
            candidates.append((priority, str(key), _text(value)))

    for meta_key in _METADATA_GRADE_KEYS:
        value = _text(metadata.get(meta_key))
        if value:
            candidates.append((0 if meta_key == "header_grade" else 1, meta_key, value))

    current_grade = _text(row.get("grade"))
    if current_grade:
        candidates.append((len(_GRADE_SOURCE_PRIORITY), "grade", current_grade))

    return candidates


def _select_enriched_grade(
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    current = _text(row.get("grade"))
    if not current or not _is_short_grade(current):
        return None, None

    candidates = _collect_grade_candidates(row, metadata)
    product_material: list[tuple[int, str, str]] = []
    spec_only: list[tuple[int, str, str]] = []

    short_upper = current.upper()
    for priority, source, value in candidates:
        if source == "grade" and value.upper() == short_upper:
            continue
        if short_upper not in value.upper():
            continue
        if len(value) <= len(current):
            continue
        if _is_spec_line(value):
            spec_only.append((priority, source, value))
        else:
            product_material.append((priority, source, value))

    pool = product_material or spec_only
    if not pool:
        return None, None

    pool.sort(key=lambda item: (-len(item[2]), item[0]))
    _, source, value = pool[0]
    return value, {"from": source, "previous": current, "selected": value}


def _resolve_item_id_from_aliases(item: dict[str, Any]) -> str | None:
    for key in _ITEM_ID_ALIASES:
        value = item.get(key)
        if not _is_placeholder_item_id(value):
            return _text(value)
    return None


def _set_canonical_item_id(item: dict[str, Any], item_id: str) -> None:
    item["item_id"] = item_id
    for alias in _ITEM_ID_ALIASES:
        if alias == "item_id":
            continue
        if alias in item and _is_placeholder_item_id(item.get(alias)):
            item[alias] = item_id


def _assign_sequential_item_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    assigned = False
    next_id = 1
    for row in rows:
        existing = _resolve_item_id_from_aliases(row)
        if existing:
            _set_canonical_item_id(row, existing)
        else:
            row["item_id"] = str(next_id)
            _set_canonical_item_id(row, str(next_id))
            assigned = True
        next_id += 1
    return rows, assigned


def _canonical_item_field(item: dict[str, Any], field_name: str) -> Any:
    if field_name in _MECHANICAL_AUTO_ACCEPT_FIELDS:
        mechanical = item.get("mechanical_properties")
        if isinstance(mechanical, dict):
            return mechanical.get(field_name)
    return item.get(field_name)


def _canonical_field_present(item: dict[str, Any], field_name: str) -> bool:
    value = _canonical_item_field(item, field_name)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_text(value))
    return True


def compute_auto_accept_missing_fields_rate(result: dict[str, Any]) -> float:
    items = result.get("items")
    if not isinstance(items, list) or not items:
        return 1.0

    missing = 0
    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        for field_name in _AUTO_ACCEPT_CRITICAL_FIELDS:
            total += 1
            if not _canonical_field_present(item, field_name):
                missing += 1
    return missing / max(total, 1)


def finalize_extraction_fields(
    metadata: dict[str, Any],
    items_dicts: list[dict[str, Any]],
) -> FinalizationResult:
    """Apply certificate date, grade, and item-id finalization before validation."""

    meta = dict(metadata) if isinstance(metadata, dict) else {}
    rows = [dict(row) for row in items_dicts if isinstance(row, dict)]
    tokens: list[str] = []
    traces: list[dict[str, Any]] = []

    _, date_tokens, date_trace = _resolve_certificate_date(meta)
    tokens.extend(date_tokens)
    traces.append(date_trace)

    identity = apply_identity_field_mapping(rows, metadata=meta)
    rows = identity.rows
    tokens.extend(identity.tokens)
    traces.extend(identity.traces)

    for index, row in enumerate(rows):
        if not _text(row.get("grade")):
            continue
        cleaned = _cleanup_grade_detail_text(_text(row.get("grade")))
        if cleaned != _text(row.get("grade")):
            row["grade"] = cleaned or None
            tokens.append("grade_trimmed:detail_suffix")
            traces.append({"step": "grade_trim", "row_index": index, "selected": cleaned})

    rows, assigned_ids = _assign_sequential_item_ids(rows)
    if assigned_ids:
        tokens.append("item_id_assigned:sequential")

    labeled_tokens, labeled_traces = apply_labeled_identifiers(meta, rows)
    tokens.extend(labeled_tokens)
    traces.extend(labeled_traces)

    return FinalizationResult(metadata=meta, rows=rows, tokens=tokens, traces=traces)


def finalize_canonical_response(result: dict[str, Any]) -> CanonicalFinalizationResult:
    """Finalize the canonical API/pipeline response dict before confidence/review."""

    payload = dict(result) if isinstance(result, dict) else {}
    items = payload.get("items")
    if not isinstance(items, list):
        items = []
    tokens: list[str] = []
    traces: list[dict[str, Any]] = []

    canonical_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        row = dict(item)

        identity = apply_identity_field_mapping([row], metadata=payload)
        row = identity.rows[0] if identity.rows else row
        tokens.extend(identity.tokens)
        traces.extend(identity.traces)
        if _text(row.get("grade")):
            cleaned = _cleanup_grade_detail_text(_text(row.get("grade")))
            if cleaned != _text(row.get("grade")):
                row["grade"] = cleaned or None
                tokens.append("grade_trimmed:detail_suffix")

        existing = _resolve_item_id_from_aliases(row)
        if existing:
            _set_canonical_item_id(row, existing)
        canonical_items.append(row)

    assigned = False
    next_id = 1
    for row in canonical_items:
        existing = _resolve_item_id_from_aliases(row)
        if existing:
            _set_canonical_item_id(row, existing)
        else:
            _set_canonical_item_id(row, str(next_id))
            assigned = True
        next_id += 1
    if assigned:
        tokens.append("item_id_assigned:sequential")

    payload["items"] = canonical_items
    missing_rate = compute_auto_accept_missing_fields_rate(payload)
    payload["missing_critical_fields_rate"] = missing_rate
    result.clear()
    result.update(payload)

    return CanonicalFinalizationResult(
        tokens=tokens,
        traces=traces,
        missing_critical_fields_rate=missing_rate,
    )


def _result_payload(obj: UniversalDocumentExtraction | dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return obj.model_dump(mode="python")


def _row_has_blocking_deviations_from_dict(item: dict[str, Any]) -> bool:
    validation = item.get("validation")
    if not isinstance(validation, dict):
        return True
    if validation.get("is_compliant") is not True:
        return True
    for deviation in validation.get("deviations") or []:
        lowered = str(deviation).lower()
        if any(marker in lowered for marker in _BLOCKING_DEVIATION_MARKERS):
            return True
    return False


def _is_hard_auto_accept_blocker(reason: str) -> bool:
    text = _text(reason)
    if not text:
        return False
    if text in _HARD_AUTO_ACCEPT_BLOCKER_EXACT:
        return True
    return any(text.startswith(prefix) for prefix in _HARD_AUTO_ACCEPT_BLOCKER_PREFIXES)


def _severe_scan_in_payload(result: dict[str, Any]) -> bool:
    for profile in (result.get("document_profile"),):
        if isinstance(profile, dict):
            bucket = _text(profile.get("quality_bucket") or profile.get("quality_class")).lower()
            if bucket == "severe_scan":
                return True
    explanation = result.get("explanation")
    if isinstance(explanation, dict):
        profile = explanation.get("document_profile")
        if isinstance(profile, dict):
            bucket = _text(profile.get("quality_bucket") or profile.get("quality_class")).lower()
            if bucket == "severe_scan":
                return True
    return False


def _iter_payload_reason_strings(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("review_reasons", "structured_reasons", "blocking_reasons", "blocking_errors"):
        values = payload.get(key)
        if isinstance(values, list):
            reasons.extend(str(value) for value in values)
    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        review_policy = explanation.get("review_policy")
        if isinstance(review_policy, dict):
            for key in (
                "review_reasons",
                "structured_reasons",
                "blocking_reasons",
                "all_reasons",
                "evidence_gaps",
            ):
                values = review_policy.get(key)
                if isinstance(values, list):
                    reasons.extend(str(value) for value in values)
        validation_outcome = explanation.get("validation_outcome")
        if isinstance(validation_outcome, dict):
            values = validation_outcome.get("review_reasons")
            if isinstance(values, list):
                reasons.extend(str(value) for value in values)
    return reasons


def _payload_has_hard_auto_accept_blockers(result: dict[str, Any]) -> bool:
    if _severe_scan_in_payload(result):
        return True
    return any(_is_hard_auto_accept_blocker(reason) for reason in _iter_payload_reason_strings(result))


def _has_blocking_review_reasons(review_reasons: list[str]) -> bool:
    return any(_is_hard_auto_accept_blocker(reason) for reason in review_reasons)


def _is_confidence_only_reason(reason: str) -> bool:
    normalized = _text(reason).lower()
    if normalized in _CONFIDENCE_ONLY_REASONS:
        return True
    return normalized.startswith("low_confidence:")


def _clean_confidence_reasons(reasons: Any) -> list[str]:
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons if not _is_confidence_only_reason(str(reason))]


def _document_review_reasons_are_confidence_only(result: dict[str, Any]) -> bool:
    reasons = list(result.get("review_reasons") or [])
    if not reasons:
        return bool(result.get("needs_review"))
    return all(_is_confidence_only_reason(str(reason)) for reason in reasons)


def _traceability_verified(result: dict[str, Any]) -> bool:
    if result.get("traceability_status") == TRACEABILITY_VERIFIED:
        return True
    items = result.get("items")
    if not isinstance(items, list) or not items:
        return False
    return all(
        isinstance(item, dict) and item.get("traceability_status") == TRACEABILITY_VERIFIED
        for item in items
    )


def _entry_has_traceability_identifier(entry: dict[str, Any]) -> bool:
    return any(_text(entry.get(field)) for field in _TRACEABILITY_IDENTIFIER_PRIORITY)


def _document_has_traceability_identifier(result: dict[str, Any]) -> bool:
    if _entry_has_traceability_identifier(result):
        return True
    items = result.get("items")
    if not isinstance(items, list):
        return False
    return any(isinstance(item, dict) and _entry_has_traceability_identifier(item) for item in items)


def _primary_traceability_value(entry: dict[str, Any]) -> str | None:
    for field in _TRACEABILITY_IDENTIFIER_PRIORITY:
        value = _text(entry.get(field))
        if value:
            return value
    return None


def _alias_source_field_and_value(entry: dict[str, Any]) -> tuple[str, str] | None:
    """Return the non-heat identifier field/value used to populate heat_number."""

    if _text(entry.get("heat_number")):
        return None
    for field in _TRACEABILITY_IDENTIFIER_PRIORITY:
        if field == "heat_number":
            continue
        value = _text(entry.get(field))
        if value:
            return field, value
    return None


@dataclass(frozen=True)
class HeatAliasInfo:
    entry: dict[str, Any]
    source_field: str
    value: str


def _apply_heat_number_alias(entry: dict[str, Any]) -> HeatAliasInfo | None:
    source = _alias_source_field_and_value(entry)
    if source is None:
        return None
    source_field, value = source
    entry["heat_number"] = value
    return HeatAliasInfo(entry=entry, source_field=source_field, value=value)


def _apply_heat_number_aliases(result: dict[str, Any]) -> list[HeatAliasInfo]:
    alias_infos: list[HeatAliasInfo] = []
    info = _apply_heat_number_alias(result)
    if info is not None:
        alias_infos.append(info)

    items = result.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            info = _apply_heat_number_alias(item)
            if info is not None:
                alias_infos.append(info)

    if not _text(result.get("heat_number")) and isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and _text(item.get("heat_number")):
                result["heat_number"] = item["heat_number"]
                break

    return alias_infos


def _item_validation_compliant(item: dict[str, Any]) -> bool:
    validation = item.get("validation")
    if not isinstance(validation, dict):
        return False
    if validation.get("is_compliant") is True:
        return True
    return _text(validation.get("outcome")).upper() == "COMPLIANT"


def _document_is_compliant(result: dict[str, Any]) -> bool:
    if result.get("is_compliant") is True:
        return True
    if _text(result.get("outcome")).upper() == "COMPLIANT":
        return True
    items = result.get("items")
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if item.get("needs_review"):
            return False
        if not _item_validation_compliant(item):
            return False
    return True


def _no_validation_deviations(result: dict[str, Any]) -> bool:
    items = result.get("items")
    if not isinstance(items, list):
        return True
    for item in items:
        if not isinstance(item, dict):
            continue
        validation = item.get("validation")
        if isinstance(validation, dict) and validation.get("deviations"):
            return False
    return True


def _all_material_fields_present(result: dict[str, Any]) -> bool:
    items = result.get("items")
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        for field_name in _AUTO_ACCEPT_CRITICAL_FIELDS:
            if not _canonical_field_present(item, field_name):
                return False
    return True


def eligible_for_confidence_exempt_material(result: dict[str, Any]) -> bool:
    if not _document_is_compliant(result):
        return False
    if not _no_validation_deviations(result):
        return False
    if not _traceability_verified(result):
        return False
    if not _document_has_traceability_identifier(result):
        return False
    if not _all_material_fields_present(result):
        return False
    return True


def eligible_for_confidence_exempt_reconcile(result: dict[str, Any]) -> bool:
    if _payload_has_hard_auto_accept_blockers(result):
        return False
    if not eligible_for_confidence_exempt_material(result):
        return False
    if not _document_review_reasons_are_confidence_only(result):
        return False
    filtered = _clean_confidence_reasons(result.get("review_reasons"))
    if _has_blocking_review_reasons(filtered):
        return False
    return True


def _ensure_review_policy_matches_auto_accept(result: dict[str, Any]) -> None:
    """Align nested explanation.review_policy when review flags are cleared."""

    if _payload_has_hard_auto_accept_blockers(result):
        return
    if result.get("needs_review") is not False and result.get("review_required") is not False:
        return

    explanation = result.get("explanation")
    if not isinstance(explanation, dict):
        return

    review_policy = explanation.get("review_policy")
    if not isinstance(review_policy, dict):
        review_policy = {}
        explanation["review_policy"] = review_policy

    for key in ("review_reasons", "structured_reasons", "evidence_gaps", "all_reasons"):
        if key in review_policy:
            review_policy[key] = _clean_confidence_reasons(review_policy[key])

    review_policy["decision"] = "auto_accept"
    review_policy["review_required"] = False
    review_policy["needs_review"] = False

    validation_outcome = explanation.get("validation_outcome")
    if isinstance(validation_outcome, dict):
        if "review_reasons" in validation_outcome:
            validation_outcome["review_reasons"] = _clean_confidence_reasons(
                validation_outcome["review_reasons"]
            )
        validation_outcome["review_required"] = False


def _update_explanation_for_auto_accept(explanation: dict[str, Any]) -> None:
    _ensure_review_policy_matches_auto_accept({"explanation": explanation, "needs_review": False})


def _default_finalize_profile() -> Any:
    from app.services.document_profiler import DocumentProfile

    return DocumentProfile(
        document_id="finalize",
        filename="document.pdf",
        page_count=1,
        has_text_layer=True,
        text_density=0.5,
        blur_score=300.0,
        noise_score=5.0,
        table_presence_hint=True,
        quality_class="digital_clean",
    )


def _attach_auto_accept_evidence(payload: dict[str, Any]) -> None:
    if payload.get("processing_decision") != "auto_accept" and payload.get("status") != "AUTO_ACCEPT":
        return
    if isinstance(payload.get("auto_accept_evidence"), dict):
        return
    from app.config import settings
    from app.services.review_policy import _build_auto_accept_evidence

    enriched = dict(payload)
    enriched.setdefault("supplier_name", "Unknown")
    enriched.setdefault("total_items_detected", len(enriched.get("items") or []))
    enriched.setdefault("items", [])

    try:
        extraction = UniversalDocumentExtraction.model_validate(enriched)
    except Exception:
        return

    review_reasons = list(payload.get("review_reasons") or [])
    confidence_only = [r for r in review_reasons if r in _CONFIDENCE_ONLY_REASONS]
    final_confidence = float(payload.get("confidence_score") or 0.0)
    evidence = _build_auto_accept_evidence(
        extraction=extraction,
        profile=_default_finalize_profile(),
        confidence_summary=dict(payload.get("confidence_breakdown") or {"_overall": final_confidence}),
        blocking_reasons=[],
        confidence_only_reasons=confidence_only,
        all_reasons=review_reasons,
        confidence_exempt=True,
        confidence_threshold=settings.review_confidence_threshold,
        final_confidence=final_confidence,
    )
    if evidence is None:
        return
    evidence["gates_passed"] = list(evidence.get("satisfied_criteria") or [])
    payload["auto_accept_evidence"] = evidence


def _apply_auto_accept_document_state(result: dict[str, Any]) -> None:
    result["review_reasons"] = _clean_confidence_reasons(result.get("review_reasons"))
    result["needs_review"] = False
    result["review_required"] = False
    result["status"] = "AUTO_ACCEPT"
    result["processing_decision"] = "auto_accept"
    result["is_compliant"] = True
    result["outcome"] = "COMPLIANT"
    result["compliance_status"] = "COMPLIANT"

    explanation = result.get("explanation")
    if isinstance(explanation, dict):
        _update_explanation_for_auto_accept(explanation)
    _attach_auto_accept_evidence(result)


def reconcile_final_document_decision(result: dict[str, Any]) -> dict[str, Any]:
    """Final document-level reconciliation after review policy and confidence."""

    if not isinstance(result, dict):
        return result

    alias_infos = _apply_heat_number_aliases(result)
    reconciled = False
    if (
        not _payload_has_hard_auto_accept_blockers(result)
        and eligible_for_confidence_exempt_reconcile(result)
    ):
        _apply_auto_accept_document_state(result)
        reconciled = True

    trace = result.get("extraction_finalization")
    if not isinstance(trace, dict):
        trace = {}
    reconcile_tokens: list[str] = list(trace.get("reconcile_tokens") or [])
    if alias_infos:
        reconcile_tokens.append("heat_number:aliased_from_traceability")
    if reconciled:
        reconcile_tokens.append("document:auto_accept_reconciled")
    if reconcile_tokens:
        trace["reconcile_tokens"] = reconcile_tokens
        result["extraction_finalization"] = trace

    preferred_alias = next(
        (info for info in alias_infos if info.entry in (result.get("items") or [])),
        alias_infos[0] if alias_infos else None,
    )
    apply_traceability_remarks(
        result,
        heat_number_aliased=bool(alias_infos),
        heat_alias_source_field=preferred_alias.source_field if preferred_alias else None,
    )

    _ensure_review_policy_matches_auto_accept(result)

    return result


def _hydrate_extraction_items_from_payload(
    extraction: UniversalDocumentExtraction,
    payload: dict[str, Any],
) -> None:
    items_payload = payload.get("items")
    if not isinstance(items_payload, list):
        return
    for index, item in enumerate(extraction.items):
        if index >= len(items_payload) or not isinstance(items_payload[index], dict):
            continue
        canon = items_payload[index]
        item.item_id = canon.get("item_id")
        item.pipe_id = canon.get("pipe_id")
        item.heat_number = canon.get("heat_number")
        item.batch_number = canon.get("batch_number")
        item.product_name = canon.get("product_name")
        item.grade = canon.get("grade")
        item.weight_or_length = canon.get("weight_or_length")
        item.dimensions = canon.get("dimensions")
        if isinstance(canon.get("standards"), list):
            item.standards = canon.get("standards")


def apply_reconcile_to_extraction(extraction: UniversalDocumentExtraction) -> dict[str, Any]:
    payload = extraction.model_dump(mode="python")
    if extraction.explanation:
        payload["explanation"] = dict(extraction.explanation)
    reconcile_final_document_decision(payload)
    _hydrate_extraction_items_from_payload(extraction, payload)
    _hydrate_extraction_review_from_payload(extraction, payload)
    extraction.is_compliant = payload.get("is_compliant", extraction.is_compliant)
    extraction.ai_analysis_remarks = payload.get("ai_analysis_remarks", extraction.ai_analysis_remarks)
    if payload.get("explanation"):
        extraction.explanation = payload["explanation"]
    return payload


def qualifies_for_confidence_exempt_auto_accept(
    extraction: UniversalDocumentExtraction | dict[str, Any],
) -> bool:
    """True when auto-accept critical fields, validation, and traceability pass."""

    return eligible_for_confidence_exempt_material(_result_payload(extraction))


def _apply_confidence_exempt_decision(payload: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    if _payload_has_hard_auto_accept_blockers(payload):
        review_reasons = list(payload.get("review_reasons") or [])
        remaining = _clean_confidence_reasons(review_reasons)
        if remaining != review_reasons:
            payload["review_reasons"] = remaining
        return tokens

    if not eligible_for_confidence_exempt_reconcile(payload):
        return tokens

    review_reasons = list(payload.get("review_reasons") or [])
    remaining = _clean_confidence_reasons(review_reasons)
    if remaining != review_reasons:
        payload["review_reasons"] = remaining
        tokens.append("auto_accept:confidence_exempt")

    if not remaining:
        payload["needs_review"] = False
        payload["review_required"] = False
        payload["processing_decision"] = "auto_accept"
        payload["compliance_status"] = "COMPLIANT"
        payload["is_compliant"] = True
        payload["outcome"] = "COMPLIANT"
        if payload.get("status") in {None, "NEEDS_REVIEW", "COMPLETED"}:
            payload["status"] = "AUTO_ACCEPT"
        tokens.append("auto_accept:decision_calibrated")
        _ensure_review_policy_matches_auto_accept(payload)
        _attach_auto_accept_evidence(payload)

    return tokens


def _hydrate_extraction_review_from_payload(
    extraction: UniversalDocumentExtraction,
    payload: dict[str, Any],
) -> None:
    extraction.needs_review = bool(payload.get("needs_review", extraction.needs_review))
    extraction.status = payload.get("status", extraction.status)
    extraction.outcome = payload.get("outcome", extraction.outcome)
    if "review_reasons" in payload:
        extraction.review_reasons = list(payload["review_reasons"])
    if payload.get("auto_accept_evidence") is not None:
        extraction.auto_accept_evidence = dict(payload["auto_accept_evidence"])


def apply_canonical_finalization_to_extraction(
    extraction: UniversalDocumentExtraction,
) -> CanonicalFinalizationResult:
    payload = extraction.model_dump(mode="python")
    result = finalize_canonical_response(payload)
    _hydrate_extraction_items_from_payload(extraction, payload)
    return result


def finalize_decision(extraction: UniversalDocumentExtraction) -> list[str]:
    """Calibrate auto-accept when confidence alone would block a clean extraction."""

    payload = extraction.model_dump(mode="python")
    tokens = _apply_confidence_exempt_decision(payload)
    _hydrate_extraction_review_from_payload(extraction, payload)
    return tokens


def finalize_decision_on_result(result: dict[str, Any]) -> list[str]:
    """Final safety calibration on canonical API response dict."""

    if not isinstance(result, dict):
        return []
    tokens = _apply_confidence_exempt_decision(result)
    reconcile_final_document_decision(result)
    _attach_auto_accept_evidence(result)
    return tokens


__all__ = [
    "AUTO_ACCEPT_CRITICAL_FIELDS",
    "CanonicalFinalizationResult",
    "FinalizationResult",
    "apply_canonical_finalization_to_extraction",
    "apply_reconcile_to_extraction",
    "compute_auto_accept_missing_fields_rate",
    "eligible_for_confidence_exempt_material",
    "eligible_for_confidence_exempt_reconcile",
    "finalize_canonical_response",
    "finalize_decision",
    "finalize_decision_on_result",
    "finalize_extraction_fields",
    "qualifies_for_confidence_exempt_auto_accept",
    "reconcile_final_document_decision",
]

# Public alias for tests and downstream imports.
AUTO_ACCEPT_CRITICAL_FIELDS = _AUTO_ACCEPT_CRITICAL_FIELDS
