"""Deterministic review-gating policy.

:func:`apply_review_policy` is the single auditable place that decides whether
an extraction result needs human review. It runs after the validator and the
confidence normaliser and appends a set of *structured* review reason tokens
on top of whatever those components have already recorded.

Token conventions
-----------------
Tokens are ``key:value`` strings so downstream tooling can parse them
programmatically. All existing free-form reasons (from the validator /
confidence normaliser) are preserved for backward compatibility — the
structured tokens are *additive*.

Canonical tokens:

- ``missing_critical_field:<field>`` — a critical field is missing across all
  rows (``yield_strength``, ``tensile_strength``, ``heat_number``, ``grade``).
- Document quality is retained as diagnostic metadata, but it is not a
  standalone review gate.
- ``low_confidence:<field>`` — final confidence is low AND a specific critical
  numeric field is suspicious or missing.
- ``validation_conflict:<reason>`` — validator reported deviations
  (non-compliant rows, suspicious numbers, heat pattern inconsistencies).
- ``row_count_inconsistent`` — reported vs. extracted item count mismatch.
- ``no_items_extracted`` — zero items parsed.
- ``table_found_but_no_rows`` — table geometry detected but rows empty.
- ``confidence_below_threshold`` — final confidence < review threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.domain.outcome_taxonomy import (
    EXPLICIT_UNMAPPED_GRADE,
    MISSING_CRITICAL_FIELD_GRADE,
    NEEDS_REVIEW,
    NON_COMPLIANT,
    UNSUPPORTED_SPEC_FAMILY,
    UNRESOLVED_SPEC,
)
from app.domain.validation_config import get_validation_config
from app.schemas.extraction import UniversalDocumentExtraction
from app.services.confidence import extraction_audit_excuses_missing_field
from app.services.document_profiler import DocumentProfile
from app.services.extraction_finalizer import qualifies_for_confidence_exempt_auto_accept

CRITICAL_STRING_FIELDS = ("heat_number", "grade")
CRITICAL_NUMERIC_FIELDS = ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")
REQUIRED_CRITICAL_FIELDS = (
    "heat_number",
    "grade",
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "elongation_percentage",
)
SUPPORTED_DOCUMENT_TYPES = {
    "certificate of analysis",
    "coa",
    "mill test certificate",
    "mill test report",
    "material test certificate",
    "mtc",
    "inspection certificate",
    "test report",
    "test certificate",
    "certificate of quality",
    "certificate of conformity",
    "material certificate",
    "3.1 certificate",
    "en 10204",
}
TOKEN_FIELD_LABELS = {
    "yield_strength_mpa": "yield_strength",
    "tensile_strength_mpa": "tensile_strength",
    "heat_number": "heat_number",
    "grade": "grade",
    "elongation_percentage": "elongation",
}
SOFT_AUDIT_REVIEW_REASONS = frozenset(
    {
        "visual ambiguity detected in row",
        "blurry/noisy document",
        "ocr text layer corrupted",
        "low identifier legibility",
        "low-confidence rows detected",
    }
)
SEVERE_SCAN_QUALITY_TOKEN = "quality_blocker:severe_scan"
AUTO_ACCEPT_POLICY_VERSION = "auto_accept_safety_v1"
_AUTO_ACCEPT_REQUIRED_SATISFIED_CRITERIA = frozenset(
    {
        "critical_fields_present",
        "traceability_verified",
        "row_alignment_verified",
        "validation_passed",
        "document_quality_acceptable",
        "no_blocking_reasons",
        "confidence_acceptable",
        "supported_document_type",
    }
)
CONFIDENCE_ONLY_REASONS = frozenset(
    {
        "confidence falls below threshold",
        "confidence_below_threshold",
        "low_confidence",
        "overall_decision_confidence_below_threshold",
    }
)
AUTO_ACCEPT_BLOCKING_EXACT = frozenset(
    {
        SEVERE_SCAN_QUALITY_TOKEN,
        "unresolved_grade",
        "ambiguous_grade",
        "explicit_unmapped_grade",
        "unsupported_spec_family",
        "unresolved_spec",
        "mechanical_table_alignment_uncertain",
        "traceability_identifier_ocr_uncertain",
        "traceability_identifier_conflict",
        "traceability_unverified",
        "critical_identifier_unverified",
        "missing_critical_identifier_group",
        "validation_conflict:row_non_compliant",
        "no_items_extracted",
        "extraction structure is incomplete",
        "suspicious_duplication:heat_and_item_id",
        "validation_conflict:confusable_heat_numbers",
        "row_count_inconsistent",
        "unsupported_document_type",
        "validation_blocking_error",
        "extraction_schema_invalid",
        "model_output_unusable",
    }
)
AUTO_ACCEPT_BLOCKING_PREFIXES = (
    "missing_critical_field:",
    "validation_conflict:",
    "critical_identifier_unverified",
    "header_row_conflict:",
)
TRACEABILITY_IDENTIFIER_GROUP_FIELDS = (
    "heat_number",
    "batch_number",
    "lot_number",
    "colata_number",
    "cast_number",
    "charge_number",
    "coil_number",
    "item_id",
    "pipe_id",
    "traceability_identifier_value",
)


@dataclass
class ReviewDecision:
    review_required: bool
    structured_reasons: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    confidence_only_reasons: list[str] = field(default_factory=list)
    soft_reasons: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    reviewer_focus: list[str] = field(default_factory=list)
    all_reasons: list[str] = field(default_factory=list)
    decision: str | None = None
    confidence_summary: dict[str, float] = field(default_factory=dict)
    auto_accept_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        decision = self.decision or ("review_required" if self.review_required else "auto_accept")
        payload: dict[str, Any] = {
            "decision": decision,
            "review_required": self.review_required,
            "needs_review": self.review_required,
            "structured_reasons": list(self.structured_reasons),
            "review_reasons": list(self.structured_reasons),
            "blocking_errors": list(self.blocking_reasons),
            "confidence_summary": dict(self.confidence_summary),
            "blocking_reasons": list(self.blocking_reasons),
            "confidence_only_reasons": list(self.confidence_only_reasons),
            "soft_reasons": list(self.soft_reasons),
            "evidence_gaps": list(self.evidence_gaps),
            "recommended_reviewer_focus": list(self.reviewer_focus),
            "all_reasons": list(self.all_reasons),
        }
        if decision == "auto_accept" and self.auto_accept_evidence is not None:
            payload["auto_accept_evidence"] = dict(self.auto_accept_evidence)
        return payload


def _normalise_token(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _document_type_supported(value: Any) -> bool:
    normalised = _normalise_token(value)
    if not normalised or normalised == "unknown document":
        return False
    return any(known in normalised for known in SUPPORTED_DOCUMENT_TYPES)


def _profile_value(document_profile: Any, *keys: str) -> Any:
    if document_profile is None:
        return None
    if isinstance(document_profile, dict):
        for key in keys:
            if key in document_profile:
                return document_profile.get(key)
        return None
    for key in keys:
        if hasattr(document_profile, key):
            return getattr(document_profile, key)
    return None


def _quality_bucket(document_profile: Any) -> str:
    return _normalise_token(_profile_value(document_profile, "quality_bucket", "quality_class"))


def is_severe_scan_profile(document_profile: Any) -> bool:
    return _quality_bucket(document_profile) == "severe_scan"


def is_confidence_only_reason(reason: str) -> bool:
    normalized = _normalise_token(reason)
    if normalized in CONFIDENCE_ONLY_REASONS:
        return True
    return normalized.startswith("low_confidence:")


def is_auto_accept_blocking_reason(reason: str) -> bool:
    if is_confidence_only_reason(reason):
        return False
    if _is_soft_audit_review_reason(reason):
        return False
    if reason in AUTO_ACCEPT_BLOCKING_EXACT:
        return True
    return any(reason.startswith(prefix) for prefix in AUTO_ACCEPT_BLOCKING_PREFIXES)


def _iter_payload_review_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("review_reasons", "structured_reasons", "blocking_reasons", "blocking_errors"):
        values = payload.get(key)
        if isinstance(values, list):
            reasons.extend(str(value) for value in values)
    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        review_policy = explanation.get("review_policy")
        if isinstance(review_policy, dict):
            for key in ("review_reasons", "structured_reasons", "blocking_reasons", "all_reasons"):
                values = review_policy.get(key)
                if isinstance(values, list):
                    reasons.extend(str(value) for value in values)
    return reasons


def collect_auto_accept_blockers_from_payload(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    explanation = payload.get("explanation")
    document_profile = payload.get("document_profile")
    if isinstance(explanation, dict):
        profile = explanation.get("document_profile")
        if profile is not None:
            document_profile = profile
    if is_severe_scan_profile(document_profile):
        blockers.append(SEVERE_SCAN_QUALITY_TOKEN)
    for reason in _iter_payload_review_reasons(payload):
        if is_auto_accept_blocking_reason(reason):
            blockers.append(reason)
    return _dedupe_preserving_order(blockers)


def _collect_auto_accept_blockers(
    *,
    extracted_json: dict[str, Any],
    document_profile: Any,
    review_reasons: Iterable[str],
    structured: Iterable[str],
) -> list[str]:
    blockers: list[str] = []
    if is_severe_scan_profile(document_profile):
        blockers.append(SEVERE_SCAN_QUALITY_TOKEN)
    for reason in [*structured, *review_reasons, *extracted_json.get("review_reasons", [])]:
        if is_auto_accept_blocking_reason(str(reason)):
            blockers.append(str(reason))
    return _dedupe_preserving_order(blockers)


def _classify_review_reasons(reasons: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    blocking: list[str] = []
    confidence_only: list[str] = []
    soft: list[str] = []
    for reason in reasons:
        text = str(reason)
        if is_confidence_only_reason(text):
            confidence_only.append(text)
        elif _is_soft_audit_review_reason(text):
            soft.append(text)
        elif is_auto_accept_blocking_reason(text):
            blocking.append(text)
    return (
        _dedupe_preserving_order(blocking),
        _dedupe_preserving_order(confidence_only),
        _dedupe_preserving_order(soft),
    )


def _auto_accept_critical_fields_present(extraction: UniversalDocumentExtraction) -> dict[str, bool]:
    from app.services.extraction_finalizer import AUTO_ACCEPT_CRITICAL_FIELDS

    extracted_json = extraction.model_dump(mode="python")
    items = _extract_items_from_json(extracted_json)
    present: dict[str, bool] = {}
    mechanical_fields = {
        "yield_strength_mpa",
        "tensile_strength_mpa",
        "elongation_percentage",
    }
    for field_name in AUTO_ACCEPT_CRITICAL_FIELDS:
        if field_name in mechanical_fields:
            present[field_name] = any(
                isinstance(item, dict)
                and isinstance(item.get("mechanical_properties"), dict)
                and not _is_missing(item["mechanical_properties"].get(field_name))
                for item in items
            )
        else:
            present[field_name] = _critical_value_present(extracted_json, field_name)
    return present


def _assess_traceability_verified(extraction: UniversalDocumentExtraction) -> bool:
    from app.services.traceability import TRACEABILITY_VERIFIED

    if not extraction.items:
        return False
    if extraction.traceability_status != TRACEABILITY_VERIFIED:
        return False
    return all(item.traceability_status == TRACEABILITY_VERIFIED for item in extraction.items)


def _assess_row_alignment_verified(
    extraction: UniversalDocumentExtraction,
    reasons: Iterable[str],
) -> bool:
    for reason in reasons:
        text = str(reason)
        if text == "mechanical_table_alignment_uncertain":
            return False
        if text.startswith("header_row_conflict:"):
            return False
    for item in extraction.items:
        mechanical = item.mechanical_properties
        if mechanical is None:
            continue
        if any(
            getattr(mechanical, field_name, None) is not None
            for field_name in ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")
        ):
            return True
    return False


def _assess_validation_passed(extraction: UniversalDocumentExtraction) -> bool:
    if not extraction.items:
        return False
    return all(
        item.validation is not None and item.validation.is_compliant is True
        for item in extraction.items
    )


def _build_auto_accept_evidence(
    *,
    extraction: UniversalDocumentExtraction,
    profile: DocumentProfile | None,
    confidence_summary: dict[str, float],
    blocking_reasons: list[str],
    confidence_only_reasons: list[str],
    all_reasons: Iterable[str],
    confidence_exempt: bool,
    confidence_threshold: float,
    final_confidence: float,
) -> dict[str, Any] | None:
    """Build auditable auto-accept evidence or return None when criteria are unverifiable."""

    document_quality = profile.quality_class if profile is not None else None
    critical_fields_present = _auto_accept_critical_fields_present(extraction)
    traceability_verified = _assess_traceability_verified(extraction)
    row_alignment_verified = _assess_row_alignment_verified(extraction, all_reasons)
    validation_passed = _assess_validation_passed(extraction)
    supported_document_type = _document_type_supported(extraction.document_type)
    confidence_acceptable = (
        final_confidence >= confidence_threshold or confidence_exempt
    )
    document_quality_acceptable = bool(document_quality) and not is_severe_scan_profile(profile)
    no_blocking_reasons = not blocking_reasons

    satisfied_criteria: list[str] = []
    if all(critical_fields_present.values()):
        satisfied_criteria.append("critical_fields_present")
    if traceability_verified:
        satisfied_criteria.append("traceability_verified")
    if row_alignment_verified:
        satisfied_criteria.append("row_alignment_verified")
    if validation_passed:
        satisfied_criteria.append("validation_passed")
    if document_quality_acceptable:
        satisfied_criteria.append("document_quality_acceptable")
    if no_blocking_reasons:
        satisfied_criteria.append("no_blocking_reasons")
    if confidence_acceptable:
        satisfied_criteria.append("confidence_acceptable")
    if supported_document_type:
        satisfied_criteria.append("supported_document_type")

    if not _AUTO_ACCEPT_REQUIRED_SATISFIED_CRITERIA.issubset(set(satisfied_criteria)):
        return None

    return {
        "policy_version": AUTO_ACCEPT_POLICY_VERSION,
        "decision": "auto_accept",
        "document_quality": document_quality,
        "critical_fields_present": critical_fields_present,
        "traceability_verified": traceability_verified,
        "row_alignment_verified": row_alignment_verified,
        "validation_passed": validation_passed,
        "confidence_summary": dict(confidence_summary),
        "blocking_reasons": [],
        "confidence_only_reasons": list(confidence_only_reasons),
        "satisfied_criteria": satisfied_criteria,
    }


def _extract_items_from_json(extracted_json: dict[str, Any]) -> list[dict[str, Any]]:
    items = extracted_json.get("items")
    return items if isinstance(items, list) else []


def _critical_value_present(extracted_json: dict[str, Any], field_name: str) -> bool:
    if not _is_missing(extracted_json.get(field_name)):
        return True

    items = _extract_items_from_json(extracted_json)
    if field_name in {"heat_number", "grade"}:
        return any(isinstance(item, dict) and not _is_missing(item.get(field_name)) for item in items)

    for item in items:
        if not isinstance(item, dict):
            continue
        mechanical = item.get("mechanical_properties")
        if isinstance(mechanical, dict) and not _is_missing(mechanical.get(field_name)):
            return True
        if not _is_missing(item.get(field_name)):
            return True
    return False


def _traceability_value_present(entry: dict[str, Any]) -> bool:
    accepted = entry.get("accepted_identifier_values")
    if isinstance(accepted, dict):
        accepted_trace = accepted.get("traceability_identifier_value")
        if not _is_missing(accepted_trace):
            return True
        for field_name in TRACEABILITY_IDENTIFIER_GROUP_FIELDS:
            if field_name in accepted and not _is_missing(accepted.get(field_name)):
                return True
    for field_name in TRACEABILITY_IDENTIFIER_GROUP_FIELDS:
        if not _is_missing(entry.get(field_name)):
            return True
    return False


def _confidence_map(confidence: Any) -> dict[str, float]:
    """Extract field confidence values from common confidence payload shapes."""

    if confidence is None:
        return {}
    if isinstance(confidence, (float, int)):
        value = max(0.0, min(float(confidence), 1.0))
        return {"_overall": value}
    if not isinstance(confidence, dict):
        return {}

    for key in ("field_confidences", "fields", "critical_fields"):
        nested = confidence.get(key)
        if isinstance(nested, dict):
            return {
                str(field): max(0.0, min(float(value), 1.0))
                for field, value in nested.items()
                if _as_float(value) is not None
            }

    values: dict[str, float] = {}
    for field, value in confidence.items():
        parsed = _as_float(value)
        if parsed is not None:
            values[str(field)] = max(0.0, min(parsed, 1.0))
    return values


def _confidence_summary(
    extracted_json: dict[str, Any],
    confidence: Any,
    required_fields: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(confidence, dict):
        min_value = _as_float(
            confidence.get("min_critical_field_confidence")
            if "min_critical_field_confidence" in confidence
            else confidence.get("min_critical_confidence")
        )
        avg_value = _as_float(confidence.get("avg_field_confidence"))
        if min_value is not None or avg_value is not None:
            safe_min = max(0.0, min(float(min_value if min_value is not None else 0.0), 1.0))
            safe_avg = max(0.0, min(float(avg_value if avg_value is not None else safe_min), 1.0))
            return {
                "min_critical_field_confidence": round(safe_min, 4),
                "avg_field_confidence": round(safe_avg, 4),
            }

    values = _confidence_map(confidence)
    if not values:
        row_confidences = [
            _as_float(item.get("row_confidence"))
            for item in _extract_items_from_json(extracted_json)
            if isinstance(item, dict)
        ]
        valid_rows = [value for value in row_confidences if value is not None]
        if valid_rows:
            values = {"_row_confidence": sum(valid_rows) / len(valid_rows)}
        else:
            overall = _as_float(extracted_json.get("confidence_score"))
            if overall is not None:
                values = {"_overall": max(0.0, min(overall, 1.0))}

    all_values = list(values.values())
    critical_values = [
        values[field]
        for field in required_fields
        if field in values
    ]
    # Aggregate-only payloads (scalar model confidence → ``_overall``) still
    # represent document-level certainty and must gate review coherently.
    if not critical_values and "_overall" in values:
        critical_values = [values["_overall"]]
    if not critical_values and "_row_confidence" in values:
        critical_values = [values["_row_confidence"]]

    min_critical = min(critical_values) if critical_values else 0.0
    avg_field = (sum(all_values) / len(all_values)) if all_values else 0.0
    return {
        "min_critical_field_confidence": round(float(min_critical), 4),
        "avg_field_confidence": round(float(avg_field), 4),
    }


def _blocking_validation_errors(validation_errors: list[Any]) -> list[str]:
    blocking: list[str] = []
    for error in validation_errors or []:
        if isinstance(error, dict):
            tokens = [
                _normalise_token(error.get("severity")),
                _normalise_token(error.get("type")),
                _normalise_token(error.get("category")),
                _normalise_token(error.get("code")),
            ]
            is_blocking = bool(error.get("blocking")) or "blocking" in tokens or any(
                "blocking" in token for token in tokens
            )
            if is_blocking:
                blocking.append(str(error.get("code") or error.get("error") or error.get("message") or "blocking_error"))
        else:
            text = str(error)
            if "blocking" in _normalise_token(text):
                blocking.append(text)
    return _dedupe_preserving_order(blocking)


def _json_confidence_exempt_eligible(extracted_json: dict[str, Any]) -> bool:
    from app.services.extraction_finalizer import eligible_for_confidence_exempt_material

    return eligible_for_confidence_exempt_material(extracted_json)


def evaluate_review_policy(
    *,
    extracted_json: dict[str, Any],
    confidence: Any,
    validation_errors: list[Any],
    document_profile: Any,
    confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    """Apply the deterministic JSON review policy.

    The policy never mutates or auto-fills ``extracted_json``. Missing critical
    values remain missing and force ``review_required``.
    """

    review_reasons: list[str] = []
    blocking_errors = _blocking_validation_errors(validation_errors)
    product_category = extracted_json.get("product_category")
    config = get_validation_config(product_category)
    required_fields = tuple(f for f in config.mandatory_fields if f not in config.optional_fields)
    summary = _confidence_summary(extracted_json, confidence, required_fields)

    missing_fields = [
        field_name
        for field_name in required_fields
        if not _critical_value_present(extracted_json, field_name)
    ]
    review_reasons.extend(f"missing_critical_field:{field}" for field in missing_fields)

    if blocking_errors:
        review_reasons.append("validation_blocking_error")

    confidence_exempt = _json_confidence_exempt_eligible(extracted_json)
    if summary["min_critical_field_confidence"] < confidence_threshold and not confidence_exempt:
        review_reasons.append("confidence_below_threshold")

    if not _document_type_supported(extracted_json.get("document_type")):
        review_reasons.append("unsupported_document_type")

    if is_severe_scan_profile(document_profile):
        review_reasons.append(SEVERE_SCAN_QUALITY_TOKEN)

    for reason in extracted_json.get("review_reasons", []) or []:
        if is_auto_accept_blocking_reason(str(reason)):
            review_reasons.append(str(reason))

    review_reasons = _dedupe_preserving_order(review_reasons)
    auto_accept_blockers = _collect_auto_accept_blockers(
        extracted_json=extracted_json,
        document_profile=document_profile,
        review_reasons=review_reasons,
        structured=review_reasons,
    )

    validation_passes = not validation_errors
    all_critical_exist = not missing_fields
    confidence_passes = summary["min_critical_field_confidence"] >= confidence_threshold or confidence_exempt
    supported_document = _document_type_supported(extracted_json.get("document_type"))

    if (
        all_critical_exist
        and validation_passes
        and confidence_passes
        and supported_document
        and not auto_accept_blockers
    ):
        decision = "auto_accept"
    else:
        decision = "review_required"

    blocking_reasons, confidence_only_reasons, soft_reasons = _classify_review_reasons(review_reasons)

    return {
        "decision": decision,
        "review_reasons": review_reasons,
        "blocking_errors": blocking_errors,
        "confidence_summary": summary,
        "blocking_reasons": blocking_reasons,
        "confidence_only_reasons": confidence_only_reasons,
        "soft_reasons": soft_reasons,
    }


apply_deterministic_review_policy = evaluate_review_policy


def _missing_critical_fields(
    extraction: UniversalDocumentExtraction,
    preprocessing_meta: dict[str, Any] | None,
) -> list[str]:
    """Return tokens for critical fields that are missing across *all* rows."""

    if not extraction.items:
        # "No items" is handled separately; do not flag every field.
        return []

    missing_tokens: list[str] = []
    audit = (preprocessing_meta or {}).get("stage_b_extraction_audit") if preprocessing_meta else None

    config = get_validation_config(extraction.product_category)
    mandatory = [f for f in config.mandatory_fields if f not in config.optional_fields]
    string_fields = [f for f in mandatory if f in ("heat_number", "grade", "item_id")]
    numeric_fields = [f for f in mandatory if f in ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")]

    # String fields: missing on every row
    for field_name in string_fields:
        if field_name == "heat_number":
            # Heat is no longer a standalone critical token when another
            # accepted traceability-group identifier exists.
            continue
        if extraction_audit_excuses_missing_field(audit, field=field_name, items=extraction.items):
            continue
        if all(not getattr(item, field_name, None) for item in extraction.items):
            label = TOKEN_FIELD_LABELS.get(field_name, field_name)
            missing_tokens.append(f"missing_critical_field:{label}")

    # Numeric fields: missing on every row
    for field_name in numeric_fields:
        all_missing = True
        for item in extraction.items:
            mp = item.mechanical_properties
            if mp is not None and getattr(mp, field_name, None) is not None:
                all_missing = False
                break
        if all_missing:
            label = TOKEN_FIELD_LABELS.get(field_name, field_name)
            missing_tokens.append(f"missing_critical_field:{label}")

    if not any(
        _traceability_value_present(
            {
                "heat_number": getattr(item, "heat_number", None),
                "batch_number": getattr(item, "batch_number", None),
                "lot_number": getattr(item, "lot_number", None),
                "colata_number": getattr(item, "colata_number", None),
                "cast_number": getattr(item, "cast_number", None),
                "charge_number": getattr(item, "charge_number", None),
                "coil_number": getattr(item, "coil_number", None),
                "item_id": getattr(item, "item_id", None),
                "pipe_id": getattr(item, "pipe_id", None),
                "traceability_identifier_value": getattr(item, "traceability_identifier_value", None),
                "accepted_identifier_values": getattr(item, "accepted_identifier_values", {}),
            }
        )
        for item in extraction.items
    ) and not _traceability_value_present(
        {
            "batch_number": getattr(extraction, "batch_number", None),
            "lot_number": getattr(extraction, "lot_number", None),
            "colata_number": getattr(extraction, "colata_number", None),
            "cast_number": getattr(extraction, "cast_number", None),
            "charge_number": getattr(extraction, "charge_number", None),
            "coil_number": getattr(extraction, "coil_number", None),
            "traceability_identifier_value": getattr(extraction, "traceability_identifier_value", None),
            "accepted_identifier_values": getattr(extraction, "accepted_identifier_values", {}),
        }
    ):
        missing_tokens.append("missing_critical_identifier_group")

    return missing_tokens


def _validation_conflict_tokens(extraction: UniversalDocumentExtraction) -> list[str]:
    tokens: list[str] = []
    any_non_compliant = False
    any_unresolved = False
    any_ambiguous = False
    any_unknown_grade = False
    suspicious_seen = False
    heat_mismatch_seen = False
    separator_mismatch_seen = False
    grade_spec_mismatch_seen = False
    unit_missing_seen = False
    duplication_seen = False
    confusable_heat_seen = False

    for item in extraction.items:
        if item.validation is None:
            continue
        # Tri-state: only flip ``any_non_compliant`` on an explicit False,
        # never on ``None`` (which means unresolved / not applicable).
        if item.validation.is_compliant is False:
            any_non_compliant = True
        outcome = (item.validation.outcome or "").upper()
        if outcome == NEEDS_REVIEW:
            joined = " ".join(item.validation.deviations).lower()
            if "unknown grade" in joined:
                any_unknown_grade = True
            if "ambiguous grade" in joined:
                any_ambiguous = True
        elif outcome == "AMBIGUOUS_GRADE":
            any_ambiguous = True
        elif outcome == EXPLICIT_UNMAPPED_GRADE:
            tokens.append("explicit_unmapped_grade")
        elif outcome == MISSING_CRITICAL_FIELD_GRADE:
            tokens.append("missing_critical_field:grade")
        elif outcome == UNRESOLVED_SPEC:
            any_unresolved = True
        elif outcome == UNSUPPORTED_SPEC_FAMILY:
            tokens.append("unsupported_spec_family")
        for deviation in item.validation.deviations:
            lowered = deviation.lower()
            if "looks suspicious" in lowered:
                suspicious_seen = True
            if "heat number" in lowered and "inconsistent" in lowered:
                heat_mismatch_seen = True
            if "mixes separators" in lowered or "malformed" in lowered:
                separator_mismatch_seen = True
            if "inconsistent with the extracted mechanical values" in lowered:
                grade_spec_mismatch_seen = True
            if "lacks a clear unit" in lowered:
                unit_missing_seen = True
            if "field-duplication hallucination" in lowered:
                duplication_seen = True

    # Promote document-level tokens already appended by the validator.
    if "suspicious_duplication:heat_and_item_id" in extraction.review_reasons:
        duplication_seen = True
    if "validation_conflict:confusable_heat_numbers" in extraction.review_reasons:
        confusable_heat_seen = True

    if any_non_compliant:
        tokens.append("validation_conflict:row_non_compliant")
    if any_unknown_grade:
        tokens.append("unresolved_grade")
    if any_ambiguous:
        tokens.append("ambiguous_grade")
    if any_unresolved:
        tokens.append("unresolved_spec")
    if suspicious_seen:
        tokens.append("validation_conflict:suspicious_numeric_values")
    if heat_mismatch_seen:
        tokens.append("validation_conflict:heat_number_inconsistency")
    if separator_mismatch_seen:
        tokens.append("validation_conflict:malformed_numeric_strings")
    if grade_spec_mismatch_seen:
        tokens.append("validation_conflict:grade_spec_mismatch")
    if unit_missing_seen:
        tokens.append("validation_conflict:missing_unit")
    if duplication_seen:
        tokens.append("suspicious_duplication:heat_and_item_id")
    if confusable_heat_seen:
        tokens.append("validation_conflict:confusable_heat_numbers")
    return tokens


def _numeric_parser_tokens(extraction: UniversalDocumentExtraction) -> list[str]:
    """Promote pipeline-level numeric-parser tokens to structured reasons."""

    tokens: list[str] = []
    for reason in extraction.review_reasons:
        if reason.startswith("numeric_uncertain:") or reason.startswith("numeric_promoted_thousands:"):
            tokens.append(reason)
        elif reason.startswith("unresolved_grade:"):
            tokens.append("unresolved_grade")
        elif reason.startswith("ambiguous_grade:"):
            tokens.append("ambiguous_grade")
        elif reason.startswith("unresolved_spec:"):
            tokens.append("unresolved_spec")
        elif reason.startswith("header_row_conflict:"):
            tokens.append(reason)
        elif reason in {
            "unresolved_grade",
            "ambiguous_grade",
            "explicit_unmapped_grade",
            "unsupported_spec_family",
            "unresolved_spec",
            "mechanical_table_alignment_uncertain",
            "traceability_identifier_ocr_uncertain",
            "traceability_identifier_conflict",
        }:
            tokens.append(reason)
    return tokens


def _numeric_field_is_suspicious(extraction: UniversalDocumentExtraction, field_name: str) -> bool:
    needle_variants = {
        "yield_strength_mpa": "yield",
        "tensile_strength_mpa": "tensile",
        "elongation_percentage": "elongation",
    }
    needle = needle_variants.get(field_name, field_name)
    for item in extraction.items:
        if item.validation is None:
            continue
        for deviation in item.validation.deviations:
            lowered = deviation.lower()
            if "looks suspicious" in lowered and needle in lowered:
                return True
    return False


def _low_confidence_field_tokens(
    extraction: UniversalDocumentExtraction,
    profile: DocumentProfile | None,
    threshold: float,
) -> list[str]:
    """On degraded scans with low final confidence, flag specific critical fields."""

    tokens: list[str] = []
    final_confidence = extraction.confidence_score or 0.0
    if final_confidence >= threshold:
        return tokens
    if profile is not None and profile.quality_class != "noisy_scan":
        # Low-confidence-on-numeric tokens are only emitted when the document
        # itself is degraded. The generic ``confidence_below_threshold`` token
        # is emitted separately below.
        return tokens

    config = get_validation_config(extraction.product_category)
    mandatory = [f for f in config.mandatory_fields if f not in config.optional_fields]
    numeric_fields = [f for f in mandatory if f in ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")]

    for field_name in numeric_fields:
        suspicious_or_missing = _numeric_field_is_suspicious(extraction, field_name)
        if not suspicious_or_missing:
            # Check missing-everywhere as well
            all_missing = True
            for item in extraction.items:
                mp = item.mechanical_properties
                if mp is not None and getattr(mp, field_name, None) is not None:
                    all_missing = False
                    break
            suspicious_or_missing = all_missing and len(extraction.items) > 0

        if suspicious_or_missing:
            label = TOKEN_FIELD_LABELS.get(field_name, field_name)
            tokens.append(f"low_confidence:{label}")
    return tokens


def _critical_identifier_unverified_tokens(
    extraction: UniversalDocumentExtraction,
    preprocessing_meta: dict[str, Any] | None,
) -> list[str]:
    """Return review tokens for rows with suppressed or uncertain identifiers."""

    tokens: list[str] = []
    guard = (preprocessing_meta or {}).get("identifier_guard")
    if isinstance(guard, dict) and guard.get("events"):
        tokens.append("critical_identifier_unverified")
        tokens.append("traceability_unverified")
    if "critical_identifier_unverified" in extraction.review_reasons:
        tokens.append("critical_identifier_unverified")
        tokens.append("traceability_unverified")
    if "traceability_unverified" in extraction.review_reasons:
        tokens.append("traceability_unverified")
    return _dedupe_preserving_order(tokens)


def _is_soft_audit_review_reason(reason: str) -> bool:
    if reason in SOFT_AUDIT_REVIEW_REASONS:
        return True
    if reason.startswith("document_quality:"):
        return True
    lowered = _normalise_token(reason)
    return "ocr" in lowered and "corrupt" in lowered


def _is_blocking_review_reason(reason: str) -> bool:
    return is_auto_accept_blocking_reason(reason)

def apply_review_policy(
    extraction: UniversalDocumentExtraction,
    *,
    profile: DocumentProfile | None = None,
    preprocessing_meta: dict[str, Any] | None = None,
    review_confidence_threshold: float = 0.75,
) -> ReviewDecision:
    """Compute the review decision and mutate ``extraction`` in place.

    The function appends structured tokens to ``extraction.review_reasons``
    without discarding existing reasons.
    """

    # 0. Clean numeric uncertainty if within category bounds
    config = get_validation_config(extraction.product_category)
    cleaned_reasons = []
    for reason in extraction.review_reasons:
        if reason.startswith("numeric_uncertain:"):
            field = reason.split(":", 1)[1]
            band_attr = field.replace("_mpa", "").replace("_percentage", "") + "_range"
            band = getattr(config, band_attr, None)
            if band is not None:
                all_within_bounds = True
                for item in extraction.items:
                    mp = item.mechanical_properties
                    if mp is not None:
                        val = getattr(mp, field, None)
                        if val is not None and band.is_suspicious(val):
                            all_within_bounds = False
                            break
                if all_within_bounds and extraction.items:
                    continue  # Drop this reason
        cleaned_reasons.append(reason)
    extraction.review_reasons = cleaned_reasons

    structured: list[str] = []
    final_confidence = extraction.confidence_score or 0.0
    confidence_exempt = qualifies_for_confidence_exempt_auto_accept(extraction)

    # 1. Missing critical fields
    structured.extend(_missing_critical_fields(extraction, preprocessing_meta))

    # 1b. Unsupported document types can never be auto-accepted.
    if not _document_type_supported(extraction.document_type):
        structured.append("unsupported_document_type")

    # 1c. Severe scan quality is a hard blocker, never auto-accepted.
    if is_severe_scan_profile(profile):
        structured.append(SEVERE_SCAN_QUALITY_TOKEN)

    # 2. Document quality is diagnostic metadata only. Concrete evidence such
    # as missing fields, low confidence, or row extraction failure gates review.

    # 3. Validation conflicts
    structured.extend(_validation_conflict_tokens(extraction))

    # 3b. Numeric-parser + header-propagation tokens surfaced from pipeline.
    structured.extend(_numeric_parser_tokens(extraction))

    # 4. Row count / item issues
    if not extraction.items:
        structured.append("no_items_extracted")
        if preprocessing_meta is not None:
            pages = preprocessing_meta.get("pages", [])
            if any(
                isinstance(page, dict)
                and isinstance(page.get("table_detection"), dict)
                and bool(page["table_detection"].get("table_found"))
                for page in pages
            ):
                structured.append("table_found_but_no_rows")

    if extraction.total_items_detected != len(extraction.items):
        structured.append("row_count_inconsistent")

    # 5. Field-level low confidence on degraded docs
    structured.extend(
        _low_confidence_field_tokens(extraction, profile, review_confidence_threshold)
    )

    # 6. Row-level identifier uncertainty must survive later confidence
    # reconciliation. A visually uncertain heat number is not verified just
    # because the row's mechanical values passed deterministic checks.
    structured.extend(_critical_identifier_unverified_tokens(extraction, preprocessing_meta))

    # 7. Confidence below threshold (generic)
    if final_confidence < review_confidence_threshold and not confidence_exempt:
        structured.append("confidence_below_threshold")

    structured = _dedupe_preserving_order(structured)
    if confidence_exempt:
        structured = [
            reason for reason in structured if reason != "confidence_below_threshold"
        ]

    extracted_json = extraction.model_dump(mode="python")
    auto_accept_blockers = _collect_auto_accept_blockers(
        extracted_json=extracted_json,
        document_profile=profile.to_dict() if profile is not None else None,
        review_reasons=extraction.review_reasons,
        structured=structured,
    )
    blocking_reasons = list(auto_accept_blockers)

    all_compliant = bool(extraction.items) and all(
        item.validation is not None and item.validation.is_compliant is True
        for item in extraction.items
    )

    extraction_confidence_floor = max(0.90, review_confidence_threshold)
    bypass_structured_review = (
        len(structured) == 0
        and extraction.items
        and final_confidence >= extraction_confidence_floor
        and bool(all_compliant)
        and not auto_accept_blockers
        and not any(item.needs_review for item in extraction.items)
    )

    combined = _dedupe_preserving_order([*extraction.review_reasons, *structured])
    needs_review = bool(structured) or bool(extraction.needs_review) or bool(combined)

    # High-confidence + clean structured policy gate: preprocessing noise /
    # non-blocking legacy reasons alone must never force NEEDS_REVIEW.
    if bypass_structured_review:
        needs_review = False
        extraction.needs_review = False
        combined = [
            reason for reason in combined if not is_confidence_only_reason(str(reason))
        ]
    elif auto_accept_blockers:
        needs_review = True
        extraction.needs_review = True
    elif confidence_exempt and not structured and not auto_accept_blockers and all_compliant:
        needs_review = False
        extraction.needs_review = False
    else:
        hard_structured = [reason for reason in structured if not _is_soft_audit_review_reason(reason)]
        soft_audit_only = combined and all(_is_soft_audit_review_reason(reason) for reason in combined)
        if (
            soft_audit_only
            and all_compliant
            and not auto_accept_blockers
            and not hard_structured
            and not any(item.needs_review for item in extraction.items)
            and final_confidence >= review_confidence_threshold
        ):
            needs_review = False
            extraction.needs_review = False

    combined_blocking, combined_confidence_only, combined_soft = _classify_review_reasons(combined)
    blocking_reasons = _dedupe_preserving_order([*blocking_reasons, *combined_blocking])
    if auto_accept_blockers:
        blocking_reasons = _dedupe_preserving_order([*blocking_reasons, *auto_accept_blockers])
        needs_review = True
        extraction.needs_review = True

    evidence_gaps = [
        reason
        for reason in structured
        if reason.startswith("missing_critical_field:")
        or reason in {"confidence_below_threshold", "table_found_but_no_rows"}
        or reason.startswith("low_confidence:")
        or reason == "critical_identifier_unverified"
        or reason == "traceability_unverified"
        or reason == "mechanical_table_alignment_uncertain"
        or reason == "traceability_identifier_ocr_uncertain"
        or reason == "traceability_identifier_conflict"
    ]
    reviewer_focus: list[str] = []
    if "unresolved_spec" in structured or "unsupported_spec_family" in structured:
        reviewer_focus.append("Confirm material family/spec source before compliance verdict.")
    if any(reason.startswith("missing_critical_field:") for reason in structured):
        reviewer_focus.append("Verify missing mechanical columns from original certificate.")
    if any(reason.startswith("header_row_conflict:") for reason in structured):
        reviewer_focus.append("Resolve header/row semantic conflicts and provenance.")
    if "mechanical_table_alignment_uncertain" in structured:
        reviewer_focus.append(
            "Verify mechanical property values against the Results/Actual column in the mechanical table."
        )
    if "traceability_identifier_ocr_uncertain" in structured or "traceability_identifier_conflict" in structured:
        reviewer_focus.append(
            "Verify traceability identifiers against the original certificate; OCR may have confused similar characters."
        )
    if "critical_identifier_unverified" in structured or "traceability_unverified" in structured:
        reviewer_focus.append("Verify suppressed traceability identifiers against the original certificate.")
    if "validation_conflict:row_non_compliant" in structured:
        reviewer_focus.append("Re-check threshold violation evidence against resolved spec.")
    if "suspicious_duplication:heat_and_item_id" in structured:
        reviewer_focus.append(
            "heat_number and item_id are identical — confirm whether the source PDF "
            "has one traceability column or two distinct ones."
        )
    if "validation_conflict:confusable_heat_numbers" in structured:
        reviewer_focus.append(
            "Two or more heat numbers differ only by OCR-confusable characters "
            "(e.g. 5/S, 0/O, 8/B) — verify each value against the original certificate."
        )

    # Mutate in place so downstream callers see the enriched reasons.
    final_review_required = needs_review or bool(auto_accept_blockers)
    extraction.review_reasons = combined
    extraction.needs_review = final_review_required
    if final_review_required:
        extraction.status = "NEEDS_REVIEW"
    else:
        extraction.status = "COMPLETED"

    extracted_json = extraction.model_dump(mode="python")
    confidence_payload = dict(extraction.confidence_breakdown or {})
    confidence_payload.setdefault("_overall", extraction.confidence_score)
    deterministic = evaluate_review_policy(
        extracted_json=extracted_json,
        confidence=confidence_payload,
        validation_errors=[],
        document_profile=profile.to_dict() if profile is not None else None,
        confidence_threshold=review_confidence_threshold,
    )
    if needs_review or auto_accept_blockers:
        decision = "review_required"
    else:
        decision = deterministic["decision"]

    auto_accept_evidence: dict[str, Any] | None = None
    if decision == "auto_accept":
        auto_accept_evidence = _build_auto_accept_evidence(
            extraction=extraction,
            profile=profile,
            confidence_summary=deterministic["confidence_summary"],
            blocking_reasons=blocking_reasons,
            confidence_only_reasons=combined_confidence_only,
            all_reasons=combined,
            confidence_exempt=confidence_exempt,
            confidence_threshold=review_confidence_threshold,
            final_confidence=final_confidence,
        )
        if auto_accept_evidence is None:
            decision = "review_required"
            final_review_required = True
            extraction.needs_review = True
            extraction.status = "NEEDS_REVIEW"

    return ReviewDecision(
        review_required=final_review_required,
        structured_reasons=structured,
        blocking_reasons=_dedupe_preserving_order(blocking_reasons),
        confidence_only_reasons=combined_confidence_only,
        soft_reasons=combined_soft,
        evidence_gaps=_dedupe_preserving_order(evidence_gaps),
        reviewer_focus=_dedupe_preserving_order(reviewer_focus),
        all_reasons=combined,
        decision=decision,
        confidence_summary=deterministic["confidence_summary"],
        auto_accept_evidence=auto_accept_evidence,
    )
