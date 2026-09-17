"""Row/document validator.

Phase 1 refactor: compliance is now tri-state. An unresolved grade or an
unresolved spec *never* becomes ``NON_COMPLIANT``; instead the validator
emits an ``UNKNOWN_GRADE`` / ``UNRESOLVED_SPEC`` outcome and a structured
review reason. See :mod:`app.domain.grade_registry`,
:mod:`app.domain.spec_registry`, and ``docs/phase1_validation_notes.md``.
"""

from __future__ import annotations

import re

from app.domain.grade_registry import GradeResolution, has_explicit_grade_value, resolve_grade
from app.domain.outcome_taxonomy import (
    COMPLIANT,
    EXPLICIT_UNMAPPED_GRADE,
    MISSING_CRITICAL_FIELD_GRADE,
    NEEDS_REVIEW,
    NON_COMPLIANT,
    NOT_VALIDATED,
    UNSUPPORTED_SPEC_FAMILY,
    UNRESOLVED_SPEC,
    aggregate_document_outcome,
)
from app.domain.spec_registry import (
    MaterialSpec,
    SpecResolution,
    get_spec,
    known_spec_grades,
    resolve_spec,
    resolve_spec_from_standard_classification,
)
from app.schemas.extraction import UniversalDocumentExtraction, ValidationResult
from app.domain.validation_config import (
    SuspiciousBand,
    get_validation_config,
)


# Validation outcome tags (kept as exported aliases for existing imports).
OUTCOME_RESOLVED_COMPLIANT = COMPLIANT
OUTCOME_RESOLVED_NON_COMPLIANT = NON_COMPLIANT
OUTCOME_UNRESOLVED_SPEC = UNRESOLVED_SPEC
OUTCOME_UNKNOWN_GRADE = NEEDS_REVIEW
OUTCOME_AMBIGUOUS_GRADE = NEEDS_REVIEW
OUTCOME_NOT_APPLICABLE = NOT_VALIDATED
OUTCOME_EXTRACTION_UNCERTAIN = NEEDS_REVIEW


# Backward-compatibility shim. The ``/health`` endpoint and some older tests
# import ``MATERIAL_SPECS`` directly from this module. It is now materialised
# from :mod:`app.domain.spec_registry` so the two sources cannot drift.
MATERIAL_SPECS: dict[str, dict[str, float]] = {
    canonical: {
        "min_yield": _spec.min_yield_mpa,
        "min_tensile": _spec.min_tensile_mpa,
        "max_tensile": _spec.max_tensile_mpa,
        "min_elongation": _spec.min_elongation_pct,
    }
    for canonical, _spec in (
        (canonical, get_spec(canonical)) for canonical in known_spec_grades()
    )
    if _spec is not None
}


def _suspicious_numeric(value: float, band: SuspiciousBand) -> bool:
    return band.is_suspicious(value)


def _weight_or_length_is_suspicious(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    if not any(char.isdigit() for char in lowered):
        return True
    if "," in lowered and "." in lowered:
        return True
    valid_units = ("kg", "mt", "ton", "tons", "m", "mm", "cm", "ft", "pcs", "pc")
    if not any(unit in lowered for unit in valid_units):
        return not bool(re.search(r"\d+(?:[.,]\d+)?", lowered))
    return False


def _canonical_heat_pattern(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


# Character groups that are visually interchangeable on degraded or low-DPI scans.
# Each frozenset is one equivalence class; membership is checked bidirectionally.
_CONFUSABLE_DIGIT_PAIRS: list[frozenset[str]] = [
    frozenset({"0", "O", "D"}),   # round shapes
    frozenset({"1", "I", "L"}),   # thin verticals
    frozenset({"5", "S"}),        # top-loop digits
    frozenset({"6", "G", "B"}),   # partially closed loops
    frozenset({"8", "B"}),        # double-loop
    frozenset({"2", "Z"}),        # diagonal-stroke confusion
]


def _chars_confusable(a: str, b: str) -> bool:
    """Return True if *a* and *b* belong to the same OCR-confusable equivalence class."""
    au, bu = a.upper(), b.upper()
    return any(au in group and bu in group for group in _CONFUSABLE_DIGIT_PAIRS)


def _heat_numbers_confusably_similar(h1: str, h2: str) -> bool:
    """Return True when two heat numbers differ only by confusable character substitutions.

    Same-length strings where every position either matches exactly or maps to a
    known OCR confusion pair indicate that one value is a misread of the other.
    Different-length strings are treated as structurally distinct (not an OCR error).
    """
    c1 = _canonical_heat_pattern(h1)
    c2 = _canonical_heat_pattern(h2)
    if len(c1) != len(c2) or c1 == c2:
        return False
    return all(a == b or _chars_confusable(a, b) for a, b in zip(c1, c2))


def _infer_dominant_heat_pattern(data: UniversalDocumentExtraction) -> tuple[int, int] | None:
    canonical_values = [_canonical_heat_pattern(item.heat_number) for item in data.items if item.heat_number]
    if len(canonical_values) < 2:
        return None
    lengths = [len(value) for value in canonical_values]
    digit_counts = [sum(char.isdigit() for char in value) for value in canonical_values]
    dominant_length = max(set(lengths), key=lengths.count)
    dominant_digits = max(set(digit_counts), key=digit_counts.count)
    return dominant_length, dominant_digits


def _grade_spec_inconsistent(item_grade: str | None, yield_value: float | None, tensile_value: float | None) -> bool:
    if not item_grade:
        return False
    grade_key = item_grade.upper()
    if "S235" in grade_key and yield_value is not None and yield_value > 420:
        return True
    if "S355" in grade_key and yield_value is not None and yield_value < 250:
        return True
    if "X65" in grade_key and tensile_value is not None and tensile_value < 430:
        return True
    return False


def _resolve_item_spec(item) -> tuple[GradeResolution, SpecResolution]:
    grade_resolution = resolve_grade(item.grade)
    spec_resolution = resolve_spec(grade_resolution)
    if spec_resolution.status == "empty":
        pair_resolution = resolve_spec_from_standard_classification(item.standards, item.classifications)
        if pair_resolution.status == "resolved":
            return grade_resolution, pair_resolution
    return grade_resolution, spec_resolution


def _validate_against_spec(item, spec: MaterialSpec, config) -> tuple[list[str], bool, float, bool]:
    """Run the per-row spec comparison.

    Returns ``(deviations, hard_violation, row_penalty, row_suspicious)``.
    ``hard_violation`` is True only for true spec breaches (below minimum /
    outside tensile band / below min elongation). Suspicious-band trips
    contribute to ``row_penalty`` but never set ``hard_violation``.
    """

    deviations: list[str] = []
    row_penalty = 0.0
    row_suspicious = False
    hard_violation = False
    mp = item.mechanical_properties

    if mp.yield_strength_mpa is not None:
        if mp.yield_strength_mpa < spec.min_yield_mpa:
            deviations.append(
                f"Yield {mp.yield_strength_mpa:.1f} MPa below minimum {spec.min_yield_mpa:.1f} MPa"
            )
            hard_violation = True
        if _suspicious_numeric(mp.yield_strength_mpa, config.yield_strength_range):
            deviations.append(f"Yield {mp.yield_strength_mpa:.1f} MPa looks suspicious.")
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.18
    else:
        deviations.append("Yield missing - cannot verify.")
        row_penalty += 0.04

    if mp.tensile_strength_mpa is not None:
        if mp.tensile_strength_mpa < spec.min_tensile_mpa:
            deviations.append(
                f"Tensile {mp.tensile_strength_mpa:.1f} MPa below minimum {spec.min_tensile_mpa:.1f} MPa"
            )
            hard_violation = True
        if mp.tensile_strength_mpa > spec.max_tensile_mpa:
            deviations.append(
                f"Tensile {mp.tensile_strength_mpa:.1f} MPa above maximum {spec.max_tensile_mpa:.1f} MPa"
            )
            hard_violation = True
        if _suspicious_numeric(mp.tensile_strength_mpa, config.tensile_strength_range):
            deviations.append(f"Tensile {mp.tensile_strength_mpa:.1f} MPa looks suspicious.")
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.18
    else:
        deviations.append("Tensile missing - cannot verify.")
        row_penalty += 0.04

    if mp.elongation_percentage is not None:
        if spec.min_elongation_pct > 0 and mp.elongation_percentage < spec.min_elongation_pct:
            deviations.append(
                f"Elongation {mp.elongation_percentage:.1f}% below minimum {spec.min_elongation_pct:.1f}%"
            )
            hard_violation = True
        if _suspicious_numeric(mp.elongation_percentage, config.elongation_range):
            deviations.append(f"Elongation {mp.elongation_percentage:.1f}% looks suspicious.")
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.18
    elif spec.min_elongation_pct > 0:
        deviations.append("Elongation missing - cannot verify.")
        row_penalty += 0.04

    if spec.canonical == "SRCDRW02":
        if mp.yield_strength_mpa is not None and mp.yield_strength_mpa > 700.0:
            deviations.append(
                f"Yield {mp.yield_strength_mpa:.1f} MPa above sanity maximum 700.0 MPa"
            )
            hard_violation = True
        if (
            mp.yield_strength_mpa is not None
            and mp.tensile_strength_mpa is not None
            and mp.tensile_strength_mpa < mp.yield_strength_mpa
        ):
            deviations.append(
                "Tensile strength is below yield strength - sanity check failed."
            )
            hard_violation = True

    return deviations, hard_violation, row_penalty, row_suspicious


def _record_grade_resolution(item, grade_resolution: GradeResolution, spec_resolution: SpecResolution) -> None:
    """Attach grade/spec resolution metadata to the row for UI + review pack."""

    payload = grade_resolution.to_dict()
    payload["spec"] = spec_resolution.to_dict()
    item.grade_resolution = payload
    if not item.grade_provenance:
        item.grade_provenance = "row" if grade_resolution.status != "empty" else "none"


def validate_document(data: UniversalDocumentExtraction) -> UniversalDocumentExtraction:
    review_reasons: list[str] = list(data.review_reasons)
    heat_pattern = _infer_dominant_heat_pattern(data)
    config = get_validation_config(data.product_category)

    for item in data.items:
        row_penalty = 0.0
        row_suspicious = False

        # Case 1: no mechanical properties -> NOT_APPLICABLE.
        if item.mechanical_properties is None:
            grade_resolution, spec_resolution = _resolve_item_spec(item)
            _record_grade_resolution(item, grade_resolution, spec_resolution)

            deviations = ["No mechanical properties - validation skipped."]
            if _weight_or_length_is_suspicious(item.weight_or_length):
                deviations.append(
                    f"Weight/length '{item.weight_or_length}' looks malformed or lacks a clear unit."
                )
                row_penalty += 0.12
                item.needs_review = True
                review_reasons.append("malformed numeric strings")

            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=NOT_VALIDATED,
                rule_evidence=[
                    {
                        "rule": "validation_prerequisite.mechanical_properties",
                        "decision": "skipped",
                        "reason": "No mechanical properties - validation skipped.",
                    }
                ],
            )
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - row_penalty)
            continue

        grade_resolution, spec_resolution = _resolve_item_spec(item)
        _record_grade_resolution(item, grade_resolution, spec_resolution)

        if data.product_category == "WIRE_ROPE":
            deviations = []
            row_penalty = 0.0
            row_suspicious = False
            
            mp = item.mechanical_properties
            if mp.tensile_strength_mpa is None:
                deviations.append("Tensile Strength Class missing - cannot verify.")
                row_penalty += 0.04
            elif _suspicious_numeric(mp.tensile_strength_mpa, config.tensile_strength_range):
                deviations.append(f"Tensile {mp.tensile_strength_mpa:.1f} MPa looks suspicious.")
                item.needs_review = True
                row_suspicious = True
                row_penalty += 0.18
                
            evidence = [
                {
                    "rule": "product_category.wire_rope",
                    "decision": "bypass_spec",
                    "evidence": {
                        "tensile_strength_mpa": mp.tensile_strength_mpa,
                    },
                }
            ]
            item.validation = ValidationResult(
                is_compliant=True,
                deviations=deviations,
                outcome=COMPLIANT,
                rule_evidence=evidence,
            )
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - row_penalty)
            if row_suspicious:
                review_reasons.append("numeric fields are suspicious")
            continue

        # Case 2: missing or unmapped grade -> review-safe outcome.
        if spec_resolution.status == "empty":
            deviations = [
                "Grade is missing from the extracted row - manual review required."
            ]
            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=MISSING_CRITICAL_FIELD_GRADE,
                rule_evidence=[
                    {
                        "rule": "spec_resolution.grade_registry",
                        "decision": "missing_critical_field_grade",
                        "reason": spec_resolution.reason,
                        "candidate_spec_family": list(spec_resolution.candidates),
                    }
                ],
            )
            item.needs_review = True
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.2)
            review_reasons.append("missing_critical_field:grade")
            continue

        if spec_resolution.status == "unknown_grade":
            grade_field_label = getattr(item, "grade_field_label", None)
            explicit_grade = has_explicit_grade_value(
                item.grade,
                grade_provenance=item.grade_provenance,
                grade_field_label=grade_field_label,
            )
            if explicit_grade:
                deviations = [
                    (
                        f"Grade '{item.grade}' is explicitly stated but not mapped "
                        "in the internal catalog - manual review required."
                    )
                ]
                item.validation = ValidationResult(
                    is_compliant=None,
                    deviations=deviations,
                    outcome=EXPLICIT_UNMAPPED_GRADE,
                    rule_evidence=[
                        {
                            "rule": "spec_resolution.grade_registry",
                            "decision": "explicit_unmapped_grade",
                            "reason": spec_resolution.reason,
                            "candidate_spec_family": list(spec_resolution.candidates),
                        }
                    ],
                )
                item.needs_review = True
                item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.1)
                review_reasons.append("explicit_unmapped_grade")
                continue

            deviations = [
                f"Unknown grade '{item.grade or '(missing)'}' - manual review required."
            ]
            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=NEEDS_REVIEW,
                rule_evidence=[
                    {
                        "rule": "spec_resolution.grade_registry",
                        "decision": "unresolved",
                        "reason": spec_resolution.reason,
                        "candidate_spec_family": list(spec_resolution.candidates),
                    }
                ],
            )
            item.needs_review = True
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.2)
            review_reasons.append(f"unresolved_grade:{grade_resolution.raw or ''}")
            continue

        # Case 3: ambiguous grade -> review-safe outcome.
        if spec_resolution.status == "ambiguous_grade":
            candidates_display = ", ".join(spec_resolution.candidates) or "multiple candidates"
            deviations = [
                f"Ambiguous grade '{item.grade}' could match {candidates_display} - manual review required."
            ]
            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=NEEDS_REVIEW,
                rule_evidence=[
                    {
                        "rule": "spec_resolution.grade_registry",
                        "decision": "ambiguous",
                        "reason": spec_resolution.reason,
                        "candidate_spec_family": list(spec_resolution.candidates),
                    }
                ],
            )
            item.needs_review = True
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.15)
            review_reasons.append(f"ambiguous_grade:{grade_resolution.raw or ''}")
            continue

        # Case 4: resolved grade but no declared spec -> UNRESOLVED_SPEC.
        if spec_resolution.status == "unresolved":
            deviations = [
                f"Grade '{spec_resolution.canonical or item.grade}' recognised but no spec is declared - manual review required."
            ]
            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=UNRESOLVED_SPEC,
                rule_evidence=[
                    {
                        "rule": "spec_resolution.family",
                        "decision": "unresolved_spec",
                        "reason": spec_resolution.reason,
                        "candidate_spec_family": list(spec_resolution.candidates),
                    }
                ],
            )
            item.needs_review = True
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.1)
            review_reasons.append(f"unresolved_spec:{spec_resolution.canonical or ''}")
            continue

        if spec_resolution.status == "unsupported_spec_family":
            deviations = [
                f"Grade family '{spec_resolution.canonical or item.grade}' is recognised but not covered by deterministic rules."
            ]
            item.validation = ValidationResult(
                is_compliant=None,
                deviations=deviations,
                outcome=UNSUPPORTED_SPEC_FAMILY,
                rule_evidence=[
                    {
                        "rule": "spec_resolution.family",
                        "decision": "unsupported_spec_family",
                        "reason": spec_resolution.reason,
                        "candidate_spec_family": list(spec_resolution.candidates),
                    }
                ],
            )
            item.needs_review = True
            item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - 0.1)
            review_reasons.append("unsupported_spec_family")
            continue

        # Case 5: resolved spec -> run the compliance checks.
        assert spec_resolution.spec is not None  # narrow for type-checkers
        deviations, hard_violation, spec_penalty, spec_suspicious = _validate_against_spec(
            item, spec_resolution.spec, config
        )
        row_penalty += spec_penalty
        row_suspicious = row_suspicious or spec_suspicious

        if _weight_or_length_is_suspicious(item.weight_or_length):
            deviations.append(
                f"Weight/length '{item.weight_or_length}' looks malformed or lacks a clear unit."
            )
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.12
            review_reasons.append("malformed numeric strings")

        if heat_pattern and item.heat_number:
            dominant_length, dominant_digits = heat_pattern
            normalized_heat = _canonical_heat_pattern(item.heat_number)
            if (
                abs(len(normalized_heat) - dominant_length) >= 3
                or abs(sum(ch.isdigit() for ch in normalized_heat) - dominant_digits) >= 3
            ):
                deviations.append(
                    f"Heat number '{item.heat_number}' is inconsistent with the dominant pattern."
                )
                item.needs_review = True
                row_suspicious = True
                row_penalty += 0.14
                review_reasons.append("heat number format inconsistency")

        if item.weight_or_length and "," in item.weight_or_length and "." in item.weight_or_length:
            deviations.append(
                f"Weight/length '{item.weight_or_length}' mixes separators and may be misread."
            )
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.12
            review_reasons.append("mixed separators confusion")

        if _grade_spec_inconsistent(
            item.grade,
            item.mechanical_properties.yield_strength_mpa,
            item.mechanical_properties.tensile_strength_mpa,
        ):
            deviations.append(
                f"Grade '{item.grade}' is inconsistent with the extracted mechanical values."
            )
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.16
            review_reasons.append("inconsistent grade/spec combinations")

        # Suspicious duplication: the model's "fill every column" instinct copies
        # a single source value into both heat_number and item_id when the PDF
        # only has one traceability column.  Identical canonical forms are the
        # clearest signal.
        if (
            item.heat_number
            and item.item_id
            and _canonical_heat_pattern(item.heat_number) == _canonical_heat_pattern(item.item_id)
        ):
            deviations.append(
                f"heat_number '{item.heat_number}' and item_id '{item.item_id}' are identical — "
                "possible field-duplication hallucination (single-column source)."
            )
            item.needs_review = True
            row_suspicious = True
            row_penalty += 0.20
            review_reasons.append("suspicious_duplication:heat_and_item_id")

        if row_suspicious:
            review_reasons.append("numeric fields are suspicious")

        evidence = [
            {
                "rule": "spec_resolution.family",
                "decision": "resolved",
                "matched_spec_family": spec_resolution.canonical,
                "candidate_spec_family": list(spec_resolution.candidates),
                "evidence": {
                    "yield_strength_mpa": item.mechanical_properties.yield_strength_mpa,
                    "tensile_strength_mpa": item.mechanical_properties.tensile_strength_mpa,
                    "elongation_percentage": item.mechanical_properties.elongation_percentage,
                },
            }
        ]
        if hard_violation:
            outcome = NON_COMPLIANT
            is_compliant_row: bool | None = False
            evidence.append(
                {
                    "rule": "deterministic_validator.threshold_check",
                    "decision": "violation",
                    "deviations": list(deviations),
                }
            )
        else:
            outcome = COMPLIANT
            is_compliant_row = True
            evidence.append(
                {
                    "rule": "deterministic_validator.threshold_check",
                    "decision": "pass",
                }
            )

        item.validation = ValidationResult(
            is_compliant=is_compliant_row,
            deviations=deviations,
            outcome=outcome,
            rule_evidence=evidence,
        )
        item.row_confidence = max(0.05, min(item.row_confidence or 1.0, 1.0) - row_penalty)

    row_outcomes = [item.validation.outcome for item in data.items if item.validation is not None]
    data.outcome = aggregate_document_outcome([entry for entry in row_outcomes if entry])
    data.is_compliant = True if data.outcome == COMPLIANT else False if data.outcome == NON_COMPLIANT else None

    if data.total_items_detected != len(data.items):
        data.needs_review = True
        review_reasons.append("row_count_inconsistent")
        data.total_items_detected = len(data.items)

    # Cross-row confusable heat number check: if any two heat numbers in the
    # document differ only by OCR-confusable character substitutions (e.g.
    # '1011005' vs '1011006' where 5/6 are visually ambiguous on a noisy scan),
    # flag the entire document for human review rather than silently accepting
    # one of the values.
    _all_heat_values = [item.heat_number for item in data.items if item.heat_number]
    if len(_all_heat_values) >= 2:
        _confusable_found = False
        for _i in range(len(_all_heat_values)):
            for _j in range(_i + 1, len(_all_heat_values)):
                if _heat_numbers_confusably_similar(_all_heat_values[_i], _all_heat_values[_j]):
                    _confusable_found = True
                    break
            if _confusable_found:
                break
        if _confusable_found:
            for item in data.items:
                if item.heat_number:
                    item.needs_review = True
            data.needs_review = True
            review_reasons.append("validation_conflict:confusable_heat_numbers")

    # Contradiction check: if audit text admits a field is unreadable/unclear but
    # heat_number is still non-null, the model contradicted itself — treat as a
    # hallucination and block auto-acceptance.
    _CONTRADICTION_KEYWORDS = frozenset({
        "unclear", "partially visible", "hard to read", "guess",
        "blurred", "cannot read", "can't read", "not readable", "illegible",
    })
    _audit_text = " ".join(filter(None, [
        str(data.ai_analysis_remarks or "").lower(),
        str(data.stage_b_extraction_audit or "").lower(),
    ]))
    if any(kw in _audit_text for kw in _CONTRADICTION_KEYWORDS):
        _contradicted_items = [item for item in data.items if item.heat_number]
        if _contradicted_items:
            for _item in _contradicted_items:
                _item.heat_number = None
                _item.needs_review = True
            data.needs_review = True
            review_reasons.append("contradictory_audit:heat_number_guessed")

    data.review_reasons = sorted(set(review_reasons))
    return data
