from __future__ import annotations

from app.domain.review_invariants import TOO_MANY_ROWS_REASON, apply_review_invariants
from app.schemas.extraction import ExtractedItem, MechanicalProperties, UniversalDocumentExtraction, ValidationResult


def _item(*, needs_review: bool = False) -> ExtractedItem:
    return ExtractedItem(
        heat_number="H-1",
        grade="S235JR",
        needs_review=needs_review,
        mechanical_properties=MechanicalProperties(
            yield_strength_mpa=250.0,
            tensile_strength_mpa=400.0,
            elongation_percentage=28.0,
        ),
        validation=ValidationResult(is_compliant=not needs_review, deviations=[], outcome="COMPLIANT"),
    )


def _extraction(items: list[ExtractedItem], **kwargs) -> UniversalDocumentExtraction:
    return UniversalDocumentExtraction(
        supplier_name="Mill",
        document_type="Mill Test Certificate",
        total_items_detected=kwargs.get("total_items_detected", len(items)),
        items=items,
        confidence_score=0.9,
        needs_review=kwargs.get("needs_review", False),
        review_reasons=list(kwargs.get("review_reasons", [])),
    )


def test_zero_items_never_emits_too_many_rows():
    extraction = _extraction(
        [],
        total_items_detected=4,
        review_reasons=[TOO_MANY_ROWS_REASON, "row_count_inconsistent"],
    )
    apply_review_invariants(extraction, raw_reported_total_items=4)
    assert extraction.total_items_detected == 0
    assert TOO_MANY_ROWS_REASON not in extraction.review_reasons
    assert "no_items_extracted" in extraction.review_reasons
    assert "row_count_inconsistent" in extraction.review_reasons


def test_single_item_never_emits_too_many_rows():
    extraction = _extraction(
        [_item(needs_review=True)],
        total_items_detected=1,
        review_reasons=[TOO_MANY_ROWS_REASON, "missing_critical_field:grade"],
        needs_review=True,
    )
    apply_review_invariants(extraction, raw_reported_total_items=1)
    assert extraction.total_items_detected == 1
    assert TOO_MANY_ROWS_REASON not in extraction.review_reasons
    assert "missing_critical_field:grade" in extraction.review_reasons


def test_many_suspicious_items_can_emit_too_many_rows():
    items = [_item(needs_review=True) for _ in range(4)]
    extraction = _extraction(items, review_reasons=["numeric fields are suspicious"])
    apply_review_invariants(extraction, raw_reported_total_items=4)
    assert extraction.total_items_detected == 4
    assert TOO_MANY_ROWS_REASON in extraction.review_reasons


def test_many_clean_items_do_not_emit_too_many_rows():
    items = [_item(needs_review=False) for _ in range(4)]
    extraction = _extraction(items, review_reasons=[TOO_MANY_ROWS_REASON])
    apply_review_invariants(extraction, raw_reported_total_items=4)
    assert TOO_MANY_ROWS_REASON not in extraction.review_reasons


def test_conflict_tokens_are_preserved():
    extraction = _extraction([_item()], review_reasons=[])
    apply_review_invariants(
        extraction,
        canonical_review_tokens=["conflicting_labeled_candidates:order_number"],
        raw_reported_total_items=1,
    )
    assert "conflicting_labeled_candidates:order_number" in extraction.review_reasons
    assert extraction.needs_review is True
