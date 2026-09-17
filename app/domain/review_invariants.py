"""Deterministic review-reason invariants derived from final canonical state."""

from __future__ import annotations

from typing import Any, Iterable

from app.schemas.extraction import UniversalDocumentExtraction

TOO_MANY_ROWS_REASON = "too many rows are suspicious"
ROW_COUNT_INCONSISTENT = "row_count_inconsistent"
NO_ITEMS_EXTRACTED = "no_items_extracted"
MIN_ITEMS_FOR_TOO_MANY_ROWS = 2
SUSPICIOUS_ROW_RATIO = 0.35
CONFLICT_PREFIX = "conflicting_labeled_candidates:"


def canonical_item_count(extraction: UniversalDocumentExtraction) -> int:
    return len(extraction.items)


def _suspicious_row_count(extraction: UniversalDocumentExtraction) -> int:
    return sum(1 for item in extraction.items if item.needs_review)


def _dedupe(reasons: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def apply_review_invariants(
    extraction: UniversalDocumentExtraction,
    *,
    canonical_review_tokens: list[str] | None = None,
    raw_reported_total_items: int | None = None,
) -> UniversalDocumentExtraction:
    """Rewrite count/conflict reasons from the final canonical extraction."""

    item_count = canonical_item_count(extraction)
    reasons = _dedupe([*extraction.review_reasons, *(canonical_review_tokens or [])])

    if raw_reported_total_items is not None and raw_reported_total_items != item_count:
        if ROW_COUNT_INCONSISTENT not in reasons:
            reasons.append(ROW_COUNT_INCONSISTENT)
    elif extraction.total_items_detected != item_count:
        if ROW_COUNT_INCONSISTENT not in reasons:
            reasons.append(ROW_COUNT_INCONSISTENT)

    extraction.total_items_detected = item_count

    if item_count == 0 and NO_ITEMS_EXTRACTED not in reasons:
        reasons.append(NO_ITEMS_EXTRACTED)

    reasons = [reason for reason in reasons if reason != TOO_MANY_ROWS_REASON]
    suspicious_rows = _suspicious_row_count(extraction)
    if item_count >= MIN_ITEMS_FOR_TOO_MANY_ROWS and suspicious_rows / item_count >= SUSPICIOUS_ROW_RATIO:
        reasons.append(TOO_MANY_ROWS_REASON)

    if item_count < MIN_ITEMS_FOR_TOO_MANY_ROWS:
        reasons = [reason for reason in reasons if reason != TOO_MANY_ROWS_REASON]

    extraction.review_reasons = _dedupe(reasons)
    if extraction.review_reasons:
        extraction.needs_review = True
    return extraction


def review_tokens_from_meta(preprocessing_meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(preprocessing_meta, dict):
        return []
    reconciliation = preprocessing_meta.get("canonical_reconciliation")
    if not isinstance(reconciliation, dict):
        return []
    tokens = reconciliation.get("review_tokens")
    if not isinstance(tokens, list):
        return []
    return [str(token) for token in tokens if str(token).strip()]


__all__ = [
    "CONFLICT_PREFIX",
    "MIN_ITEMS_FOR_TOO_MANY_ROWS",
    "TOO_MANY_ROWS_REASON",
    "apply_review_invariants",
    "canonical_item_count",
    "review_tokens_from_meta",
]
