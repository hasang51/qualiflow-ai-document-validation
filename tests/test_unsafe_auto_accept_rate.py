from __future__ import annotations

from scripts.evaluate_outputs import _aggregate_academic, _rate


def test_unsafe_auto_accept_rate_is_zero_hard_gate():
    rows = [
        {
            "field_hits": 1,
            "field_total": 1,
            "critical_hits": 1,
            "critical_total": 1,
            "document_type_hit": 1,
            "document_type_total": 1,
            "processing_decision_hit": 1,
            "processing_decision_total": 1,
            "review_known": True,
            "review_required": True,
            "unsafe_auto_accept": 0,
            "unsafe_auto_accept_total": 1,
            "missing_required": 0,
            "required_total": 1,
            "latency_ms": 10.0,
        },
        {
            "field_hits": 1,
            "field_total": 1,
            "critical_hits": 1,
            "critical_total": 1,
            "document_type_hit": 1,
            "document_type_total": 1,
            "processing_decision_hit": 1,
            "processing_decision_total": 1,
            "review_known": True,
            "review_required": False,
            "unsafe_auto_accept": 0,
            "unsafe_auto_accept_total": 0,
            "missing_required": 0,
            "required_total": 1,
            "latency_ms": 12.0,
        },
    ]
    summary = _aggregate_academic(rows)
    assert summary["unsafe_auto_accept_rate"] == 0.0


def test_unsafe_auto_accept_rate_detects_gold_review_marked_auto_accept():
    rows = [
        {
            "field_hits": 0,
            "field_total": 1,
            "critical_hits": 0,
            "critical_total": 1,
            "document_type_hit": 0,
            "document_type_total": 1,
            "processing_decision_hit": 0,
            "processing_decision_total": 1,
            "review_known": True,
            "review_required": False,
            "unsafe_auto_accept": 1,
            "unsafe_auto_accept_total": 1,
            "missing_required": 0,
            "required_total": 1,
            "latency_ms": None,
        }
    ]
    summary = _aggregate_academic(rows)
    assert summary["unsafe_auto_accept_rate"] == _rate(1, 1)
    assert summary["unsafe_auto_accept_rate"] == 1.0
