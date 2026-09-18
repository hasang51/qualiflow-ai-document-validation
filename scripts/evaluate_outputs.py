"""Metric primitives and CLI evaluation for QualiFlow outputs.

The importable helpers are used by :mod:`scripts.run_eval` and the unit tests.
The CLI evaluates JSON predictions against a manually annotated gold subset
indexed by ``data/gold/metadata_20.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUMERIC_TOLERANCE = 1.0

CRITICAL_FIELDS = (
    "supplier_name",
    "document_type",
    "heat_numbers",
    "grades",
    "yield_strength_mpa",
    "tensile_strength_mpa",
)

SIMPLE_FIELDS = ("supplier_name", "document_type", "certificate_date")
LIST_STRING_FIELDS = ("heat_numbers", "grades")
LIST_NUMERIC_FIELDS = ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")
ALL_FIELDS = SIMPLE_FIELDS + LIST_STRING_FIELDS + LIST_NUMERIC_FIELDS

ACADEMIC_TARGET_FIELDS = (
    "supplier_name",
    "document_type",
    "certificate_date",
    "item_id",
    "heat_number",
    "grade",
    "weight_or_length",
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "elongation_percentage",
    "compliance_outcome",
    "review_required",
    "review_reasons",
)

ACADEMIC_CRITICAL_FIELDS = (
    "heat_number",
    "grade",
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "elongation_percentage",
    "compliance_outcome",
    "review_required",
)

REQUIRED_FIELDS = ACADEMIC_CRITICAL_FIELDS

UNVERIFIED_GOLD_STATUSES = frozenset(
    {"candidate", "unverified", "needs_manual_verification", "needs_manual"}
)
TRACEABILITY_FIELDS = ("heat_number", "batch_number", "lot_number", "cast_number")
DECISION_FIELDS = frozenset(
    {"review_required", "review_reasons", "processing_decision", "compliance_outcome"}
)


@dataclass
class FieldComparison:
    field: str
    equal: bool
    gold: Any = None
    predicted: Any = None


@dataclass
class DocumentEvaluation:
    document_id: str
    filename: str
    mode: str | None = None
    route_used: str | None = None
    quality_class: str | None = None
    field_hits: int = 0
    field_total: int = 0
    critical_hits: int = 0
    critical_total: int = 0
    compliance_correct: bool | None = None
    review_required: bool | None = None
    latency_ms: float | None = None
    completeness: float | None = None
    comparisons: list[FieldComparison] = field(default_factory=list)


def _normalise_string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _normalise_list_string(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(v).strip() for v in value if v not in (None, "")]
    else:
        items = [chunk.strip() for chunk in str(value).split("|") if chunk.strip()]
    return sorted({_normalise_string(item) for item in items if _normalise_string(item)})


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_list_float(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).split("|")
    cleaned: list[float] = []
    for item in raw:
        parsed = _coerce_float(item)
        if parsed is not None:
            cleaned.append(round(parsed, 2))
    return sorted(cleaned)


def _lists_equal_float(a: list[float], b: list[float], tolerance: float = NUMERIC_TOLERANCE) -> bool:
    if len(a) != len(b):
        return False
    a_sorted, b_sorted = sorted(a), sorted(b)
    return all(abs(x - y) <= tolerance for x, y in zip(a_sorted, b_sorted))


def compare_field(field_name: str, gold_value: Any, predicted_value: Any) -> FieldComparison:
    if field_name in SIMPLE_FIELDS:
        eq = _normalise_string(gold_value) == _normalise_string(predicted_value)
        return FieldComparison(field=field_name, equal=eq, gold=gold_value, predicted=predicted_value)
    if field_name in LIST_STRING_FIELDS:
        a = _normalise_list_string(gold_value)
        b = _normalise_list_string(predicted_value)
        return FieldComparison(field=field_name, equal=a == b, gold=a, predicted=b)
    if field_name in LIST_NUMERIC_FIELDS:
        a = _normalise_list_float(gold_value)
        b = _normalise_list_float(predicted_value)
        return FieldComparison(field=field_name, equal=_lists_equal_float(a, b), gold=a, predicted=b)
    return FieldComparison(field=field_name, equal=gold_value == predicted_value, gold=gold_value, predicted=predicted_value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_verified_gold(payload: dict[str, Any] | None) -> bool:
    """Legacy gold files without a status are treated as verified.

    Candidate / unverified annotations must never enter scored accuracy.
    """

    if not isinstance(payload, dict) or not payload:
        return False
    status = _normalise_string(payload.get("_ground_truth_status"))
    if not status:
        return True
    return status == "verified"


def _split_scalar_or_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and "|" in value:
        return [chunk.strip() for chunk in value.split("|")]
    return [value]


def _normalise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = _normalise_string(value)
    if lowered in {"true", "yes", "y", "1", "review", "needs_review", "manual_review"}:
        return True
    if lowered in {"false", "no", "n", "0", "auto_accept", "accepted", "pass"}:
        return False
    return None


def _normalise_academic_value(value: Any) -> list[Any] | str | float | bool | None:
    """Normalize a value for exact-match academic evaluation.

    Strings use lowercase/trim/collapsed whitespace. Lists are normalized item by
    item and sorted by representation so row order does not dominate evaluation.
    Numeric values remain numeric, enabling tolerance-based comparison.
    """

    if _is_missing(value):
        return None
    parts = _split_scalar_or_list(value)
    if len(parts) > 1:
        normalized = [_normalise_academic_value(part) for part in parts]
        return sorted([part for part in normalized if part is not None], key=lambda item: repr(item))
    if isinstance(value, bool):
        return value
    numeric = _coerce_float(value)
    if numeric is not None and not isinstance(value, bool):
        return round(numeric, 4)
    return _normalise_string(value)


def _academic_equal(gold_value: Any, predicted_value: Any) -> bool:
    if _is_missing(predicted_value):
        return False

    gold_parts = _split_scalar_or_list(gold_value)
    pred_parts = _split_scalar_or_list(predicted_value)
    if len(gold_parts) != len(pred_parts):
        return False

    gold_numeric = [_coerce_float(part) for part in gold_parts]
    pred_numeric = [_coerce_float(part) for part in pred_parts]
    if all(part is not None for part in gold_numeric) and all(part is not None for part in pred_numeric):
        return _lists_equal_float(
            [float(part) for part in gold_numeric if part is not None],
            [float(part) for part in pred_numeric if part is not None],
            tolerance=NUMERIC_TOLERANCE,
        )

    return _normalise_academic_value(gold_value) == _normalise_academic_value(predicted_value)


def predicted_fields_from_extraction(extraction: dict) -> dict[str, Any]:
    items = extraction.get("items") or []
    heats = [item.get("heat_number") for item in items if item.get("heat_number")]
    grades = [item.get("grade") for item in items if item.get("grade")]
    yields: list[float] = []
    tensiles: list[float] = []
    elongs: list[float] = []
    for item in items:
        mp = item.get("mechanical_properties") or {}
        if mp.get("yield_strength_mpa") is not None:
            yields.append(float(mp["yield_strength_mpa"]))
        if mp.get("tensile_strength_mpa") is not None:
            tensiles.append(float(mp["tensile_strength_mpa"]))
        if mp.get("elongation_percentage") is not None:
            elongs.append(float(mp["elongation_percentage"]))
    return {
        "supplier_name": extraction.get("supplier_name"),
        "document_type": extraction.get("document_type"),
        "certificate_date": extraction.get("certificate_date"),
        "is_compliant": extraction.get("is_compliant"),
        "heat_numbers": heats,
        "grades": grades,
        "yield_strength_mpa": yields,
        "tensile_strength_mpa": tensiles,
        "elongation_percentage": elongs,
    }


def _unwrap_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    extraction = payload.get("extraction")
    if isinstance(extraction, dict):
        merged = dict(extraction)
        for key, value in payload.items():
            if key != "extraction" and key not in merged:
                merged[key] = value
        return merged
    return payload


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    return items if isinstance(items, list) else []


def _flatten_academic_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return canonical academic fields from flat or QualiFlow-style JSON."""

    data = _unwrap_prediction(payload)
    items = _extract_items(data)

    def first_present(*keys: str) -> Any:
        for key in keys:
            if key in data:
                return data.get(key)
        return None

    def item_values(key: str) -> list[Any]:
        direct_plural = data.get(f"{key}s")
        if direct_plural is not None:
            return _split_scalar_or_list(direct_plural)
        direct = data.get(key)
        if direct is not None:
            return _split_scalar_or_list(direct)
        return [item.get(key) for item in items if not _is_missing(item.get(key))]

    def mechanical_values(key: str) -> list[Any]:
        direct = data.get(key)
        if direct is not None:
            return _split_scalar_or_list(direct)
        values: list[Any] = []
        for item in items:
            mp = item.get("mechanical_properties") if isinstance(item, dict) else None
            if isinstance(mp, dict) and not _is_missing(mp.get(key)):
                values.append(mp.get(key))
        return values

    compliance = first_present("compliance_outcome", "outcome", "is_compliant")
    if isinstance(compliance, bool):
        compliance = "compliant" if compliance else "non_compliant"

    review_required = first_present("review_required", "needs_review")
    review_bool = _normalise_bool(review_required)
    review_required = review_bool if review_bool is not None else review_required

    fields = {
        "supplier_name": first_present("supplier_name"),
        "document_type": first_present("document_type"),
        "certificate_date": first_present("certificate_date"),
        "item_id": item_values("item_id"),
        "heat_number": item_values("heat_number"),
        "batch_number": item_values("batch_number"),
        "lot_number": item_values("lot_number"),
        "cast_number": item_values("cast_number"),
        "grade": item_values("grade"),
        "weight_or_length": item_values("weight_or_length"),
        "yield_strength_mpa": mechanical_values("yield_strength_mpa"),
        "tensile_strength_mpa": mechanical_values("tensile_strength_mpa"),
        "elongation_percentage": mechanical_values("elongation_percentage"),
        "compliance_outcome": compliance,
        "review_required": review_required,
        "review_reasons": first_present("review_reasons"),
        "processing_decision": first_present("processing_decision"),
        "latency_ms": first_present("latency_ms"),
    }
    if _is_missing(fields["processing_decision"]) and not _is_missing(review_required):
        fields["processing_decision"] = "needs_review" if review_bool else "auto_accept"
    return fields


def evaluate_document(
    *,
    gold_record: dict,
    per_document: dict,
) -> DocumentEvaluation:
    extraction = per_document.get("extraction") or {}
    predicted = predicted_fields_from_extraction(extraction)

    comparisons: list[FieldComparison] = []
    field_hits = 0
    field_total = 0
    critical_hits = 0
    critical_total = 0

    for field_name in ALL_FIELDS:
        comp = compare_field(field_name, gold_record.get(field_name), predicted.get(field_name))
        comparisons.append(comp)
        field_total += 1
        if comp.equal:
            field_hits += 1
        if field_name in CRITICAL_FIELDS:
            critical_total += 1
            if comp.equal:
                critical_hits += 1

    # Compliance comparison
    compliance_correct: bool | None = None
    if gold_record.get("is_compliant") is not None:
        gold_compliant = bool(gold_record.get("is_compliant"))
        predicted_compliant = extraction.get("is_compliant")
        if predicted_compliant is None:
            compliance_correct = False
        else:
            compliance_correct = bool(predicted_compliant) == gold_compliant

    # Completeness = fraction of critical fields where we produced *any* value
    produced = 0
    for field_name in CRITICAL_FIELDS:
        value = predicted.get(field_name)
        if isinstance(value, list):
            if value:
                produced += 1
        else:
            if value not in (None, ""):
                produced += 1
    completeness = produced / len(CRITICAL_FIELDS)

    return DocumentEvaluation(
        document_id=per_document.get("document_id") or gold_record.get("document_id"),
        filename=per_document.get("filename") or gold_record.get("filename") or "",
        mode=per_document.get("mode"),
        route_used=per_document.get("route_used"),
        quality_class=(per_document.get("profile") or {}).get("quality_class"),
        field_hits=field_hits,
        field_total=field_total,
        critical_hits=critical_hits,
        critical_total=critical_total,
        compliance_correct=compliance_correct,
        review_required=extraction.get("needs_review"),
        latency_ms=per_document.get("latency_ms"),
        completeness=round(completeness, 4),
        comparisons=comparisons,
    )


def aggregate_metrics(evaluations: Iterable[DocumentEvaluation]) -> dict[str, float | int]:
    evaluations = list(evaluations)
    n = len(evaluations)
    if n == 0:
        return {
            "n": 0,
            "field_accuracy": 0.0,
            "critical_field_accuracy": 0.0,
            "compliance_decision_accuracy": 0.0,
            "review_rate": 0.0,
            "stp_rate": 0.0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "extraction_completeness": 0.0,
        }
    total_field = sum(ev.field_total for ev in evaluations) or 1
    total_field_hits = sum(ev.field_hits for ev in evaluations)
    total_critical = sum(ev.critical_total for ev in evaluations) or 1
    total_critical_hits = sum(ev.critical_hits for ev in evaluations)
    compliance_samples = [ev.compliance_correct for ev in evaluations if ev.compliance_correct is not None]
    compliance_accuracy = (sum(1 for v in compliance_samples if v) / len(compliance_samples)) if compliance_samples else 0.0
    review_samples = [ev.review_required for ev in evaluations if ev.review_required is not None]
    review_rate = (sum(1 for v in review_samples if v) / len(review_samples)) if review_samples else 0.0
    stp_rate = 1.0 - review_rate
    latencies = [ev.latency_ms for ev in evaluations if ev.latency_ms is not None]
    average_latency = mean(latencies) if latencies else 0.0
    if latencies:
        sorted_latencies = sorted(latencies)
        idx = max(0, min(len(sorted_latencies) - 1, int(round(0.95 * (len(sorted_latencies) - 1)))))
        p95_latency = sorted_latencies[idx]
    else:
        p95_latency = 0.0
    completeness_samples = [ev.completeness for ev in evaluations if ev.completeness is not None]
    avg_completeness = mean(completeness_samples) if completeness_samples else 0.0
    return {
        "n": n,
        "field_accuracy": round(total_field_hits / total_field, 4),
        "critical_field_accuracy": round(total_critical_hits / total_critical, 4),
        "compliance_decision_accuracy": round(compliance_accuracy, 4),
        "review_rate": round(review_rate, 4),
        "stp_rate": round(stp_rate, 4),
        "average_latency_ms": round(average_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "extraction_completeness": round(avg_completeness, 4),
    }


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve_ground_truth_path(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        metadata_path.parent / path,
        metadata_path.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _value_to_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _average(values: list[float]) -> float:
    return round(mean(values), 1) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def _evaluate_academic_document(
    *,
    metadata_row: dict[str, str],
    gold_payload: dict[str, Any] | None,
    prediction_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id = metadata_row.get("doc_id", "")
    quality_bucket = metadata_row.get("quality_bucket", "")
    verified = _is_verified_gold(gold_payload)
    gold_source = dict(gold_payload or {}) if verified else {}
    if (
        verified
        and _is_missing(gold_source.get("document_type"))
        and not _is_missing(metadata_row.get("document_type"))
    ):
        gold_source["document_type"] = metadata_row.get("document_type")
    gold_fields = _flatten_academic_fields(gold_source) if verified else {}
    pred_fields = _flatten_academic_fields(prediction_payload or {})
    prediction_missing = prediction_payload is None
    gold_missing = gold_payload is None

    field_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    field_total = 0
    field_hits = 0
    critical_total = 0
    critical_hits = 0
    missing_required = 0
    required_total = 0
    extraction_misses = 0
    fail_closed_review = 0
    traceability_total = 0
    traceability_hits = 0

    for field_name in ACADEMIC_TARGET_FIELDS:
        gold_value = gold_fields.get(field_name)
        pred_value = pred_fields.get(field_name)
        has_gold = verified and not _is_missing(gold_value)
        if has_gold:
            field_total += 1
            equal = False if prediction_missing else _academic_equal(gold_value, pred_value)
            if equal:
                field_hits += 1
            elif field_name not in DECISION_FIELDS:
                extraction_misses += 1
        else:
            equal = None

        if field_name in ACADEMIC_CRITICAL_FIELDS and has_gold:
            critical_total += 1
            if equal:
                critical_hits += 1

        if field_name in REQUIRED_FIELDS and has_gold:
            required_total += 1
            if _is_missing(pred_value):
                missing_required += 1

        field_rows.append(
            {
                "doc_id": doc_id,
                "quality_bucket": quality_bucket,
                "field": field_name,
                "has_gold": has_gold,
                "correct": "" if equal is None else bool(equal),
                "gold_value": _value_to_cell(gold_value) if verified else "",
                "predicted_value": _value_to_cell(pred_value),
                "prediction_missing": prediction_missing or _is_missing(pred_value),
                "verified_gold": verified,
            }
        )

        if has_gold and not equal:
            reason = "missing_prediction_file" if prediction_missing else "missing_field" if _is_missing(pred_value) else "mismatch"
            if field_name == "review_required":
                gold_review_flag = _normalise_bool(gold_value)
                pred_review_flag = _normalise_bool(pred_value)
                if gold_review_flag is False and pred_review_flag is True:
                    reason = "fail_closed_review"
            failure_rows.append(
                {
                    "doc_id": doc_id,
                    "file_name": metadata_row.get("file_name", ""),
                    "quality_bucket": quality_bucket,
                    "field": field_name,
                    "reason": reason,
                    "gold_value": _value_to_cell(gold_value),
                    "predicted_value": _value_to_cell(pred_value),
                }
            )

    for field_name in TRACEABILITY_FIELDS:
        gold_value = gold_fields.get(field_name)
        if not verified or _is_missing(gold_value):
            continue
        traceability_total += 1
        pred_value = pred_fields.get(field_name)
        if not prediction_missing and _academic_equal(gold_value, pred_value):
            traceability_hits += 1

    document_type_total = 0
    document_type_hit = 0
    if verified and not _is_missing(gold_fields.get("document_type")):
        document_type_total = 1
        document_type_hit = int(not prediction_missing and _academic_equal(gold_fields.get("document_type"), pred_fields.get("document_type")))

    processing_total = 0
    processing_hit = 0
    if verified and not _is_missing(gold_fields.get("processing_decision")):
        processing_total = 1
        processing_hit = int(not prediction_missing and _academic_equal(gold_fields.get("processing_decision"), pred_fields.get("processing_decision")))

    pred_review = _normalise_bool(pred_fields.get("review_required"))
    gold_review = _normalise_bool(gold_fields.get("review_required")) if verified else None
    review_known = pred_review is not None and not prediction_missing
    review_required_gold = gold_review is True
    unsafe_auto_accept = int(verified and review_required_gold and pred_review is False)
    if verified and gold_review is False and pred_review is True:
        fail_closed_review = 1

    latency = _coerce_float(pred_fields.get("latency_ms")) if not prediction_missing else None
    pred_source = dict(prediction_payload or {})
    schema_failure = 0
    if not prediction_missing:
        extraction_source = pred_source.get("extraction") if isinstance(pred_source.get("extraction"), dict) else {}
        if pred_source.get("schema_failure") is True or extraction_source.get("schema_failure") is True:
            schema_failure = 1
        reasons = pred_source.get("review_reasons") or extraction_source.get("review_reasons") or []
        if isinstance(reasons, list) and "extraction_schema_invalid" in reasons:
            schema_failure = 1
        usage_source = pred_source.get("llm_usage") if isinstance(pred_source.get("llm_usage"), dict) else {}
        if usage_source.get("schema_failure") is True:
            schema_failure = 1
    usage = pred_source.get("llm_usage") if isinstance(pred_source.get("llm_usage"), dict) else {}
    input_tokens = int(usage.get("total_input_tokens") or pred_source.get("input_tokens") or 0)
    output_tokens = int(usage.get("total_output_tokens") or pred_source.get("output_tokens") or 0)
    estimated_cost = usage.get("estimated_cost_usd")
    if estimated_cost is None:
        estimated_cost = pred_source.get("estimated_cost_usd")

    doc_row = {
        "doc_id": doc_id,
        "file_name": metadata_row.get("file_name", ""),
        "quality_bucket": quality_bucket,
        "gold_missing": gold_missing,
        "verified_gold": verified,
        "prediction_missing": prediction_missing,
        "field_hits": field_hits,
        "field_total": field_total,
        "critical_hits": critical_hits,
        "critical_total": critical_total,
        "document_type_hit": document_type_hit,
        "document_type_total": document_type_total,
        "processing_decision_hit": processing_hit,
        "processing_decision_total": processing_total,
        "review_known": review_known,
        "review_required": bool(pred_review) if review_known else "",
        "unsafe_auto_accept": unsafe_auto_accept,
        "unsafe_auto_accept_total": int(verified and review_required_gold),
        "fail_closed_review": fail_closed_review,
        "extraction_misses": extraction_misses,
        "traceability_hits": traceability_hits,
        "traceability_total": traceability_total,
        "missing_required": missing_required,
        "required_total": required_total,
        "latency_ms": latency,
        "schema_failure": schema_failure,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost if estimated_cost is not None else 0,
    }
    return doc_row, field_rows, failure_rows


def _aggregate_academic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified_rows = [row for row in rows if row.get("verified_gold")]
    predicted_rows = [row for row in rows if not row.get("prediction_missing")]
    field_hits = sum(int(row["field_hits"]) for row in verified_rows)
    field_total = sum(int(row["field_total"]) for row in verified_rows)
    critical_hits = sum(int(row["critical_hits"]) for row in verified_rows)
    critical_total = sum(int(row["critical_total"]) for row in verified_rows)
    doc_type_hits = sum(int(row["document_type_hit"]) for row in verified_rows)
    doc_type_total = sum(int(row["document_type_total"]) for row in verified_rows)
    processing_hits = sum(int(row["processing_decision_hit"]) for row in verified_rows)
    processing_total = sum(int(row["processing_decision_total"]) for row in verified_rows)
    review_known = sum(1 for row in predicted_rows if row["review_known"])
    review_count = sum(1 for row in predicted_rows if row["review_known"] and row["review_required"] is True)
    auto_accept_count = sum(1 for row in predicted_rows if row["review_known"] and row["review_required"] is False)
    unsafe = sum(int(row["unsafe_auto_accept"]) for row in verified_rows)
    unsafe_total = sum(int(row["unsafe_auto_accept_total"]) for row in verified_rows)
    fail_closed = sum(int(row.get("fail_closed_review") or 0) for row in verified_rows)
    extraction_misses = sum(int(row.get("extraction_misses") or 0) for row in verified_rows)
    traceability_hits = sum(int(row.get("traceability_hits") or 0) for row in verified_rows)
    traceability_total = sum(int(row.get("traceability_total") or 0) for row in verified_rows)
    missing_required = sum(int(row["missing_required"]) for row in verified_rows)
    required_total = sum(int(row["required_total"]) for row in verified_rows)
    latencies = [float(row["latency_ms"]) for row in predicted_rows if row.get("latency_ms") is not None]
    schema_failures = sum(int(row.get("schema_failure") or 0) for row in predicted_rows)
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in predicted_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in predicted_rows)
    estimated_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in predicted_rows)
    n_predicted = len(predicted_rows)
    n_verified = len(verified_rows)

    return {
        "n_documents": len(rows),
        "n_verified_documents": n_verified,
        "n_predictions": n_predicted,
        "field_accuracy": _rate(field_hits, field_total) if field_total else "",
        "critical_field_accuracy": _rate(critical_hits, critical_total) if critical_total else "",
        "document_type_accuracy": _rate(doc_type_hits, doc_type_total) if doc_type_total else "",
        "processing_decision_accuracy": _rate(processing_hits, processing_total) if processing_total else "",
        "review_rate": _rate(review_count, review_known),
        "auto_accept_count": auto_accept_count,
        "unsafe_auto_accept_count": unsafe,
        "unsafe_auto_accept_rate": _rate(unsafe, unsafe_total),
        "fail_closed_review_count": fail_closed,
        "extraction_miss_count": extraction_misses,
        "traceability_exact_matches": traceability_hits,
        "traceability_exact_misses": max(0, traceability_total - traceability_hits),
        "traceability_total": traceability_total,
        "missing_required_field_rate": _rate(missing_required, required_total),
        "schema_failures": schema_failures,
        "schema_failure_rate": _rate(schema_failures, n_predicted),
        "average_latency_ms": _average(latencies) if latencies else "",
        "p50_latency_ms": _percentile(latencies, 0.50) if latencies else "",
        "p95_latency_ms": _percentile(latencies, 0.95) if latencies else "",
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "estimated_cost_per_document_usd": round(estimated_cost / n_predicted, 6) if n_predicted else 0.0,
    }


def _metrics_by_field(field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name in ACADEMIC_TARGET_FIELDS:
        samples = [row for row in field_rows if row["field"] == field_name and row["has_gold"]]
        correct = sum(1 for row in samples if row["correct"] is True)
        missing_predictions = sum(1 for row in samples if row["prediction_missing"])
        rows.append(
            {
                "field": field_name,
                "samples": len(samples),
                "correct": correct,
                "accuracy": _rate(correct, len(samples)),
                "missing_predictions": missing_predictions,
            }
        )
    return rows


def _metrics_by_quality_bucket(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = sorted({row.get("quality_bucket") or "unknown" for row in doc_rows})
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        bucket_rows = [row for row in doc_rows if (row.get("quality_bucket") or "unknown") == bucket]
        aggregate = _aggregate_academic(bucket_rows)
        aggregate["quality_bucket"] = bucket
        rows.append(aggregate)
    return rows


def _write_eval_report(
    *,
    path: Path,
    metadata_path: Path,
    predictions_dir: Path,
    summary: dict[str, Any],
    failure_count: int,
    by_bucket: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# QualiFlow Evaluation Report",
        "",
        f"- metadata: `{metadata_path}`",
        f"- predictions: `{predictions_dir}`",
        f"- documents in metadata: {summary.get('n_documents')}",
        f"- verified ground-truth documents: {summary.get('n_verified_documents')}",
        f"- predictions scored as pipeline observations: {summary.get('n_predictions')}",
        "",
        "## Pipeline observations (all processed PDFs)",
        "",
        "These figures describe live pipeline behaviour. They are not production accuracy.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| total PDFs in metadata | {summary.get('n_documents')} |",
        f"| successful predictions | {summary.get('successful', summary.get('n_predictions'))} |",
        f"| failed | {summary.get('failed', '')} |",
        f"| skipped | {summary.get('skipped', '')} |",
        f"| review rate | {summary.get('review_rate')} |",
        f"| auto-accept count | {summary.get('auto_accept_count')} |",
        f"| schema failures | {summary.get('schema_failures')} |",
        f"| latency p50 (ms) | {summary.get('p50_latency_ms')} |",
        f"| latency p95 (ms) | {summary.get('p95_latency_ms')} |",
        f"| total input tokens | {summary.get('total_input_tokens')} |",
        f"| total output tokens | {summary.get('total_output_tokens')} |",
        f"| estimated total cost (USD) | {summary.get('estimated_cost_usd')} |",
        f"| estimated average cost / document (USD) | {summary.get('estimated_cost_per_document_usd')} |",
        "",
        "## Verified ground-truth accuracy",
        "",
        "Accuracy denominators include only documents with `_ground_truth_status=verified` (legacy gold without a status is treated as verified). Candidate / unverified files are excluded.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| verified GT count | {summary.get('n_verified_documents')} |",
        f"| field accuracy | {summary.get('field_accuracy')} |",
        f"| critical field accuracy | {summary.get('critical_field_accuracy')} |",
        f"| traceability exact matches | {summary.get('traceability_exact_matches')} |",
        f"| traceability exact misses | {summary.get('traceability_exact_misses')} |",
        f"| unsafe auto-accept count | {summary.get('unsafe_auto_accept_count')} |",
        f"| unsafe auto-accept rate | {summary.get('unsafe_auto_accept_rate')} |",
        f"| extraction field misses | {summary.get('extraction_miss_count')} |",
        f"| fail-closed review (gold auto-accept, pred review) | {summary.get('fail_closed_review_count')} |",
        "",
        "## Extraction misses vs fail-closed review",
        "",
        "Extraction misses are verified gold fields the model got wrong or left empty. Fail-closed review is when gold would allow auto-accept but the live policy still sent the document to human review. Fail-closed behaviour is a safety outcome, not an unsafe auto-accept. Unsafe auto-accept is counted only when verified gold requires review and the prediction auto-accepts.",
        "",
        "Do not claim production accuracy from unverified documents.",
        "",
        "## Method",
        "",
        "String fields are compared by exact match after lowercase conversion, whitespace trimming, and internal whitespace collapse. Numeric fields use a tolerance when both gold and predicted values are numeric. Missing prediction fields are counted as incorrect whenever a verified gold value exists.",
        "",
        f"Scored field mismatches written: {failure_count}",
    ]
    if by_bucket:
        lines.extend(["", "## Results by quality bucket", "", "| bucket | n | verified | review_rate | field_accuracy | schema_failures |", "| --- | --- | --- | --- | --- | --- |"])
        for row in by_bucket:
            lines.append(
                f"| {row.get('quality_bucket')} | {row.get('n_documents')} | {row.get('n_verified_documents')} | {row.get('review_rate')} | {row.get('field_accuracy')} | {row.get('schema_failures')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_academic_evaluation(
    *,
    metadata_path: Path,
    predictions_dir: Path,
    out_dir: Path,
    metric: str = "both",
    field_policy_path: Path | None = None,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "doc_id",
            "file_name",
            "quality_bucket",
            "document_type",
            "pages",
            "has_text_layer",
            "ground_truth_path",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"metadata missing required columns: {', '.join(sorted(missing_columns))}")

        for metadata_row in reader:
            doc_id = metadata_row.get("doc_id", "").strip()
            if not doc_id:
                continue

            gt_path_value = metadata_row.get("ground_truth_path", "").strip()
            gt_path = _resolve_ground_truth_path(metadata_path, gt_path_value) if gt_path_value else Path("")
            pred_path = predictions_dir / f"{doc_id}.json"

            gold_payload: dict[str, Any] | None = None
            prediction_payload: dict[str, Any] | None = None
            if gt_path_value and gt_path.exists():
                gold_payload = _read_json(gt_path)
            if pred_path.exists():
                prediction_payload = _read_json(pred_path)

            doc_row, per_field_rows, per_failure_rows = _evaluate_academic_document(
                metadata_row=metadata_row,
                gold_payload=gold_payload,
                prediction_payload=prediction_payload,
            )
            doc_rows.append(doc_row)
            field_rows.extend(per_field_rows)
            failure_rows.extend(per_failure_rows)

    summary = _aggregate_academic(doc_rows)
    by_bucket = _metrics_by_quality_bucket(doc_rows)
    by_field = _metrics_by_field(field_rows)

    _write_rows(out_dir / "metrics_summary.csv", list(summary.keys()), [summary])
    _write_rows(
        out_dir / "metrics_by_quality_bucket.csv",
        ["quality_bucket", *summary.keys()],
        by_bucket,
    )
    _write_rows(
        out_dir / "metrics_by_field.csv",
        ["field", "samples", "correct", "accuracy", "missing_predictions"],
        by_field,
    )
    _write_rows(
        out_dir / "failure_cases.csv",
        ["doc_id", "file_name", "quality_bucket", "field", "reason", "gold_value", "predicted_value"],
        failure_rows,
    )
    _write_eval_report(
        path=out_dir / "eval_report.md",
        metadata_path=metadata_path,
        predictions_dir=predictions_dir,
        summary=summary,
        failure_count=len(failure_rows),
        by_bucket=by_bucket,
    )
    _write_eval_report(
        path=out_dir / "eval_summary.md",
        metadata_path=metadata_path,
        predictions_dir=predictions_dir,
        summary=summary,
        failure_count=len(failure_rows),
        by_bucket=by_bucket,
    )
    observation_fields = [
        "doc_id",
        "file_name",
        "quality_bucket",
        "verified_gold",
        "prediction_missing",
        "review_required",
        "schema_failure",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "fail_closed_review",
        "extraction_misses",
        "unsafe_auto_accept",
    ]
    _write_rows(out_dir / "pipeline_observations.csv", observation_fields, doc_rows)
    metrics_payload = {
        "documents_evaluated": summary.get("n_documents"),
        "metadata_path": str(metadata_path),
        "aggregate": {
            "n": summary.get("n_documents"),
            **{k: v for k, v in summary.items() if k != "n_documents"},
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if metric in {"raw_exact", "business_normalized", "both"}:
        from evaluation.metrics import aggregate_field_results, run_field_evaluation
        from evaluation.policy import load_evaluation_policy
        from evaluation.reporting import field_results_to_rows, write_evaluation_outputs

        policy = load_evaluation_policy(policy_path=field_policy_path)
        enhanced_results = []
        enhanced_rows: list[dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for metadata_row in reader:
                doc_id = metadata_row.get("doc_id", "").strip()
                if not doc_id:
                    continue
                gt_path_value = metadata_row.get("ground_truth_path", "").strip()
                gt_path = _resolve_ground_truth_path(metadata_path, gt_path_value) if gt_path_value else Path("")
                pred_path = predictions_dir / f"{doc_id}.json"
                gold_payload = _read_json(gt_path) if gt_path_value and gt_path.exists() else None
                if not _is_verified_gold(gold_payload):
                    continue
                prediction_payload = _read_json(pred_path) if pred_path.exists() else None
                gold_fields = _flatten_academic_fields(gold_payload or {})
                pred_fields = _flatten_academic_fields(prediction_payload or {})
                per_doc_results = run_field_evaluation(
                    field_names=ACADEMIC_TARGET_FIELDS,
                    gold_fields=gold_fields,
                    pred_fields=pred_fields,
                    policy=policy,
                    prediction_missing=prediction_payload is None,
                )
                enhanced_results.extend(per_doc_results)
                enhanced_rows.extend(
                    field_results_to_rows(
                        per_doc_results,
                        doc_id=doc_id,
                        quality_bucket=metadata_row.get("quality_bucket", ""),
                    )
                )
        aggregate = aggregate_field_results(enhanced_results)
        enhanced_summary = {
            "raw_exact_accuracy": aggregate.raw_exact_accuracy,
            "business_normalized_accuracy": aggregate.business_normalized_accuracy,
            "accuracy_delta": aggregate.accuracy_delta,
            "harmless_normalization_accepts": aggregate.harmless_normalization_accepts,
            "true_mismatches": aggregate.true_mismatches,
            "review_needed": aggregate.review_needed,
            "critical_identifier_mismatches": aggregate.critical_identifier_mismatches,
        }
        if metric == "raw_exact":
            summary = {**summary, **{k: v for k, v in enhanced_summary.items() if k.startswith("raw_exact") or k == "harmless_normalization_accepts"}}
        elif metric == "business_normalized":
            summary = {**summary, **enhanced_summary}
        else:
            summary = {**summary, **enhanced_summary}
        write_evaluation_outputs(
            out_dir=out_dir,
            aggregate=aggregate,
            field_rows=enhanced_rows,
            legacy_summary=summary,
            metadata_path=metadata_path,
            predictions_dir=predictions_dir,
            output_json=output_json,
            output_csv=output_csv,
            output_md=output_md,
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate QualiFlow prediction JSON files against academic gold annotations.")
    parser.add_argument("--metadata", default="data/gold/metadata_20.csv", help="Gold metadata CSV.")
    parser.add_argument("--predictions", default="outputs/predictions/", help="Directory containing <doc_id>.json predictions.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to outputs/eval_runs/<timestamp>/.")
    parser.add_argument(
        "--metric",
        choices=["raw_exact", "business_normalized", "both"],
        default="both",
        help="Evaluation metric mode. Default runs both raw and business-normalized layers.",
    )
    parser.add_argument(
        "--field-policy",
        default=None,
        help="Path to evaluation field policy YAML. Defaults to config/evaluation_field_policy.yaml.",
    )
    parser.add_argument("--output-json", default=None, help="Override enhanced metrics JSON output path.")
    parser.add_argument("--output-csv", default=None, help="Override per-field comparison CSV output path.")
    parser.add_argument("--output-md", default=None, help="Override enhanced Markdown report output path.")
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata)
    predictions_dir = Path(args.predictions)
    out_dir = Path(args.out) if args.out else Path("outputs") / "eval_runs" / _utc_ts()

    if not metadata_path.exists():
        print(f"[error] metadata not found: {metadata_path}", file=sys.stderr)
        return 2
    if not predictions_dir.exists():
        print(f"[error] predictions directory not found: {predictions_dir}", file=sys.stderr)
        return 2

    try:
        summary = run_academic_evaluation(
            metadata_path=metadata_path,
            predictions_dir=predictions_dir,
            out_dir=out_dir,
            metric=args.metric,
            field_policy_path=Path(args.field_policy) if args.field_policy else None,
            output_json=Path(args.output_json) if args.output_json else None,
            output_csv=Path(args.output_csv) if args.output_csv else None,
            output_md=Path(args.output_md) if args.output_md else None,
        )
    except Exception as exc:
        print(f"[error] evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote evaluation outputs to {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
