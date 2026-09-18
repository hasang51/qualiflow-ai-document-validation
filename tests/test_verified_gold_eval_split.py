from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evaluate_outputs import run_academic_evaluation


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "doc_id",
        "file_name",
        "quality_bucket",
        "document_type",
        "pages",
        "has_text_layer",
        "ground_truth_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_unverified_and_candidate_gold_are_excluded_from_accuracy(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()

    verified = {
        "supplier_name": "ACME Steel",
        "document_type": "Mill Test Certificate",
        "heat_number": "H1",
        "lot_number": "L9",
        "grade": "S355J2",
        "yield_strength_mpa": 380.0,
        "tensile_strength_mpa": 500.0,
        "elongation_percentage": 25.0,
        "review_required": True,
        "_ground_truth_status": "verified",
    }
    candidate = {
        "supplier_name": "Should Not Score",
        "heat_number": "WRONG",
        "review_required": False,
        "_ground_truth_status": "candidate",
    }
    _write_json(gold_dir / "doc_verified.json", verified)
    _write_json(gold_dir / "doc_candidate.json", candidate)

    _write_json(
        pred_dir / "doc_verified.json",
        {
            "latency_ms": 1000,
            "llm_usage": {"total_input_tokens": 10, "total_output_tokens": 5, "estimated_cost_usd": 0.01},
            "extraction": {
                "supplier_name": "ACME Steel",
                "document_type": "Mill Test Certificate",
                "items": [
                    {
                        "heat_number": "H1",
                        "lot_number": "L9",
                        "grade": "S355J2",
                        "mechanical_properties": {
                            "yield_strength_mpa": 380.0,
                            "tensile_strength_mpa": 500.0,
                            "elongation_percentage": 25.0,
                        },
                    }
                ],
                "needs_review": True,
                "review_reasons": ["synthetic_fixture"],
            },
        },
    )
    _write_json(
        pred_dir / "doc_candidate.json",
        {
            "latency_ms": 2000,
            "llm_usage": {"total_input_tokens": 20, "total_output_tokens": 8, "estimated_cost_usd": 0.02},
            "extraction": {
                "supplier_name": "Other Corp",
                "document_type": "Something Else Entirely",
                "items": [{"heat_number": "NOPE"}],
                "needs_review": True,
                "review_reasons": ["missing_critical_field:grade"],
            },
        },
    )
    _write_json(
        pred_dir / "doc_unverified.json",
        {
            "latency_ms": 3000,
            "llm_usage": {"total_input_tokens": 30, "total_output_tokens": 9, "estimated_cost_usd": 0.03},
            "extraction": {
                "supplier_name": "Ignored",
                "document_type": "Ignored Type",
                "needs_review": False,
            },
        },
    )

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {
                "doc_id": "doc_verified",
                "file_name": "a.pdf",
                "quality_bucket": "temiz",
                "document_type": "Mill Test Certificate",
                "pages": "1",
                "has_text_layer": "true",
                "ground_truth_path": str(gold_dir / "doc_verified.json"),
            },
            {
                "doc_id": "doc_candidate",
                "file_name": "b.pdf",
                "quality_bucket": "bulanik",
                "document_type": "Mill Test Certificate",
                "pages": "1",
                "has_text_layer": "false",
                "ground_truth_path": str(gold_dir / "doc_candidate.json"),
            },
            {
                "doc_id": "doc_unverified",
                "file_name": "c.pdf",
                "quality_bucket": "bulanik",
                "document_type": "Mill Test Certificate",
                "pages": "1",
                "has_text_layer": "false",
                "ground_truth_path": "",
            },
        ],
    )

    summary = run_academic_evaluation(
        metadata_path=metadata_path,
        predictions_dir=pred_dir,
        out_dir=tmp_path / "out",
        metric="both",
    )

    assert summary["n_documents"] == 3
    assert summary["n_verified_documents"] == 1
    assert summary["n_predictions"] == 3
    assert summary["field_accuracy"] == 1.0
    assert summary["critical_field_accuracy"] == 1.0
    assert summary["traceability_exact_matches"] == 2
    assert summary["traceability_exact_misses"] == 0
    assert summary["unsafe_auto_accept_count"] == 0
    assert summary["unsafe_auto_accept_rate"] == 0.0
    assert summary["review_rate"] == 0.6667
    assert summary["p50_latency_ms"] == 2000.0
    assert summary["p95_latency_ms"] == 3000.0
    assert summary["total_input_tokens"] == 60
    assert summary["estimated_cost_usd"] == 0.06
    assert (tmp_path / "out" / "eval_summary.md").exists()
    assert (tmp_path / "out" / "pipeline_observations.csv").exists()


def test_fail_closed_review_is_not_unsafe_auto_accept(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    _write_json(
        gold_dir / "doc.json",
        {
            "supplier_name": "ACME",
            "document_type": "MTC",
            "heat_number": "H1",
            "review_required": False,
            "_ground_truth_status": "verified",
        },
    )
    _write_json(
        pred_dir / "doc.json",
        {
            "latency_ms": 500,
            "extraction": {
                "supplier_name": "Other",
                "document_type": "MTC",
                "items": [{"heat_number": "H2"}],
                "needs_review": True,
                "review_reasons": ["traceability_unverified"],
            },
        },
    )
    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(
        metadata_path,
        [
            {
                "doc_id": "doc",
                "file_name": "a.pdf",
                "quality_bucket": "temiz",
                "document_type": "MTC",
                "pages": "1",
                "has_text_layer": "true",
                "ground_truth_path": str(gold_dir / "doc.json"),
            }
        ],
    )
    summary = run_academic_evaluation(
        metadata_path=metadata_path,
        predictions_dir=pred_dir,
        out_dir=tmp_path / "out",
        metric="raw_exact",
    )
    assert summary["n_verified_documents"] == 1
    assert summary["unsafe_auto_accept_count"] == 0
    assert summary["fail_closed_review_count"] == 1
    assert summary["extraction_miss_count"] >= 1
    assert summary["traceability_exact_misses"] == 1
