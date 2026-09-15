from __future__ import annotations

from pathlib import Path

from scripts.evaluate_outputs import run_academic_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_m2_gold_eval_hard_gate(tmp_path: Path):
    summary = run_academic_evaluation(
        metadata_path=ROOT / "data" / "gold" / "m2" / "metadata.csv",
        predictions_dir=ROOT / "data" / "gold" / "m2" / "predictions",
        out_dir=tmp_path,
    )
    assert summary["n_documents"] == 6
    assert summary["unsafe_auto_accept_rate"] == 0.0
    assert summary["schema_failures"] == 1
    assert summary["review_rate"] == 1.0
    assert "field_accuracy" in summary
    assert "critical_field_accuracy" in summary
    assert "estimated_cost_usd" in summary
    assert "average_latency_ms" in summary
    assert summary["total_input_tokens"] > 0
