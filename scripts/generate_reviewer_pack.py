"""Generate a compact reviewer-facing markdown summary for a set of live-extracted docs.

Outputs:
- ``data/gold_candidates/reviewer_pack.md`` — one section per document
- Refreshes ``data/gold_candidates/annotation_sheet.csv`` from the live batch run

Usage::

    python -m scripts.generate_reviewer_pack --run-dir data/batch_runs/20260420T_live_full

The pack shows:
- document metadata (filename, quality_class, route_used, page_count)
- extracted critical fields
- confidence and validation result
- review reasons
- blank override columns for the reviewer

IMPORTANT: All extracted values are PREANNOTATED — NOT VERIFIED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts._dataset_common import DEFAULT_BATCH_RUNS_DIR, DEFAULT_GOLD_CANDIDATES_DIR, ensure_dir


BANNER = (
    "> **PREANNOTATED — NOT VERIFIED — REQUIRES HUMAN REVIEW**\n"
    "> The values below were extracted by the multimodal pipeline.\n"
    "> Fill the `verified_*` columns in `annotation_sheet.csv` and run `scripts.promote_verified_gold`.\n"
)


def _fmt_val(v) -> str:
    if v is None:
        return "_null_"
    if v is True:
        return "✓ True"
    if v is False:
        return "✗ False"
    return str(v)


def _doc_section(data: dict) -> str:
    filename = data.get("filename", "?")
    doc_id = data.get("document_id", "?")
    profile = data.get("profile") or {}
    extraction = data.get("extraction") or {}
    review = data.get("review_policy") or {}
    llm_usage = data.get("llm_usage") or {}
    page_sel = data.get("page_selection") or {}

    items = extraction.get("items") or []
    heats = "|".join(str(i.get("heat_number")) for i in items if i.get("heat_number"))
    grades = "|".join(str(i.get("grade")) for i in items if i.get("grade"))

    def yields_str():
        vals = []
        for item in items:
            mp = item.get("mechanical_properties") or {}
            if mp.get("yield_strength_mpa") is not None:
                vals.append(str(mp["yield_strength_mpa"]))
        return "|".join(vals) or "_null_"

    def tensile_str():
        vals = []
        for item in items:
            mp = item.get("mechanical_properties") or {}
            if mp.get("tensile_strength_mpa") is not None:
                vals.append(str(mp["tensile_strength_mpa"]))
        return "|".join(vals) or "_null_"

    def elong_str():
        vals = []
        for item in items:
            mp = item.get("mechanical_properties") or {}
            if mp.get("elongation_percentage") is not None:
                vals.append(str(mp["elongation_percentage"]))
        return "|".join(vals) or "_null_"

    reasons = (review.get("structured_reasons") or []) + (review.get("all_reasons") or [])
    review_reasons_str = " | ".join(reasons) if reasons else "—"

    cost_approx = ""
    if llm_usage:
        inp = llm_usage.get("total_input_tokens", 0)
        out = llm_usage.get("total_output_tokens", 0)
        usd = (inp / 1000.0 * 0.003) + (out / 1000.0 * 0.015)
        cost_approx = f"~${usd:.4f} ({inp} in / {out} out tokens)"

    lines = [
        f"## {filename}",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| document_id | `{doc_id}` |",
        f"| quality_class | `{profile.get('quality_class', '?')}` |",
        f"| page_count | {profile.get('page_count', '?')} |",
        f"| route_used | `{data.get('route_used', '?')}` |",
        f"| pages_sent | {page_sel.get('selected_count', llm_usage.get('pages_sent', '?'))} / {page_sel.get('total_pages', profile.get('page_count', '?'))} |",
        f"| blur_score | {profile.get('blur_score', '?')} |",
        f"| noise_score | {profile.get('noise_score', '?')} |",
        f"| latency_ms | {data.get('latency_ms', '?')} |",
        f"| LLM cost (approx) | {cost_approx or '—'} |",
        "",
        "**Extracted values** _(PREANNOTATED — NOT VERIFIED)_",
        "",
        "| field | extracted | verified (fill this) |",
        "| --- | --- | --- |",
        f"| supplier_name | {_fmt_val(extraction.get('supplier_name'))} | |",
        f"| document_type | {_fmt_val(extraction.get('document_type'))} | |",
        f"| certificate_date | {_fmt_val(extraction.get('certificate_date'))} | |",
        f"| is_compliant | {_fmt_val(extraction.get('is_compliant'))} | |",
        f"| confidence_score | {_fmt_val(extraction.get('confidence_score'))} | |",
        f"| heat_numbers | {heats or '_null_'} | |",
        f"| grades | {grades or '_null_'} | |",
        f"| yield_strength_mpa | {yields_str()} | |",
        f"| tensile_strength_mpa | {tensile_str()} | |",
        f"| elongation_percentage | {elong_str()} | |",
        "",
        f"**Review decision**: needs_review=`{review.get('needs_review', '?')}`",
        "",
        f"**Review reasons**: {review_reasons_str}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def run(run_dir: Path, output_dir: Path) -> Path:
    per_doc_dir = run_dir / "per_document"
    if not per_doc_dir.exists():
        raise FileNotFoundError(f"No per_document/ under {run_dir}")

    live_docs = sorted(per_doc_dir.glob("*.json"))
    if not live_docs:
        raise FileNotFoundError(f"No per-document JSONs found in {per_doc_dir}")

    ensure_dir(output_dir)
    sections = [
        "# QualiFlow — Reviewer Pack\n",
        BANNER,
        f"**Run dir**: `{run_dir}`  \n**Live docs**: {len(live_docs)}\n\n---\n",
    ]
    for f in live_docs:
        data = json.loads(f.read_text(encoding="utf-8"))
        sections.append(_doc_section(data))

    pack_path = output_dir / "reviewer_pack.md"
    pack_path.write_text("\n".join(sections), encoding="utf-8")
    return pack_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a human reviewer pack from a batch run.")
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_BATCH_RUNS_DIR / "20260420T_live_full"),
        help="Path to the batch run directory containing per_document/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GOLD_CANDIDATES_DIR),
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        alt = DEFAULT_BATCH_RUNS_DIR / run_dir.name
        if alt.exists():
            run_dir = alt
        else:
            print(f"[error] run dir not found: {args.run_dir}", file=sys.stderr)
            return 2

    pack_path = run(run_dir, Path(args.output_dir))
    print(f"[ok] reviewer pack: {pack_path}")
    print("[remember] values are PREANNOTATED — NOT VERIFIED — REQUIRES HUMAN REVIEW.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
