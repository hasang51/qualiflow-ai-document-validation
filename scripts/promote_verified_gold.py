"""Convert a reviewer-completed annotation sheet into verified gold.

Usage (after a human fills the ``verified_*`` columns in ``annotation_sheet.csv``)::

    python -m scripts.promote_verified_gold \
        --annotation-sheet data/gold_candidates/annotation_sheet.csv \
        --output-dir data/gold_verified/

This reads rows where ``review_status`` is one of the accepted values
(``VERIFIED``, ``ACCEPTED``, or ``OK``), maps the ``verified_*`` columns to
the normalised gold schema, and writes:

- ``data/gold_verified/annotations.csv`` — flat CSV, one row per document.
- ``data/gold_verified/annotations.jsonl`` — one JSON object per line.

Rows where ``review_status`` is still ``PENDING_REVIEW`` or blank are silently
skipped and reported.

The output files can be fed into ``scripts/run_eval.py`` once converted to the
gold metadata + per-document prediction layout used by the academic evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scripts._dataset_common import DEFAULT_GOLD_VERIFIED_DIR, ensure_dir

ACCEPTED_STATUSES = {"VERIFIED", "ACCEPTED", "OK"}

GOLD_FIELDS = [
    "document_id",
    "filename",
    "quality_class",
    "route_used",
    "supplier_name",
    "document_type",
    "certificate_date",
    "is_compliant",
    "heat_numbers",
    "grades",
    "yield_strength_mpa",
    "tensile_strength_mpa",
    "elongation_percentage",
    "review_status",
    "reviewer_notes",
]


def _normalise_row(row: dict) -> dict | None:
    status = (row.get("review_status") or "").strip().upper()
    if status not in ACCEPTED_STATUSES:
        return None

    def pick(verified_col: str, extracted_col: str) -> str:
        v = (row.get(verified_col) or "").strip()
        return v if v else (row.get(extracted_col) or "").strip()

    return {
        "document_id": (row.get("document_id") or "").strip(),
        "filename": (row.get("filename") or "").strip(),
        "quality_class": (row.get("quality_class") or "").strip(),
        "route_used": (row.get("route_used") or "").strip(),
        "supplier_name": pick("verified_supplier_name", "extracted_supplier_name"),
        "document_type": pick("verified_document_type", "extracted_document_type"),
        "certificate_date": pick("verified_certificate_date", "extracted_certificate_date"),
        "is_compliant": pick("verified_is_compliant", "extracted_is_compliant"),
        "heat_numbers": pick("verified_heat_numbers", "extracted_heat_numbers"),
        "grades": pick("verified_grades", "extracted_grades"),
        "yield_strength_mpa": pick("verified_yield_strength_mpa", "extracted_yield_strength_mpa"),
        "tensile_strength_mpa": pick("verified_tensile_strength_mpa", "extracted_tensile_strength_mpa"),
        "elongation_percentage": pick("verified_elongation_percentage", "extracted_elongation_percentage"),
        "review_status": status,
        "reviewer_notes": (row.get("reviewer_notes") or "").strip(),
    }


def run(annotation_sheet: Path, output_dir: Path) -> tuple[int, int]:
    """Returns ``(accepted_count, skipped_count)``."""
    ensure_dir(output_dir)
    accepted: list[dict] = []
    skipped = 0

    with annotation_sheet.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = (row.get("document_id") or "").strip()
            if doc_id.startswith("#"):
                continue  # banner row
            normalised = _normalise_row(row)
            if normalised is None:
                skipped += 1
            else:
                accepted.append(normalised)

    csv_path = output_dir / "annotations.csv"
    jsonl_path = output_dir / "annotations.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        for row in accepted:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(accepted), skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote reviewer-verified annotation rows to final gold truth."
    )
    parser.add_argument(
        "--annotation-sheet",
        default="data/gold_candidates/annotation_sheet.csv",
        help="Path to the annotation_sheet.csv filled by a reviewer.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_GOLD_VERIFIED_DIR),
        help="Where to write annotations.{csv,jsonl}.",
    )
    args = parser.parse_args(argv)

    sheet = Path(args.annotation_sheet)
    if not sheet.exists():
        print(f"[error] annotation sheet not found: {sheet}", file=sys.stderr)
        return 2

    accepted, skipped = run(sheet, Path(args.output_dir))
    print(f"[ok] accepted rows: {accepted}")
    print(f"[ok] skipped rows (not yet verified): {skipped}")
    if accepted == 0:
        print(
            "[warn] no rows were accepted. "
            "Set review_status to VERIFIED, ACCEPTED, or OK in the annotation sheet.",
            file=sys.stderr,
        )
        return 1
    print(f"[ok] gold written to: {args.output_dir}")
    print(
        "[next] run: python -m scripts.run_eval "
        "--metadata data/gold/metadata_20.csv "
        "--predictions outputs/predictions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
