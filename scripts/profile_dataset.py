"""Run the deterministic document profiler across every discovered PDF.

No Bedrock calls are made here. Input is ``data/manifests/documents_discovery.jsonl``
(produced by :mod:`scripts.discover_documents`), output is a set of profile
manifests suitable for downstream manifest building and gold-candidate
selection.

Usage::

    python -m scripts.profile_dataset
    python -m scripts.profile_dataset --discovery data/manifests/documents_discovery.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts._dataset_common import DEFAULT_MANIFEST_DIR, ensure_dir


def _load_discovery(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _profile_one(abs_path: str, document_id: str) -> dict:
    # Import lazily so ``-h`` works even if CV / pdf2image can't load.
    from app.services.document_profiler import profile_document

    profile = profile_document(abs_path, document_id=document_id)
    return profile.to_dict()


def run(discovery_path: Path, output_dir: Path, limit: int | None) -> tuple[Path, Path, Path]:
    ensure_dir(output_dir)
    rows = _load_discovery(discovery_path)
    if limit is not None:
        rows = rows[:limit]

    profile_jsonl = output_dir / "documents_manifest.jsonl"
    profile_csv = output_dir / "documents_manifest.csv"
    summary_csv = output_dir / "profile_summary.csv"

    quality_counter: Counter = Counter()
    text_layer_counter: Counter = Counter()

    fieldnames = [
        "document_id",
        "filename",
        "page_count",
        "has_text_layer",
        "text_density",
        "blur_score",
        "noise_score",
        "table_presence_hint",
        "quality_class",
        "reasons",
    ]

    with profile_jsonl.open("w", encoding="utf-8") as jsonl_out, profile_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_out:
        csv_writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        csv_writer.writeheader()
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            abs_path = row.get("abs_path")
            filename = row.get("filename", "<unknown>")
            doc_id = row.get("document_id") or Path(abs_path or filename).stem
            started = time.perf_counter()
            try:
                profile = _profile_one(abs_path, doc_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] profiling failed for {filename}: {exc}", file=sys.stderr)
                profile = {
                    "document_id": doc_id,
                    "filename": filename,
                    "page_count": 0,
                    "has_text_layer": False,
                    "text_density": 0.0,
                    "blur_score": 0.0,
                    "noise_score": 0.0,
                    "table_presence_hint": False,
                    "quality_class": "noisy_scan",
                    "reasons": [f"profile_error:{type(exc).__name__}"],
                }
            elapsed = time.perf_counter() - started
            profile_out = {**profile, "sha256": row.get("sha256"), "abs_path": abs_path, "size_bytes": row.get("size_bytes")}
            jsonl_out.write(json.dumps(profile_out, ensure_ascii=False) + "\n")

            csv_row = {key: profile.get(key) for key in fieldnames if key != "reasons"}
            csv_row["reasons"] = "|".join(profile.get("reasons", []))
            csv_writer.writerow(csv_row)

            quality_counter[profile["quality_class"]] += 1
            text_layer_counter[bool(profile["has_text_layer"])] += 1
            print(
                f"[{idx}/{total}] {filename} -> {profile['quality_class']} "
                f"(pages={profile['page_count']}, text_layer={profile['has_text_layer']}, "
                f"blur={profile['blur_score']:.1f}, noise={profile['noise_score']:.1f}) "
                f"[{elapsed:.1f}s]"
            )

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_documents", total])
        for quality_class, count in sorted(quality_counter.items()):
            writer.writerow([f"quality_class:{quality_class}", count])
        writer.writerow(["has_text_layer:true", text_layer_counter.get(True, 0)])
        writer.writerow(["has_text_layer:false", text_layer_counter.get(False, 0)])

    return profile_jsonl, profile_csv, summary_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile every discovered PDF (no API calls).")
    parser.add_argument(
        "--discovery",
        default=str(DEFAULT_MANIFEST_DIR / "documents_discovery.jsonl"),
        help="Path to the discovery JSONL produced by discover_documents.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MANIFEST_DIR),
        help=f"Where to write profile manifests (default: {DEFAULT_MANIFEST_DIR}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Profile only the first N documents.")
    args = parser.parse_args(argv)

    discovery_path = Path(args.discovery)
    if not discovery_path.exists():
        print(f"[error] discovery file not found: {discovery_path}", file=sys.stderr)
        return 2

    profile_jsonl, profile_csv, summary_csv = run(
        discovery_path=discovery_path,
        output_dir=Path(args.output_dir),
        limit=args.limit,
    )
    print(f"[ok] wrote {profile_jsonl}")
    print(f"[ok] wrote {profile_csv}")
    print(f"[ok] wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
