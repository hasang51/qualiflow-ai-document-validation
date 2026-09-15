"""Reproducible batch extraction over a dataset manifest.

For each PDF:
1. profile (:mod:`app.services.document_profiler`)
2. choose route (:mod:`app.services.extraction_router`) — unless ``--force-route``
3. preprocess (:mod:`app.services.preprocessing`) with the chosen route
4. run the multimodal extraction (:mod:`app.services.extraction_pipeline`)
5. capture the validated + confidence-normalised + review-gated result

Outputs (under ``data/batch_runs/<timestamp>/``):
- ``per_document/<document_id>.json`` — extraction + profile + route + latency
- ``summary.csv`` — one row per document (the key thesis artefact)
- ``summary.json`` — aggregate counts
- ``errors.jsonl`` — documents that failed to extract, with error class + message

Flags:
- ``--manifest`` / ``--subset`` — which manifest to iterate
- ``--limit N`` — only process the first N rows (useful for smoke tests)
- ``--dry-run`` — skip the Bedrock call; only profile + route + preprocess stats
- ``--force-route {native_multimodal, rendered_multimodal, preprocessed_multimodal}``
  — bypass the router (Experiment Modes B and C)
- ``--mode`` — short alias for the experiment mode (``B``, ``C``, ``D``)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.batch_policy import BatchRunPolicy

from scripts._dataset_common import (
    DEFAULT_BATCH_RUNS_DIR,
    DEFAULT_GOLD_CANDIDATES_DIR,
    DEFAULT_MANIFEST_DIR,
    ensure_dir,
)

logger = logging.getLogger("qualiflow.batch")

MODE_TO_ROUTE: dict[str, str | None] = {
    "B": "rendered_multimodal",
    "C": "preprocessed_multimodal",
    "D": None,  # router decides
}


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flatten_summary_row(
    document_id: str,
    filename: str,
    *,
    quality_class: str | None,
    route_used: str | None,
    page_count: int | None,
    mode: str,
    extraction: dict | None,
    review: dict | None,
    latency_ms: float,
    status: str,
    error: str | None,
) -> dict:
    items = (extraction or {}).get("items", []) or []
    total_detected = (extraction or {}).get("total_items_detected")
    confidence = (extraction or {}).get("confidence_score")
    needs_review = (extraction or {}).get("needs_review")
    is_compliant = (extraction or {}).get("is_compliant")

    missing_yield = 0
    missing_tensile = 0
    missing_heat = 0
    for item in items:
        mp = item.get("mechanical_properties") or {}
        if mp.get("yield_strength_mpa") is None:
            missing_yield += 1
        if mp.get("tensile_strength_mpa") is None:
            missing_tensile += 1
        if not item.get("heat_number"):
            missing_heat += 1

    structured_reasons = (review or {}).get("structured_reasons") or (review or {}).get("review_reasons") or []
    all_reasons = (review or {}).get("all_reasons") or (extraction or {}).get("review_reasons") or []

    llm_usage = (extraction or {}).get("llm_usage") or {}

    return {
        "document_id": document_id,
        "filename": filename,
        "mode": mode,
        "route_used": route_used,
        "quality_class": quality_class,
        "page_count": page_count,
        "status": status,
        "extraction_status": (extraction or {}).get("status"),
        "validation_status": "COMPLIANT" if is_compliant else ("NON_COMPLIANT" if is_compliant is False else "UNKNOWN"),
        "review_required": needs_review,
        "confidence_score": confidence,
        "raw_model_confidence": (extraction or {}).get("raw_model_confidence"),
        "total_items_detected": total_detected,
        "items_extracted": len(items),
        "missing_heat_numbers": missing_heat,
        "missing_yield_values": missing_yield,
        "missing_tensile_values": missing_tensile,
        "structured_review_reasons": "|".join(structured_reasons),
        "all_review_reasons": "|".join(all_reasons),
        "latency_ms": round(latency_ms, 1),
        "error": error or "",
        "input_tokens": llm_usage.get("total_input_tokens", ""),
        "output_tokens": llm_usage.get("total_output_tokens", ""),
        "pages_sent": llm_usage.get("pages_sent", ""),
    }


def _run_single(
    row: dict,
    *,
    mode: str,
    force_route: str | None,
    dry_run: bool,
    policy: "BatchRunPolicy | None" = None,
    rate_limit_retry_wait_s: float = 65.0,
) -> tuple[dict, dict | None, str | None]:
    """Process one document. Returns ``(summary_row, per_document_json, error)``.

    ``per_document_json`` is None for dry runs / errors. ``error`` is a string
    if processing failed.
    """

    from app.services.batch_policy import BatchRunPolicy
    from app.services.document_profiler import profile_document
    from app.services.extraction_router import choose_route, force_route as manual_route
    from app.services.page_selector import select_pages
    from app.services.preprocessing import preprocess_pdf

    if policy is None:
        policy = BatchRunPolicy.from_env()

    document_id = row["document_id"]
    filename = row["filename"]
    abs_path = row["abs_path"]
    started = time.perf_counter()

    try:
        profile = profile_document(abs_path, document_id=document_id)
        if force_route is not None:
            decision = manual_route(force_route)
        else:
            decision = choose_route(profile)

        # Cost guard: check policy skip before preprocessing or any API call.
        skip, skip_reason = policy.should_skip_document(
            profile.quality_class, profile.blur_score
        )
        if skip and not dry_run:
            latency_ms = (time.perf_counter() - started) * 1000.0
            summary_row = _flatten_summary_row(
                document_id=document_id,
                filename=filename,
                quality_class=profile.quality_class,
                route_used=decision.runtime_route,
                page_count=profile.page_count,
                mode=mode,
                extraction=None,
                review=None,
                latency_ms=latency_ms,
                status="SKIPPED",
                error=skip_reason,
            )
            return summary_row, None, None

        if dry_run:
            latency_ms = (time.perf_counter() - started) * 1000.0
            summary_row = _flatten_summary_row(
                document_id=document_id,
                filename=filename,
                quality_class=profile.quality_class,
                route_used=decision.runtime_route,
                page_count=profile.page_count,
                mode=mode,
                extraction=None,
                review=None,
                latency_ms=latency_ms,
                status="DRY_RUN",
                error=None,
            )
            return summary_row, None, None

        from app.config import settings
        from app.services.extraction_pipeline import run_multi_stage_extraction
        from app.services.storage import artifact_dir_for_hash

        # Page cap: use policy limit for this quality class.
        max_pages = min(
            policy.max_pages_for_quality(profile.quality_class),
            settings.max_pages_for_llm,
        )

        artifact_dir = artifact_dir_for_hash(row.get("sha256") or document_id)
        preprocessing_meta: dict = {
            "document_id": document_id,
            "filename": filename,
            "profile": profile.to_dict(),
            "route_decision": decision.to_dict(),
        }

        processed_pages, pre_meta = preprocess_pdf(
            abs_path,
            artifact_dir=artifact_dir,
            route=decision.runtime_route,
            max_pages=max_pages,
        )
        preprocessing_meta.update(pre_meta)

        # Page selection: pick the most informative pages within the cap.
        page_selection = select_pages(
            processed_pages,
            quality_class=profile.quality_class,
            max_pages=max_pages,
        )
        selected_pages = page_selection.selected_pages
        preprocessing_meta["page_selection"] = {
            "total_pages": page_selection.total_pages,
            "selected_count": len(page_selection.selected_indices),
            "selected_indices": page_selection.selected_indices,
            "pages_skipped": page_selection.pages_skipped,
            "reason": page_selection.reason,
        }
        if page_selection.pages_skipped:
            logger.info(
                "Page selector skipped %d/%d pages for %s (%s)",
                page_selection.pages_skipped,
                page_selection.total_pages,
                filename,
                page_selection.reason,
            )

        # Extraction — one bounded retry on rate limit.
        try:
            extraction = run_multi_stage_extraction(
                selected_pages, preprocessing_meta, profile=profile
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            lowered = f"{type(exc).__name__} {exc}".lower()
            is_rate_limit = "ratelimit" in lowered or "rate_limit" in lowered or "429" in lowered
            if not is_rate_limit:
                raise
            stop, stop_reason = policy.record_rate_limit()
            if stop:
                raise RuntimeError(f"batch_halted: {stop_reason}") from exc
            logger.warning(
                "rate-limit hit while extracting %s; sleeping %.0fs before one retry",
                filename,
                rate_limit_retry_wait_s,
            )
            time.sleep(rate_limit_retry_wait_s)
            extraction = run_multi_stage_extraction(
                selected_pages, preprocessing_meta, profile=profile
            )

        # Record usage and count this as a new live doc.
        llm_usage = preprocessing_meta.get("llm_usage") or {}
        policy.record_usage(
            input_tokens=llm_usage.get("total_input_tokens", 0),
            output_tokens=llm_usage.get("total_output_tokens", 0),
        )
        policy.record_new_live_doc()

        latency_ms = (time.perf_counter() - started) * 1000.0

        extraction_dict = extraction.model_dump()
        review_dict = preprocessing_meta.get("review_policy") or {
            "needs_review": extraction.needs_review,
            "structured_reasons": [],
            "all_reasons": extraction.review_reasons,
        }

        per_document = {
            "document_id": document_id,
            "filename": filename,
            "mode": mode,
            "route_used": decision.runtime_route,
            "profile": profile.to_dict(),
            "route_decision": decision.to_dict(),
            "latency_ms": round(latency_ms, 1),
            "extraction": extraction_dict,
            "review_policy": review_dict,
            "llm_usage": llm_usage,
            "page_selection": preprocessing_meta.get("page_selection"),
            "preprocessing_summary": {
                "page_count": pre_meta.get("page_count"),
                "route_used": pre_meta.get("route_used"),
                "adaptive_preprocessing_summary": pre_meta.get("adaptive_preprocessing_summary"),
                "confidence_assessment": preprocessing_meta.get("confidence_assessment"),
            },
        }

        summary_row = _flatten_summary_row(
            document_id=document_id,
            filename=filename,
            quality_class=profile.quality_class,
            route_used=decision.runtime_route,
            page_count=profile.page_count,
            mode=mode,
            extraction={**extraction_dict, "llm_usage": llm_usage},
            review=review_dict,
            latency_ms=latency_ms,
            status="OK",
            error=None,
        )
        return summary_row, per_document, None

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.exception("batch extraction failed for %s", filename)
        summary_row = _flatten_summary_row(
            document_id=document_id,
            filename=filename,
            quality_class=None,
            route_used=force_route,
            page_count=None,
            mode=mode,
            extraction=None,
            review=None,
            latency_ms=latency_ms,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
        )
        return summary_row, None, f"{type(exc).__name__}: {exc}"


SUMMARY_FIELDS = [
    "document_id",
    "filename",
    "mode",
    "route_used",
    "quality_class",
    "page_count",
    "status",
    "extraction_status",
    "validation_status",
    "review_required",
    "confidence_score",
    "raw_model_confidence",
    "total_items_detected",
    "items_extracted",
    "missing_heat_numbers",
    "missing_yield_values",
    "missing_tensile_values",
    "structured_review_reasons",
    "all_review_reasons",
    "latency_ms",
    "error",
    "input_tokens",
    "output_tokens",
    "pages_sent",
]


def _aggregate(summary_rows: list[dict]) -> dict:
    from statistics import mean, median

    latencies = [row["latency_ms"] for row in summary_rows if isinstance(row.get("latency_ms"), (int, float))]
    ok_rows = [row for row in summary_rows if row["status"] == "OK"]
    review_rows = [row for row in ok_rows if row.get("review_required")]
    p95 = 0.0
    if latencies:
        sorted_latencies = sorted(latencies)
        idx = max(0, min(len(sorted_latencies) - 1, int(round(0.95 * (len(sorted_latencies) - 1)))))
        p95 = sorted_latencies[idx]

    route_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for row in summary_rows:
        route_counts[row.get("route_used") or "unknown"] = route_counts.get(row.get("route_used") or "unknown", 0) + 1
        quality_counts[row.get("quality_class") or "unknown"] = quality_counts.get(row.get("quality_class") or "unknown", 0) + 1

    return {
        "total_documents": len(summary_rows),
        "ok": len(ok_rows),
        "errors": sum(1 for r in summary_rows if r["status"] == "ERROR"),
        "dry_run": sum(1 for r in summary_rows if r["status"] == "DRY_RUN"),
        "review_required": len(review_rows),
        "review_rate": (len(review_rows) / max(1, len(ok_rows))) if ok_rows else 0.0,
        "stp_rate": ((len(ok_rows) - len(review_rows)) / max(1, len(ok_rows))) if ok_rows else 0.0,
        "mean_latency_ms": round(mean(latencies), 1) if latencies else 0.0,
        "median_latency_ms": round(median(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(p95, 1),
        "route_counts": route_counts,
        "quality_counts": quality_counts,
    }


def run(
    manifest_path: Path,
    *,
    output_root: Path,
    mode: str,
    force_route: str | None,
    dry_run: bool,
    limit: int | None,
    timestamp: str | None,
    inter_doc_sleep_s: float = 0.0,
    resume: bool = False,
    policy: "BatchRunPolicy | None" = None,
) -> Path:
    from app.services.batch_policy import BatchRunPolicy

    if policy is None:
        policy = BatchRunPolicy.from_env()

    rows = _load_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    ts = timestamp or _utc_ts()
    run_dir = ensure_dir(output_root / ts)
    per_doc_dir = ensure_dir(run_dir / "per_document")
    errors_path = run_dir / "errors.jsonl"
    summary_csv_path = run_dir / "summary.csv"
    summary_json_path = run_dir / "summary.json"
    usage_csv_path = run_dir / "usage.csv"
    config_path = run_dir / "config.json"

    config = {
        "manifest": str(manifest_path.resolve()),
        "mode": mode,
        "force_route": force_route,
        "dry_run": dry_run,
        "limit": limit,
        "timestamp": ts,
        "total_candidates": len(rows),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows: list[dict] = []
    usage_rows: list[dict] = []
    error_rows: list[dict] = []
    total = len(rows)
    batch_halted = False
    halt_reason = ""

    for idx, row in enumerate(rows, start=1):
        per_doc_out = per_doc_dir / f"{row['document_id']}.json"
        if (resume or policy.use_resume) and per_doc_out.exists():
            print(
                f"[{idx}/{total}] {row['filename']}  (resume: reusing {per_doc_out.name})"
            )
            try:
                cached = json.loads(per_doc_out.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cached = None
            if cached is not None:
                extraction_dict = cached.get("extraction") or {}
                review_dict = cached.get("review_policy") or {}
                summary_rows.append(
                    _flatten_summary_row(
                        document_id=cached.get("document_id", row["document_id"]),
                        filename=cached.get("filename", row["filename"]),
                        quality_class=(cached.get("profile") or {}).get("quality_class"),
                        route_used=cached.get("route_used"),
                        page_count=(cached.get("profile") or {}).get("page_count"),
                        mode=cached.get("mode", mode),
                        extraction={**extraction_dict, "llm_usage": cached.get("llm_usage")},
                        review=review_dict,
                        latency_ms=float(cached.get("latency_ms") or 0.0),
                        status="OK",
                        error=None,
                    )
                )
                continue

        # Budget guard before starting this document.
        if not dry_run:
            ok, budget_reason = policy.budget_ok()
            if not ok:
                batch_halted = True
                halt_reason = budget_reason
                print(f"[HALT] Budget exhausted before [{idx}/{total}]: {halt_reason}")
                break

        print(f"[{idx}/{total}] {row['filename']}  (mode={mode}, dry_run={dry_run})")
        summary_row, per_doc_json, error = _run_single(
            row, mode=mode, force_route=force_route, dry_run=dry_run, policy=policy
        )
        summary_rows.append(summary_row)
        if per_doc_json is not None:
            per_doc_out.write_text(json.dumps(per_doc_json, indent=2, ensure_ascii=False), encoding="utf-8")
            # Write one usage row per live doc.
            usage_rows.append({
                "document_id": row["document_id"],
                "filename": row["filename"],
                "input_tokens": summary_row.get("input_tokens", ""),
                "output_tokens": summary_row.get("output_tokens", ""),
                "pages_sent": summary_row.get("pages_sent", ""),
                "estimated_cost_usd": round(
                    (int(summary_row.get("input_tokens") or 0) / 1000.0 * 0.003)
                    + (int(summary_row.get("output_tokens") or 0) / 1000.0 * 0.015),
                    6,
                ),
                "latency_ms": summary_row.get("latency_ms", ""),
            })
        if error is not None:
            error_rows.append(
                {
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "error": error,
                }
            )
            with errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_rows[-1], ensure_ascii=False) + "\n")

        status = summary_row.get("status")
        if status == "OK":
            print(
                f"    -> route={summary_row['route_used']}, quality={summary_row['quality_class']}, "
                f"items={summary_row['items_extracted']}/{summary_row['total_items_detected']}, "
                f"confidence={summary_row['confidence_score']}, review={summary_row['review_required']}, "
                f"{summary_row['latency_ms']}ms"
            )
        elif status == "DRY_RUN":
            print(
                f"    -> DRY route={summary_row['route_used']}, quality={summary_row['quality_class']}, "
                f"{summary_row['latency_ms']}ms"
            )
        else:
            print(f"    -> ERROR: {summary_row['error']}")

        # Pace requests to stay under Bedrock's per-minute token rate limit.
        # Applies to live runs only (not dry-run) and not to the final iteration.
        if inter_doc_sleep_s > 0 and not dry_run and idx < total and summary_row.get("status") == "OK":
            print(f"    (sleeping {inter_doc_sleep_s:.0f}s to stay under rate limit)")
            time.sleep(inter_doc_sleep_s)

    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for summary_row in summary_rows:
            writer.writerow(summary_row)

    # Write per-document usage CSV.
    if usage_rows:
        usage_fields = ["document_id", "filename", "input_tokens", "output_tokens", "pages_sent", "estimated_cost_usd", "latency_ms"]
        with usage_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer2 = csv.DictWriter(handle, fieldnames=usage_fields)
            writer2.writeheader()
            for urow in usage_rows:
                writer2.writerow(urow)

    aggregate = _aggregate(summary_rows)
    policy_summary = policy.usage_summary() if not dry_run else {}
    if batch_halted:
        policy_summary["halted"] = True
        policy_summary["halt_reason"] = halt_reason
    summary_json_path.write_text(
        json.dumps(
            {"config": config, "aggregate": aggregate, "policy_summary": policy_summary},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[ok] batch outputs: {run_dir}")
    print(f"[ok] aggregate: {json.dumps(aggregate, indent=2)}")
    if policy_summary:
        print(f"[ok] policy/usage: {json.dumps(policy_summary, indent=2)}")
    if batch_halted:
        print(f"[HALT] batch halted: {halt_reason}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run batch extraction across a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "BatchRunPolicy environment overrides (defaults in parentheses): "
            "QUALIFLOW_MAX_LIVE_DOCS (5), QUALIFLOW_BUDGET_USD (2.00), "
            "QUALIFLOW_SKIP_SEVERE_SCAN (1), QUALIFLOW_BATCH_SLEEP_S (45), "
            "QUALIFLOW_RESUME (1 — skip documents when per_document/<id>.json already exists)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_DIR / "manifest.jsonl"),
        help="Full dataset manifest (from build_manifest.py).",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Alternative subset manifest (e.g. gold_candidates_manifest.jsonl).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    parser.add_argument("--dry-run", action="store_true", help="Skip the Bedrock call; profile + route only.")
    parser.add_argument(
        "--force-route",
        choices=["native_multimodal", "rendered_multimodal", "preprocessed_multimodal"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=["B", "C", "D"],
        default="D",
        help="Experiment mode (B=no-routing, C=preprocessed-fixed, D=routed-hybrid).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_BATCH_RUNS_DIR),
    )
    parser.add_argument("--timestamp", default=None, help="Override the run timestamp (for reruns).")
    parser.add_argument(
        "--inter-doc-sleep-s",
        type=float,
        default=float(os.getenv("QUALIFLOW_BATCH_SLEEP_S", "35.0")),
        help=(
            "Seconds to sleep between documents so the batch stays under "
            "Bedrock's per-minute token rate limit. Default 35s. Set to 0 "
            "to disable pacing."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If a per-document JSON already exists under the given timestamp, "
            "reuse it instead of re-running extraction. Useful after a rate-"
            "limit abort."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("QUALIFLOW_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from app.services.batch_policy import BatchRunPolicy

    force_route = args.force_route or MODE_TO_ROUTE.get(args.mode)
    manifest_path = Path(args.subset) if args.subset else Path(args.manifest)
    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    # Default candidate-subset helper: if no subset provided but gold candidates file exists,
    # print a gentle hint.
    if args.subset is None:
        candidate_default = DEFAULT_GOLD_CANDIDATES_DIR / "gold_candidates_manifest.jsonl"
        if candidate_default.exists():
            print(f"[hint] gold candidate manifest available at {candidate_default} (pass --subset to use it)")

    # Build policy from env (all pilot defaults) then override with CLI flags.
    policy = BatchRunPolicy.from_env()
    policy.inter_doc_sleep_s = max(0.0, args.inter_doc_sleep_s)

    run(
        manifest_path=manifest_path,
        output_root=Path(args.output_root),
        mode=args.mode,
        force_route=force_route,
        dry_run=args.dry_run,
        limit=args.limit,
        timestamp=args.timestamp,
        inter_doc_sleep_s=max(0.0, args.inter_doc_sleep_s),
        resume=args.resume,
        policy=policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
