from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException

from app.config import settings
from app.services.extraction_providers.base import ExtractionOutputError, ExtractionProviderError
from app.domain.document_context import build_document_context, propagate_context_to_rows
from app.domain.field_mapping_registry import CANONICAL_FIELDS, explain_mapping, get_registry_snapshot, resolve_canonical_field
from app.domain.header_propagation import apply_updates, propagate_to_rows
from app.domain.numeric_parser import parse_mechanical_properties
from app.domain.outcome_taxonomy import NEEDS_REVIEW, aggregate_document_outcome
from app.domain.review_invariants import apply_review_invariants, review_tokens_from_meta
from app.schemas.extraction import ExtractedItem, MechanicalProperties, UniversalDocumentExtraction
from app.services.canonical_reconciler import apply_canonical_reconciliation
from app.services.extraction_finalizer import (
    apply_canonical_finalization_to_extraction,
    apply_reconcile_to_extraction,
    finalize_decision,
    finalize_extraction_fields,
)
from app.services.extraction_router import RouteDecision
from app.services.semantic_normalizer import normalize_rows_semantics
from app.services.confidence import normalize_confidence
from app.services.document_profiler import DocumentProfile
from app.services.preprocessing import EncodedVariant, ProcessedPage
from app.services.review_policy import apply_review_policy
from app.services.mechanical_table_mapper import apply_mechanical_table_mapping
from app.services.identity_field_mapper import apply_identity_field_mapping
from app.services.row_shape_normalizer import (
    backfill_single_item_context,
    collapse_alternative_classification_rows,
    collapse_vertical_mechanical_rows,
)
from app.services.identifier_verification_guard import apply_identifier_verification_guard
from app.services.traceability import sanitize_unverified_traceability_for_user, validate_traceability
from app.services.validator import validate_document

logger = logging.getLogger("qualiflow.pipeline")

CRITICAL_IDENTIFIER_FIELDS = (
    "heat_number",
    "batch_number",
    "lot_number",
    "colata_number",
    "cast_number",
    "charge_number",
    "coil_number",
    "item_id",
    "pipe_id",
    "certificate_number",
    "order_number",
)
DEGRADED_IDENTIFIER_QUALITY_CLASSES = {"noisy_scan", "degraded_scan", "severe_scan"}
IDENTIFIER_CONFIDENCE_THRESHOLD = 0.80
SCHEMA_FAILURE_TOKENS = (
    "extraction_schema_invalid",
    "model_output_unusable",
)


METADATA_TOOL: dict[str, Any] = {
    "name": "submit_document_metadata",
    "description": "Submit top-level document metadata only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_name": {"type": "string"},
            "document_type": {"type": "string"},
            "product_category": {
                "type": ["string", "null"],
                "enum": ["WIRE_ROPE", "STRUCTURAL_STEEL", "PIPE", "OTHER", None]
            },
            "certificate_date": {"type": ["string", "null"]},
            "labeled_dates": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["label", "value"],
                },
            },
            "heat_number": {"type": ["string", "null"]},
            "batch_number": {"type": ["string", "null"]},
            "lot_number": {"type": ["string", "null"]},
            "colata_number": {"type": ["string", "null"]},
            "certificate_number": {"type": ["string", "null"]},
            "order_number": {"type": ["string", "null"]},
            "header_grade": {"type": ["string", "null"]},
            "product_description": {"type": ["string", "null"]},
            "dimensions": {"type": ["string", "null"]},
            "standards": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "weight_or_length": {"type": ["string", "null"]},
            "ai_analysis_remarks": {"type": ["string", "null"]},
            "confidence_score": {"type": "number"},
            "field_confidence": {
                "type": ["object", "null"],
                "properties": {
                    "heat_number": {"type": "number"},
                    "batch_number": {"type": "number"},
                    "lot_number": {"type": "number"},
                    "colata_number": {"type": "number"},
                    "certificate_number": {"type": "number"},
                    "order_number": {"type": "number"},
                },
            },
        },
        "required": ["supplier_name", "document_type", "confidence_score"],
    },
}

ITEM_TOOL: dict[str, Any] = {
    "name": "submit_line_items",
    "description": "Submit every extracted product/material item with strict nulls for unknown values.",
    "input_schema": {
        "type": "object",
        "properties": {
            "extraction_audit": {
                "type": ["string", "null"],
                "description": (
                    "REQUIRED. 1-2 sentences listing visible identity labels and every table header "
                    "before extracting items. Example: 'Identity: Batch No, Product, Diameter. "
                    "Chemistry columns C/Si/Mn. Mechanical rows Yield/Tensile/Elongation. "
                    "No repeating product-row table.'"
                ),
            },
            "total_items_detected": {"type": "integer"},
            "mechanical_table_rows": {
                "type": ["array", "null"],
                "description": (
                    "When mechanical properties are shown as a property-per-row table "
                    "(e.g. Proof Strength Rp0.2, Tensile Strength Rm, Elongation), "
                    "emit one object per visible property row with the property label "
                    "and every visible value column (Specified, Min, Max, Results, etc.). "
                    "Use null when no such table is present."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "property": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "chemical_table_rows": {
                "type": ["array", "null"],
                "description": (
                    "When chemical composition is shown as a matrix (element columns or "
                    "element-per-row), emit one object per visible chemistry row with the "
                    "element/property label and every visible value column. Use null when "
                    "no such table is present."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "property": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": ["string", "null"]},
                        "heat_number": {"type": ["string", "null"]},
                        "batch_number": {"type": ["string", "null"]},
                        "lot_number": {"type": ["string", "null"]},
                        "colata_number": {"type": ["string", "null"]},
                        "certificate_number": {"type": ["string", "null"]},
                        "order_number": {"type": ["string", "null"]},
                        "product_name": {"type": ["string", "null"]},
                        "grade": {"type": ["string", "null"]},
                        "weight_or_length": {"type": ["string", "null"]},
                        "dimensions": {"type": ["string", "null"]},
                        "standards": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "chemical_composition": {
                            "type": ["object", "null"],
                            "additionalProperties": {"type": ["number", "null"]},
                        },
                        "mechanical_properties": {
                            "type": ["object", "null"],
                            "properties": {
                                "yield_strength_mpa": {"type": ["number", "null"]},
                                "tensile_strength_mpa": {"type": ["number", "null"]},
                                "elongation_percentage": {"type": ["number", "null"]},
                            },
                        },
                        "row_confidence": {"type": ["number", "null"]},
                        "needs_review": {"type": ["boolean", "null"]},
                        "field_confidence": {
                            "type": ["object", "null"],
                            "properties": {
                                "heat_number": {"type": "number"},
                                "batch_number": {"type": "number"},
                                "lot_number": {"type": "number"},
                                "colata_number": {"type": "number"},
                                "item_id": {"type": "number"},
                                "certificate_number": {"type": "number"},
                                "order_number": {"type": "number"},
                                "yield_strength_mpa": {"type": "number"},
                                "tensile_strength_mpa": {"type": "number"},
                                "elongation_percentage": {"type": "number"}
                            }
                        },
                        "source_page": {"type": ["integer", "null"]},
                    },
                },
            },
        },
        "required": ["total_items_detected", "items"],
    },
}

METADATA_PROMPT = (
    "You are Stage A of an industrial document extraction pipeline. "
    "Use full-page document context to extract only document-level metadata. "
    "STRICT IDENTIFIER TRACEABILITY RULE: For heat_number, batch_number, certificate_number, and order_number, every character must be directly readable from the document image. "
    "The same strict readable-character rule applies to lot_number and colata_number. "
    "If any digit or character is unclear, blurry, cropped, partially visible, or ambiguous, return null for that identifier. "
    "Identifiers must not be inferred from nearby rows, surrounding context, repeated table patterns, expected formats, or other data. "
    "Never complete a partially visible identifier sequence. Do not use pattern completion for identifiers. "
    "For ambiguous critical identifiers, set needs_review=true where the schema allows it and use review reason critical_identifier_unverified. "
    "NUMERIC RULE: Extract numeric test values only when the complete number and decimal separator are clearly visible. "
    "For document_type, use a concise standard label such as 'Mill Test Certificate'. "
    "For product_category, classify the document content as 'WIRE_ROPE' (if it mentions steel wire rope, çelik halat, or wire rope constructions), 'STRUCTURAL_STEEL' (plates, beams, structural sections), 'PIPE' (tubes, pipes, hollow sections), or 'OTHER'. "
    "DATE RULE: Extract every visible date label/value pair into labeled_dates (e.g. label='Certificate Date', value='08/07/2024'). "
    "Do NOT use PO date, purchase order date, delivery date, analysis timestamp, upload timestamp, or issue metadata as certificate_date. "
    "Set certificate_date only from certificate/test/issue-style document dates when clearly labelled; otherwise leave certificate_date null and rely on labeled_dates. "
    "HEADER FIELD AUTONOMY: certificate_number, order_number/PO, heat, batch, lot, colata, product_description, header_grade, standards, and dimensions are independent even when they share one header line. "
    "Recognize labeled variants such as Cert, Certificate No, Certificato, PO, Order, Ordine, Batch, Heat, Lot, Lotto, Colata, N. Colata, Product, Grade, Diameter. "
    "Keep document identifiers independent: certificate_number, order_number, and heat/batch/lot/colata must not copy each other. "
    "product_description is the labeled product/material identity only. Do not put grade, standards, classification, diameter, certificate, or PO text into product_description. "
    "Never infer product_description from a standard or classification. Never infer header_grade from the supplier/manufacturer name. "
    "header_grade is only the labeled Grade/Quality/Kalite value. Do not copy a standard or classification into header_grade. "
    "Put diameter/size with its unit in dimensions. Put visible standards/classifications in standards as separate list values. "
    "If a labeled identity field is missing or ambiguous, leave it null. "
    "Confidence_score must reflect extraction certainty. Low-confidence fields MUST be null, not guessed. "
    "Provide field_confidence for heat_number, batch_number, lot_number, colata_number, certificate_number, and order_number when any candidate is visible. "
    "In ai_analysis_remarks, clearly explain when metadata is suppressed due to unreadable characters."
)

ITEM_PROMPT = (
    # ── PERSONA ─────────────────────────────────────────────────────────────
    "You are a forensic document auditor embedded in a high-stakes industrial "
    "compliance pipeline. Your sole function is to transcribe what is physically "
    "visible on the certificate image — nothing more. "
    "You are NOT an assistant whose job is to be helpful by completing gaps. "
    "You are an evidence recorder. A null value is never wrong. "
    "A fabricated value can invalidate a shipment, endanger lives, and constitute "
    "compliance fraud. When in doubt, null.\n\n"
    # ── RULE 0 · MANDATORY PRE-EXTRACTION COLUMN AUDIT (Chain of Thought) ───
    "RULE 0 — COLUMN AUDIT (fill extraction_audit FIRST, before any item):\n"
    "Scan identity labels and every table. Write 1-2 sentences in extraction_audit "
    "listing visible identity fields and every column label you can physically see. "
    "Example: 'Identity: Batch No, Product, Diameter. Chemistry: C/Si/Mn. "
    "Mechanical rows: Yield/Tensile/Elongation. No repeating product-row table.' "
    "A repeating product-row table is not required. One product/material with an "
    "identity block plus chemistry and/or mechanical matrices is one item, not zero. "
    "If you cannot identify a column or labeled field for a value, that field must "
    "be null. This step is not optional — it prevents you from inventing columns.\n\n"
    # ── RULE 1 · FIELD AUTONOMY ──────────────────────────────────────────────
    "RULE 1 — FIELD AUTONOMY (heat_number ≠ item_id):\n"
    "heat_number and item_id are independent fields. "
    "If no column is labelled Item ID, Article No, Pos., Position, or a clear "
    "item-reference label, set item_id to null for every row. "
    "Copying heat_number into item_id is strictly forbidden under any circumstance. "
    "When the document has only one traceability column, exactly one of the two "
    "fields receives that value; the other is null. "
    "On a single-product certificate, fully readable labeled identity fields "
    "(Heat No, Batch No, Lot, Lotto, Colata, Certificate No, Cert, Order/PO) may populate the "
    "single item even when they are not table columns.\n\n"
    # ── RULE 2 · STRICT NULL POLICY ──────────────────────────────────────────
    "RULE 2 — STRICT NULL POLICY:\n"
    "For every identifier field (heat_number, item_id, batch_number, certificate_number, order_number, lot_number, colata_number): if you cannot read EVERY character with "
    "absolute certainty from the pixel content, return null. "
    "Do not complete partial sequences. Do not infer from adjacent rows or "
    "expected format patterns. Do not use the document's header value to fill "
    "a blank cell in a multi-row product table unless that cell itself is "
    "visibly populated. "
    "Null is an honest answer. A wrong identifier is compliance fraud.\n\n"
    # ── RULE 3 · OCR CHARACTER VIGILANCE ─────────────────────────────────────
    "RULE 3 — OCR CHARACTER VIGILANCE:\n"
    "The following pairs are visually ambiguous on degraded, low-DPI, or "
    "stencil-printed scans: 5/S, 0/O/D, 1/I/L, 8/B, 6/G, 2/Z. "
    "If any digit in an identifier could belong to one of these pairs, you must: "
    "(a) set that identifier's field_confidence below 0.70, "
    "(b) set needs_review=true for the row, "
    "(c) return null for the identifier if field_confidence is below 0.80. "
    "Do NOT resolve the ambiguity by choosing the 'more likely' digit — "
    "that is a guess, not an observation.\n\n"
    # ── RULE 4 · HALLUCINATION RISK PROTOCOL ─────────────────────────────────
    "RULE 4 — HALLUCINATION RISK PROTOCOL:\n"
    "If any cell in a row is ambiguous, partially occluded, blurred, or could "
    "be a misread of another value, set needs_review=true and "
    "row_confidence <= 0.40 for that entire row. "
    "This flags the row for downstream suppression and human verification "
    "rather than allowing a guessed value to reach auto-acceptance.\n\n"
    # ── RULE 5 · NUMERIC & STRUCTURAL RULES ──────────────────────────────────
    "RULE 5 — NUMERIC AND STRUCTURAL RULES:\n"
    "Numeric test values (yield, tensile, elongation, chemistry) may only be extracted "
    "when the complete number and its decimal separator are clearly visible. "
    "If the certificate describes one product/material (identity block plus chemistry "
    "and/or mechanical matrices) rather than repeating product rows, emit exactly one "
    "item that aggregates product identity, traceability, chemistry, mechanicals, "
    "dimensions, and standards. total_items_detected must match emitted items. "
    "Do not treat chemistry-element rows or shielding-gas classification rows "
    "(for example M21 / C1) as separate products. "
    "If mechanical properties appear vertically as separate Yield / Tensile / "
    "Elongation lines for the same product, merge them into one item object. "
    "When a Mechanical Properties table lists one property per row with columns such "
    "as Specified, Min, Max, and Results, populate mechanical_table_rows with each "
    "visible property row and all visible column values. Put observed/test values "
    "only under Results/Result/Actual/Observed — never copy Specified/Min/Max into "
    "canonical mechanical_properties. Prefer Rp0.2 over Rp1.0 for yield when both exist. "
    "When a chemical composition matrix is visible, populate chemical_table_rows and "
    "put observed element values on the item in chemical_composition. Never copy "
    "Specified/Min/Max chemistry into chemical_composition. "
    "FIELD SEPARATION: product_name, grade, standards, and dimensions are independent. "
    "certificate_number, order_number, and heat/batch/lot/colata are also independent. "
    "product_name is the labeled Product/Material identity only. "
    "grade is the labeled Grade/Quality/Kalite value only. "
    "Never infer product_name from a standard or classification. Never infer grade from the supplier/manufacturer name. "
    "Put diameter/size with its unit in dimensions; keep mass/length in weight_or_length. "
    "Put visible classification/standard labels (EN ISO, AWS, ASTM, ISO, M21, C1) in standards as a list of separate values. "
    "Never put a standard or classification string into product_name or grade. "
    "Never synthesize a grade from a standard. If product identity is not explicitly labeled, product_name must be null. "
    "Keep certificate_number, order_number, and heat/batch/lot/colata independent even when they share a header block or one header line. "
    "Recognize labeled variants such as Cert, Certificate No, Certificato, PO, Order, Ordine, Batch, Heat, Lot, Lotto, Colata, N. Colata. "
    "If a labeled identity field is missing or ambiguous, leave it null. "
    "Set source_page for the item. "
    "If product category is WIRE_ROPE, extract Tensile Strength Class into "
    "tensile_strength_mpa; treat Construction or Core labels as grade. "
    "Do not emit rows for header lines, totals, footers, or implied/blank rows. "
    "Do not return items=[] when a single product/material is visibly described.\n\n"
    # ── RULE 6 · CONTRADICTION PROHIBITION ───────────────────────────────────
    "RULE 6 — CRITICAL: DO NOT CONTRADICT YOUR OWN AUDIT:\n"
    "If in extraction_audit you describe a heat_number (or any identifier) as "
    "'unclear', 'partially visible', 'blurred', 'hard to read', or 'cannot read', "
    "you MUST output null for that field in every affected row. "
    "Emitting a guessed string value while simultaneously stating it is unclear "
    "is a FATAL SYSTEM FAILURE — it will be detected as a hallucination, the row "
    "will be flagged needs_review=true, and the document will be blocked from "
    "auto-acceptance. There are no exceptions to this rule."
)


def _build_diagnostic_summary(
    *,
    pages: list[ProcessedPage],
    metadata: dict[str, Any],
    items: list[ExtractedItem],
    total_items_detected: int,
    confidence_score: float,
) -> dict[str, Any]:
    document_type = str(metadata.get("document_type", "Unknown document"))
    supplier_name = str(metadata.get("supplier_name", "Unknown supplier"))
    return {
        "document_understood": document_type != "Unknown document" or supplier_name != "Unknown supplier",
        "rows_extracted": len(items),
        "has_table_like_structure": any(page.table_crop_available for page in pages),
        "document_type": document_type,
        "total_items_detected": total_items_detected,
        "items_array_length": len(items),
        "confidence_score_from_model": confidence_score,
    }


def _image_block(image: EncodedVariant) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": image.media_type, "data": image.data},
    }


def _metadata_blocks(pages: list[ProcessedPage]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen_variant_keys: set[str] = set()

    for page in pages[: settings.max_pages_for_llm]:
        primary = page.full_page_variant_name if page.full_page_variant_name in page.variants else "contrast"
        if primary not in page.variants:
            primary = "full_gray"
        _append_unique_block(blocks, seen_variant_keys, page.variants[primary], f"p{page.page_number}:{primary}")

        # Phase 2 cost fix: only add full_gray for page 1 when it differs
        # from the already-added primary variant, to avoid sending the same
        # image twice.
        if page.page_number == 1 and primary != "full_gray" and "full_gray" in page.variants:
            _append_unique_block(blocks, seen_variant_keys, page.variants["full_gray"], f"p{page.page_number}:full_gray")

    blocks.append({"type": "text", "text": "Extract document-level metadata only."})
    return blocks


def _row_extraction_blocks(pages: list[ProcessedPage]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen_variant_keys: set[str] = set()

    for page in pages[: settings.max_pages_for_llm]:
        preferred = page.selected_variants if page.selected_variants else []
        for variant_name in preferred:
            if variant_name in page.variants:
                _append_unique_block(blocks, seen_variant_keys, page.variants[variant_name], f"p{page.page_number}:{variant_name}")

        if page.table_crop_available:
            # Table crop + adaptive binary are most informative for row extraction.
            for vname in ("table_crop", "adaptive_binary"):
                if vname in page.variants:
                    _append_unique_block(blocks, seen_variant_keys, page.variants[vname], f"p{page.page_number}:{vname}")
        else:
            # No table crop: use one clean full-page view only.
            full_page_name = page.full_page_variant_name if page.full_page_variant_name in page.variants else "contrast"
            if full_page_name in page.variants:
                _append_unique_block(blocks, seen_variant_keys, page.variants[full_page_name], f"p{page.page_number}:{full_page_name}")

    return blocks


def _append_unique_block(
    blocks: list[dict[str, Any]],
    seen: set[str],
    image: "EncodedVariant",
    key: str,
) -> None:
    """Append an image block only if we haven't already included this key."""
    if key not in seen:
        seen.add(key)
        blocks.append(_image_block(image))


def _items_blocks(pages: list[ProcessedPage]) -> list[dict[str, Any]]:
    blocks = _row_extraction_blocks(pages)
    blocks.append({
        "type": "text",
        "text": (
            "Step 1: Fill extraction_audit — list every column header you can physically see "
            "in the table (1-2 sentences). "
            "Step 2: Extract line items strictly from those visible columns. "
            "Any field without a corresponding visible column must be null."
        ),
    })
    return blocks


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token usage from a legacy response object safely."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _validate_stage_a_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from app.services.extraction_providers.output_schemas import StageAMetadata

    try:
        return StageAMetadata.model_validate(metadata).model_dump()
    except ValidationError as exc:
        raise ExtractionOutputError("Stage A metadata failed schema validation.") from exc


def _validate_stage_b_payload(item_payload: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from app.services.extraction_providers.output_schemas import StageBLineItems

    try:
        return StageBLineItems.model_validate(item_payload).model_dump()
    except ValidationError as exc:
        raise ExtractionOutputError("Stage B line items failed schema validation.") from exc


def _run_metadata_extraction(
    client: Any,
    pages: list[ProcessedPage],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Returns ``(metadata_dict, usage_dict)``. ``client`` is unused; kept for test patch compatibility."""
    del client
    from app.services.extraction_providers.factory import get_extraction_provider

    metadata, usage = get_extraction_provider().extract_metadata(pages)
    return _validate_stage_a_payload(metadata), usage


def _run_row_extraction(
    client: Any,
    pages: list[ProcessedPage],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Returns ``(item_payload_dict, usage_dict)``. ``client`` is unused; kept for test patch compatibility."""
    del client
    from app.services.extraction_providers.factory import get_extraction_provider

    item_payload, usage = get_extraction_provider().extract_line_items(pages, metadata)
    return _validate_stage_b_payload(item_payload), usage


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise ExtractionProviderError(f"LLM response missing tool output: {tool_name}")


def _value_from_canonical_or_alias(payload: dict[str, Any], canonical_field: str) -> Any:
    if canonical_field in payload:
        value = payload.get(canonical_field)
        if value not in (None, "", []):
            return value
    for key, value in payload.items():
        if key == canonical_field:
            continue
        if resolve_canonical_field(str(key)) == canonical_field and value not in (None, "", []):
            return value
    return payload.get(canonical_field)


def _identifier_aliases(field_name: str) -> tuple[str, ...]:
    aliases = {
        "heat_number": ("heat_number", "heat_no", "heat no", "heat"),
        "batch_number": (
            "batch_number",
            "batch_no",
            "batch n",
            "batch n°",
            "batch",
        ),
        "lot_number": ("lot_number", "lot_no", "lot", "lotto", "n lotto"),
        "colata_number": ("colata_number", "colata", "colata/batch", "n colata", "colata n°"),
        "cast_number": ("cast_number", "cast_no", "cast"),
        "charge_number": ("charge_number", "charge_no", "charge"),
        "coil_number": ("coil_number", "coil_no", "coil"),
        "item_id": ("item_id", "pipe_id", "pipe_coil_id"),
        "pipe_id": ("pipe_id", "item_id", "pipe_coil_id"),
        "certificate_number": (
            "certificate_number",
            "certificate_no",
            "cert_number",
            "cert_no",
            "certificate",
            "cert",
            "certificato",
        ),
        "order_number": (
            "order_number",
            "order_no",
            "purchase_order",
            "po_number",
            "po_no",
            "po",
            "order",
            "ordine",
        ),
    }
    return aliases.get(field_name, (field_name,))


def _raw_identifier_value(payload: dict[str, Any], field_name: str) -> Any:
    for key in _identifier_aliases(field_name):
        if key in payload:
            return payload.get(key)
    return _value_from_canonical_or_alias(payload, field_name)


def _profile_requires_identifier_gate(
    quality_class: str | None,
    quality_reasons: list[str] | None,
) -> bool:
    return (
        quality_class in DEGRADED_IDENTIFIER_QUALITY_CLASSES
        or "low_identifier_legibility" in set(quality_reasons or [])
    )


def _field_confidence(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[float | None, str | None]:
    field_confidence = payload.get("field_confidence")
    if isinstance(field_confidence, dict):
        raw_value = field_confidence.get(field_name)
        if isinstance(raw_value, (float, int)) and not isinstance(raw_value, bool):
            return max(0.0, min(float(raw_value), 1.0)), "field_confidence"

    direct_key = f"{field_name}_confidence"
    raw_value = payload.get(direct_key)
    if isinstance(raw_value, (float, int)) and not isinstance(raw_value, bool):
        return max(0.0, min(float(raw_value), 1.0)), direct_key

    row_confidence = payload.get("row_confidence")
    if isinstance(row_confidence, (float, int)) and not isinstance(row_confidence, bool):
        return max(0.0, min(float(row_confidence), 1.0)), "row_confidence"

    return None, None


def _identifier_confidence_map(
    payload: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for field_name in fields:
        confidence, _ = _field_confidence(payload, field_name)
        if confidence is not None:
            values[field_name] = confidence
    return values


def _identifier_issue_reason(
    payload: dict[str, Any],
    field_name: str,
    *,
    gate_required: bool,
) -> tuple[str | None, float | None, str | None]:
    confidence, confidence_source = _field_confidence(payload, field_name)
    item_reasons = {str(reason).lower() for reason in payload.get("review_reasons") or []}
    low_confidence_markers = {"low_identifier_confidence", f"weak_{field_name}_evidence"}
    ambiguity_markers = {
        "visual_ambiguity",
        "ambiguous_identifier",
        f"{field_name}_uncertain",
    }
    has_low_confidence_reason = bool(item_reasons & low_confidence_markers) or any(
        field_name in reason and any(marker in reason for marker in ("weak", "low"))
        for reason in item_reasons
    )
    has_ambiguity_reason = bool(item_reasons & ambiguity_markers) or any(
        field_name in reason
        and any(marker in reason for marker in ("uncertain", "ambiguous"))
        for reason in item_reasons
    )
    has_visual_ambiguity = bool(payload.get("visual_ambiguity")) or bool(
        payload.get(f"{field_name}_visual_ambiguity")
    )

    if has_visual_ambiguity or has_ambiguity_reason:
        return "visual_ambiguity", confidence, confidence_source
    if has_low_confidence_reason:
        return "low_identifier_confidence", confidence, confidence_source
    if confidence is not None and confidence < IDENTIFIER_CONFIDENCE_THRESHOLD:
        return "low_identifier_confidence", confidence, confidence_source
    if gate_required and confidence is None:
        return "missing_identifier_confidence", confidence, confidence_source
    return None, confidence, confidence_source


def _identifier_suppression_event(
    *,
    field_name: str,
    raw_candidate: Any,
    reason: str,
    confidence: float | None,
    confidence_source: str | None,
    quality_class: str | None,
    scope: str,
    row_index: int | None = None,
) -> dict[str, Any]:
    note = (
        f"Suppressed {field_name} candidate because {reason.replace('_', ' ')} "
        "prevents strict critical identifier acceptance."
    )
    event: dict[str, Any] = {
        "scope": scope,
        "field": field_name,
        "raw_candidate": raw_candidate,
        "accepted_value": None,
        "reason": reason,
        "confidence": confidence,
        "confidence_source": confidence_source,
        "quality_class": quality_class,
        "evidence_note": note,
    }
    if row_index is not None:
        event["row_index"] = row_index
    return event


def _apply_identifier_gate(
    payload: dict[str, Any],
    *,
    fields: tuple[str, ...],
    quality_class: str | None,
    quality_reasons: list[str] | None,
    scope: str,
    row_index: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gated = dict(payload)
    gate_required = _profile_requires_identifier_gate(quality_class, quality_reasons)
    events: list[dict[str, Any]] = []

    for field_name in fields:
        raw_candidate = _raw_identifier_value(payload, field_name)
        if raw_candidate is None or (isinstance(raw_candidate, str) and not raw_candidate.strip()):
            continue
        reason, confidence, confidence_source = _identifier_issue_reason(
            payload,
            field_name,
            gate_required=gate_required,
        )
        if reason is None:
            gated[field_name] = raw_candidate
            continue
        for key in _identifier_aliases(field_name):
            if key in gated or key == field_name:
                gated[key] = None
        events.append(
            _identifier_suppression_event(
                field_name=field_name,
                raw_candidate=raw_candidate,
                reason=reason,
                confidence=confidence,
                confidence_source=confidence_source,
                quality_class=quality_class,
                scope=scope,
                row_index=row_index,
            )
        )

    return gated, events


def _merge_identifier_guard_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None], dict[str, Any]] = {}
    for event in events:
        key = (str(event.get("scope") or "row"), event.get("row_index") if isinstance(event.get("row_index"), int) else None)
        bucket = grouped.setdefault(
            key,
            {
                "scope": key[0],
                "suppressed_fields": [],
                "suppressed_identifiers": [],
            },
        )
        if key[1] is not None:
            bucket["row_index"] = key[1]
        field_name = str(event.get("field"))
        if field_name not in bucket["suppressed_fields"]:
            bucket["suppressed_fields"].append(field_name)
        bucket["suppressed_identifiers"].append(event)
    return list(grouped.values())


def _mapping_diagnostics(metadata: dict[str, Any], items_raw: list[dict[str, Any]]) -> dict[str, Any]:
    observed_headers: list[str] = list(metadata.keys())
    for row in items_raw[:1]:
        observed_headers.extend(str(key) for key in row.keys())
        if isinstance(row.get("mechanical_properties"), dict):
            observed_headers.extend(str(key) for key in row["mechanical_properties"].keys())

    explanations = [explain_mapping(header) for header in observed_headers]
    resolved = [entry for entry in explanations if entry.get("matched_canonical_field")]
    unresolved = [entry for entry in explanations if not entry.get("matched_canonical_field")]
    return {
        "registry_version": "v1",
        "canonical_fields": sorted(CANONICAL_FIELDS.keys()),
        "resolved_headers_count": len(resolved),
        "unresolved_headers_count": len(unresolved),
        "resolved_headers": resolved,
        "unresolved_headers": unresolved,
        "registry_snapshot": get_registry_snapshot(),
    }


def _normalize_dimensions(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def _normalize_standards(raw: Any) -> list[str] | None:
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if not isinstance(raw, list):
        return None
    values = [str(item).strip() for item in raw if str(item).strip()]
    return values or None


def _normalize_source_page(raw: Any) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw >= 1 else None


def _normalize_chemical_composition(raw: Any) -> dict[str, float | None] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    from app.services.mechanical_table_mapper import canonicalize_element_symbol, parse_chemistry_value

    composition: dict[str, float | None] = {}
    for key, value in raw.items():
        symbol = canonicalize_element_symbol(str(key)) or str(key).strip()
        if not symbol:
            continue
        parsed = parse_chemistry_value(value)
        if parsed is None:
            continue
        composition[symbol] = parsed
    return composition or None


def _mechanical_confidence(field_conf: dict[str, Any], key: str, default: float = 1.0) -> float:
    """Return a numeric mechanical-field confidence; null is fail-closed."""

    raw = field_conf.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _normalize_row_dict(
    item_raw: dict[str, Any],
    row_index: int = 0,
    quality_class: str | None = None,
    quality_reasons: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flatten an LLM item payload into the canonical-key dict shape.

    Applies strict identifier acceptance before values enter the main row.
    """
    field_conf = item_raw.get("field_confidence") or {}
    identifier_payload, suppression_events = _apply_identifier_gate(
        item_raw,
        fields=CRITICAL_IDENTIFIER_FIELDS,
        quality_class=quality_class,
        quality_reasons=quality_reasons,
        scope="row",
        row_index=row_index,
    )

    # 2. Mechanical Properties
    mp_raw = item_raw.get("mechanical_properties")
    mp_payload: dict[str, Any] | None = None
    if isinstance(mp_raw, dict):
        ys = _value_from_canonical_or_alias(mp_raw, "yield_strength_mpa")
        if _mechanical_confidence(field_conf, "yield_strength_mpa") < 0.8:
            ys = None
        ts = _value_from_canonical_or_alias(mp_raw, "tensile_strength_mpa")
        if _mechanical_confidence(field_conf, "tensile_strength_mpa") < 0.8:
            ts = None
        el = _value_from_canonical_or_alias(mp_raw, "elongation_percentage")
        if _mechanical_confidence(field_conf, "elongation_percentage") < 0.8:
            el = None
        mp_payload = {
            "yield_strength_mpa": ys,
            "tensile_strength_mpa": ts,
            "elongation_percentage": el,
        }

    row_conf = item_raw.get("row_confidence")
    if not isinstance(row_conf, (float, int)):
        row_conf = None
    elif row_conf < 0 or row_conf > 1:
        row_conf = None

    normalized = {
        "item_id": identifier_payload.get("item_id"),
        "pipe_id": identifier_payload.get("pipe_id"),
        "heat_number": identifier_payload.get("heat_number"),
        "batch_number": identifier_payload.get("batch_number"),
        "lot_number": identifier_payload.get("lot_number"),
        "colata_number": identifier_payload.get("colata_number"),
        "cast_number": identifier_payload.get("cast_number"),
        "charge_number": identifier_payload.get("charge_number"),
        "coil_number": identifier_payload.get("coil_number"),
        "certificate_number": identifier_payload.get("certificate_number"),
        "order_number": identifier_payload.get("order_number"),
        "traceability_identifier_type": _value_from_canonical_or_alias(item_raw, "traceability_identifier_type"),
        "traceability_identifier_label": _value_from_canonical_or_alias(item_raw, "traceability_identifier_label"),
        "traceability_identifier_value": _value_from_canonical_or_alias(item_raw, "traceability_identifier_value"),
        "product_name": _normalize_dimensions(_value_from_canonical_or_alias(item_raw, "product_name")),
        "grade": _value_from_canonical_or_alias(item_raw, "grade"),
        "weight_or_length": _value_from_canonical_or_alias(item_raw, "weight_or_length"),
        "dimensions": _normalize_dimensions(_value_from_canonical_or_alias(item_raw, "dimensions")),
        "standards": _normalize_standards(_value_from_canonical_or_alias(item_raw, "standards")),
        "chemical_composition": _normalize_chemical_composition(item_raw.get("chemical_composition")),
        "mechanical_properties": mp_payload,
        "row_confidence": row_conf,
        "_identifier_confidence": _identifier_confidence_map(item_raw, CRITICAL_IDENTIFIER_FIELDS),
        "needs_review": bool(item_raw.get("needs_review"))
                        or bool(suppression_events),
        "source_page": _normalize_source_page(item_raw.get("source_page")),
    }
    return normalized, suppression_events


def _apply_numeric_parser(
    rows: list[dict[str, Any]],
    preprocessing_meta: dict[str, Any],
) -> list[str]:
    """Normalise numeric values in-place.

    Returns a list of structured review-reason tokens (``numeric_uncertain:<field>``
    and ``numeric_promoted_thousands:<field>``) to surface to the caller.
    """

    trace: list[dict[str, Any]] = []
    tokens: list[str] = []
    for index, row in enumerate(rows):
        mp_raw = row.get("mechanical_properties")
        if not isinstance(mp_raw, dict):
            continue
        values, row_trace = parse_mechanical_properties(mp_raw)
        row["mechanical_properties"] = values
        serialisable_trace = {key: parsed.to_dict() for key, parsed in row_trace.items()}
        trace.append({"row_index": index, "fields": serialisable_trace})
        for field_name, parsed in row_trace.items():
            if parsed.promoted_thousands:
                tokens.append(f"numeric_promoted_thousands:{field_name}")
            if parsed.uncertain and parsed.value is not None:
                tokens.append(f"numeric_uncertain:{field_name}")
    preprocessing_meta["numeric_parser_trace"] = trace
    return tokens


def _apply_header_propagation(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    ai_remarks: str | None,
    preprocessing_meta: dict[str, Any],
) -> list[str]:
    """Back-fill missing row grades from a strong header grade.

    Returns structured review tokens describing propagation conflicts.
    """

    result = propagate_to_rows(rows, metadata=metadata, ai_remarks=ai_remarks)
    preprocessing_meta["header_propagation"] = result.to_dict()
    apply_updates(rows, result)
    return list(result.conflicts)


def _build_explanation_payload(
    *,
    extraction: UniversalDocumentExtraction,
    profile: DocumentProfile | None,
    route_decision: RouteDecision | None,
    review_policy: dict[str, Any],
    preprocessing_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "document_profile": profile.to_dict() if profile else None,
        "route_decision": route_decision.to_dict() if route_decision else None,
        "validation_outcome": {
            "document_outcome": extraction.outcome,
            "review_required": extraction.needs_review,
            "review_reasons": list(extraction.review_reasons),
        },
        "traceability": {
            "status": extraction.traceability_status,
            "confidence": extraction.traceability_confidence,
            "accepted_identifier_values": dict(extraction.accepted_identifier_values),
            "raw_identifier_candidates": dict(extraction.raw_identifier_candidates),
        },
        "review_policy": review_policy,
        "evidence_propagation_notes": preprocessing_meta.get("document_context", {}).get(
            "propagation_notes", []
        ),
        "unresolved_or_ambiguous": [
            reason
            for reason in extraction.review_reasons
            if "unresolved" in reason or "ambiguous" in reason or "unsupported" in reason
        ],
        "confidence_breakdown": extraction.confidence_breakdown or {},
        "identifier_guard": preprocessing_meta.get("identifier_guard"),
        "evidence": preprocessing_meta.get("evidence", []),
    }


def _row_dict_to_item(row: dict[str, Any]) -> ExtractedItem:
    mp_raw = row.get("mechanical_properties")
    mechanical = None
    if isinstance(mp_raw, dict):
        mechanical = MechanicalProperties(
            yield_strength_mpa=mp_raw.get("yield_strength_mpa"),
            tensile_strength_mpa=mp_raw.get("tensile_strength_mpa"),
            elongation_percentage=mp_raw.get("elongation_percentage"),
        )
    return ExtractedItem(
        item_id=row.get("item_id"),
        pipe_id=row.get("pipe_id"),
        heat_number=row.get("heat_number"),
        batch_number=row.get("batch_number"),
        lot_number=row.get("lot_number"),
        colata_number=row.get("colata_number"),
        cast_number=row.get("cast_number"),
        charge_number=row.get("charge_number"),
        coil_number=row.get("coil_number"),
        certificate_number=row.get("certificate_number"),
        order_number=row.get("order_number"),
        order_date=row.get("order_date"),
        traceability_identifier_type=row.get("traceability_identifier_type"),
        traceability_identifier_label=row.get("traceability_identifier_label"),
        traceability_identifier_value=row.get("traceability_identifier_value"),
        product_name=row.get("product_name"),
        product_details=row.get("product_details"),
        grade=row.get("grade"),
        weight_or_length=row.get("weight_or_length"),
        dimensions=row.get("dimensions"),
        standards=row.get("standards") if isinstance(row.get("standards"), list) else None,
        classifications=row.get("classifications") if isinstance(row.get("classifications"), list) else None,
        chemical_composition=row.get("chemical_composition") if isinstance(row.get("chemical_composition"), dict) else None,
        mechanical_properties=mechanical,
        row_confidence=row.get("row_confidence"),
        needs_review=row.get("needs_review", False),
        grade_provenance=row.get("grade_provenance") or ("row" if row.get("grade") else None),
        grade_field_label=row.get("grade_field_label"),
        source_page=row.get("source_page"),
    )


def run_multi_stage_extraction(
    pages: list[ProcessedPage],
    preprocessing_meta: dict[str, Any],
    profile: DocumentProfile | None = None,
    route_decision: RouteDecision | None = None,
) -> UniversalDocumentExtraction:
    schema_failure = False
    from app.services.evidence import evidence_index, pack_evidence, select_metadata_evidence, select_row_evidence

    if pages and isinstance(getattr(pages[0], "variants", None), dict):
        metadata_evidence = pack_evidence(
            select_metadata_evidence(pages),
            max_request_bytes=settings.bedrock_max_request_bytes,
            prompt_overhead_bytes=80_000,
        )
        row_evidence = pack_evidence(
            select_row_evidence(pages),
            max_request_bytes=settings.bedrock_max_request_bytes,
            prompt_overhead_bytes=80_000,
        )
        preprocessing_meta["evidence"] = evidence_index(row_evidence or metadata_evidence)
    else:
        preprocessing_meta.setdefault("evidence", [])
    try:
        metadata, usage_a = _run_metadata_extraction(None, pages)
        item_payload, usage_b = _run_row_extraction(None, pages, metadata)
    except HTTPException:
        raise
    except ExtractionOutputError:
        logger.exception("LLM structured output failed; routing to review.")
        schema_failure = True
        metadata = {
            "supplier_name": "Unknown supplier",
            "document_type": "Unknown document",
            "confidence_score": 0.0,
        }
        item_payload = {"total_items_detected": 0, "items": []}
        usage_a = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        usage_b = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
        preprocessing_meta["schema_failure"] = True
    except ExtractionProviderError:
        logger.exception("LLM extraction failed.")
        raise HTTPException(status_code=502, detail="Vision extraction failed.") from None
    except Exception:
        logger.exception("LLM extraction failed.")
        raise HTTPException(status_code=502, detail="Vision extraction failed.") from None

    q_class = profile.quality_class if profile else None
    q_reasons = profile.reasons if profile else []
    metadata, metadata_suppression_events = _apply_identifier_gate(
        metadata,
        fields=(
            "heat_number",
            "batch_number",
            "lot_number",
            "colata_number",
            "cast_number",
            "charge_number",
            "coil_number",
            "certificate_number",
            "order_number",
        ),
        quality_class=q_class,
        quality_reasons=q_reasons,
        scope="metadata",
    )

    # Store token usage for budget tracking and thesis reporting.
    combined_usage = {
        "provider": settings.extraction_provider,
        "model_id": settings.bedrock_model_id if settings.extraction_provider == "bedrock" else "mock",
        "region": settings.bedrock_region,
        "stage_a_input_tokens": int(usage_a.get("input_tokens", 0) or 0),
        "stage_a_output_tokens": int(usage_a.get("output_tokens", 0) or 0),
        "stage_b_input_tokens": int(usage_b.get("input_tokens", 0) or 0),
        "stage_b_output_tokens": int(usage_b.get("output_tokens", 0) or 0),
        "total_input_tokens": int(usage_a.get("input_tokens", 0) or 0) + int(usage_b.get("input_tokens", 0) or 0),
        "total_output_tokens": int(usage_a.get("output_tokens", 0) or 0) + int(usage_b.get("output_tokens", 0) or 0),
        "latency_ms": int(usage_a.get("latency_ms", 0) or 0) + int(usage_b.get("latency_ms", 0) or 0),
        "pages_sent": len(pages),
        "schema_failure": schema_failure,
        "estimated_cost_usd": 0.0,
        "cost_is_estimate": True,
    }
    total_in = combined_usage["total_input_tokens"]
    total_out = combined_usage["total_output_tokens"]
    if settings.extraction_provider == "bedrock":
        combined_usage["estimated_cost_usd"] = round(
            total_in / 1_000_000 * settings.bedrock_input_cost_per_million
            + total_out / 1_000_000 * settings.bedrock_output_cost_per_million,
            6,
        )
    preprocessing_meta["llm_usage"] = combined_usage
    logger.info("LLM usage: %s", combined_usage)

    items_raw = item_payload.get("items", [])

    # Capture the CoT column-audit whisper from Stage B for debug/thesis tracing.
    stage_b_audit = item_payload.get("extraction_audit")
    if stage_b_audit:
        preprocessing_meta["stage_b_extraction_audit"] = stage_b_audit

    suppression_events = list(metadata_suppression_events)
    items_dicts = []

    for idx, row in enumerate(items_raw):
        if not isinstance(row, dict):
            continue
        norm_row, suppressed = _normalize_row_dict(
            row,
            idx,
            quality_class=q_class,
            quality_reasons=q_reasons,
        )
        items_dicts.append(norm_row)
        if suppressed:
            suppression_events.extend(suppressed)

    if suppression_events:
        preprocessing_meta["identifier_guard"] = {
            "policy": "strict_critical_identifier_acceptance_gate",
            "quality_class": q_class,
            "low_identifier_legibility": "low_identifier_legibility" in set(q_reasons),
            "confidence_threshold": IDENTIFIER_CONFIDENCE_THRESHOLD,
            "events": _merge_identifier_guard_events(suppression_events),
        }
    row_shape_tokens: list[str] = []
    row_shape_traces: list[dict[str, Any]] = []
    vertical_result = collapse_vertical_mechanical_rows(items_dicts, metadata=metadata)
    if vertical_result.tokens:
        items_dicts = vertical_result.rows
        row_shape_tokens.extend(vertical_result.tokens)
        row_shape_traces.append(vertical_result.trace)
    classification_result = collapse_alternative_classification_rows(items_dicts, metadata=metadata)
    if classification_result.tokens:
        items_dicts = classification_result.rows
        row_shape_tokens.extend(classification_result.tokens)
        row_shape_traces.append(classification_result.trace)
    backfill_result = backfill_single_item_context(items_dicts, metadata=metadata)
    if backfill_result.tokens:
        items_dicts = backfill_result.rows
        row_shape_tokens.extend(backfill_result.tokens)
        row_shape_traces.append(backfill_result.trace)
    if row_shape_tokens or row_shape_traces:
        preprocessing_meta["row_shape_normalization"] = {
            "tokens": row_shape_tokens,
            "traces": row_shape_traces,
        }
    mechanical_table_tokens: list[str] = []
    pre_mechanical_count = len(items_dicts)
    items_dicts, mechanical_table_tokens, mechanical_table_trace = apply_mechanical_table_mapping(
        items_dicts,
        item_payload=item_payload if isinstance(item_payload, dict) else {},
        items_raw=[row for row in items_raw if isinstance(row, dict)],
    )
    if pre_mechanical_count == 0 and len(items_dicts) == 1:
        backfill_after_mechanical = backfill_single_item_context(items_dicts, metadata=metadata)
        if backfill_after_mechanical.tokens:
            items_dicts = backfill_after_mechanical.rows
            row_shape_tokens.extend(backfill_after_mechanical.tokens)
            row_shape_traces.append(backfill_after_mechanical.trace)
    identity_result = apply_identity_field_mapping(items_dicts, metadata=metadata)
    items_dicts = identity_result.rows
    if identity_result.tokens or identity_result.traces:
        row_shape_tokens.extend(identity_result.tokens)
        row_shape_traces.extend(identity_result.traces)
        preprocessing_meta["identity_field_mapping"] = {
            "tokens": identity_result.tokens,
            "traces": identity_result.traces,
        }
    if row_shape_tokens or row_shape_traces:
        preprocessing_meta["row_shape_normalization"] = {
            "tokens": row_shape_tokens,
            "traces": row_shape_traces,
        }
    if mechanical_table_tokens or mechanical_table_trace:
        preprocessing_meta["mechanical_table_mapping"] = {
            "tokens": mechanical_table_tokens,
            "trace": mechanical_table_trace,
        }
    semantic_rows = normalize_rows_semantics(items_dicts)
    preprocessing_meta["semantic_normalization"] = [entry.to_dict() for entry in semantic_rows]
    items_dicts = [entry.canonical_values for entry in semantic_rows]
    numeric_tokens = _apply_numeric_parser(items_dicts, preprocessing_meta)
    propagation_tokens = _apply_header_propagation(
        items_dicts,
        metadata=metadata,
        ai_remarks=metadata.get("ai_analysis_remarks") if isinstance(metadata, dict) else None,
        preprocessing_meta=preprocessing_meta,
    )
    context = build_document_context(
        items_dicts,
        metadata=metadata if isinstance(metadata, dict) else None,
        pages_meta=preprocessing_meta.get("pages") if isinstance(preprocessing_meta.get("pages"), list) else None,
    )
    context_tokens = propagate_context_to_rows(items_dicts, context)
    preprocessing_meta["document_context"] = context.to_dict()
    finalization = finalize_extraction_fields(metadata, items_dicts)
    metadata = finalization.metadata
    items_dicts = finalization.rows
    preprocessing_meta["extraction_finalization"] = finalization.to_dict()
    reconciliation = apply_canonical_reconciliation(metadata, items_dicts)
    metadata = reconciliation.metadata
    items_dicts = reconciliation.rows
    preprocessing_meta["canonical_reconciliation"] = reconciliation.to_dict()
    preprocessing_meta["identifier_confidence"] = {
        "metadata": _identifier_confidence_map(
            metadata,
            (
                "heat_number",
                "batch_number",
                "lot_number",
                "colata_number",
                "cast_number",
                "charge_number",
                "coil_number",
                "certificate_number",
                "order_number",
            ),
        ),
        "rows": [
            row.get("_identifier_confidence", {})
            if isinstance(row.get("_identifier_confidence"), dict)
            else {}
            for row in items_dicts
        ],
    }
    items = [_row_dict_to_item(row) for row in items_dicts]
    preprocessing_meta["field_mapping_diagnostics"] = _mapping_diagnostics(
        metadata=metadata,
        items_raw=[row for row in items_raw if isinstance(row, dict)],
    )
    raw_model_confidence = float(metadata.get("confidence_score", 0.0))
    raw_model_confidence = max(0.0, min(raw_model_confidence, 1.0))
    raw_reported_total_items = int(item_payload.get("total_items_detected", len(items)))
    if row_shape_tokens:
        raw_reported_total_items = len(items)
    diagnostic_summary = _build_diagnostic_summary(
        pages=pages,
        metadata=metadata,
        items=items,
        total_items_detected=raw_reported_total_items,
        confidence_score=raw_model_confidence,
    )
    preprocessing_meta["diagnostic_summary"] = diagnostic_summary
    logger.info("Extraction diagnostic summary: %s", json.dumps(diagnostic_summary, ensure_ascii=False))
    extraction = UniversalDocumentExtraction(
        supplier_name=str(metadata.get("supplier_name", "Unknown supplier")),
        document_type=str(metadata.get("document_type", "Unknown document")),
        product_category=metadata.get("product_category"),
        certificate_date=metadata.get("certificate_date"),
        batch_number=metadata.get("batch_number"),
        lot_number=_raw_identifier_value(metadata, "lot_number"),
        colata_number=_raw_identifier_value(metadata, "colata_number"),
        cast_number=_raw_identifier_value(metadata, "cast_number"),
        charge_number=_raw_identifier_value(metadata, "charge_number"),
        coil_number=_raw_identifier_value(metadata, "coil_number"),
        certificate_number=metadata.get("certificate_number"),
        order_number=metadata.get("order_number"),
        order_date=metadata.get("order_date"),
        traceability_identifier_type=metadata.get("traceability_identifier_type"),
        traceability_identifier_label=metadata.get("traceability_identifier_label"),
        traceability_identifier_value=metadata.get("traceability_identifier_value"),
        total_items_detected=raw_reported_total_items,
        items=items,
        confidence_score=raw_model_confidence,
        raw_model_confidence=raw_model_confidence,
        ai_analysis_remarks=metadata.get("ai_analysis_remarks"),
        stage_b_extraction_audit=stage_b_audit,
    )

    if schema_failure:
        extraction.review_reasons.extend(SCHEMA_FAILURE_TOKENS)
        extraction.needs_review = True
        extraction.confidence_score = 0.0
        extraction.raw_model_confidence = 0.0

    if suppression_events:
        extraction.review_reasons.append("critical_identifier_unverified")
        extraction.needs_review = True

    if reconciliation.review_tokens:
        extraction.review_reasons.extend(reconciliation.review_tokens)
        extraction.needs_review = True

    # Seed review reasons from numeric parser + header propagation before
    # validation so the validator sees a coherent, pre-normalised state.
    if numeric_tokens or propagation_tokens or context_tokens or mechanical_table_tokens:
        extraction.review_reasons = sorted(
            set(
                [
                    *extraction.review_reasons,
                    *numeric_tokens,
                    *propagation_tokens,
                    *context_tokens,
                    *mechanical_table_tokens,
                ]
            )
        )
        extraction.needs_review = extraction.needs_review or bool(
            numeric_tokens or propagation_tokens or context_tokens or mechanical_table_tokens
        )

    extraction = validate_document(extraction)
    extraction = validate_traceability(extraction, preprocessing_meta)
    identifier_verification = apply_identifier_verification_guard(extraction)
    if identifier_verification.tokens or identifier_verification.assessments:
        preprocessing_meta["identifier_verification_guard"] = identifier_verification.to_dict()

    remarks = str(extraction.ai_analysis_remarks or "").lower()
    if any(token in remarks for token in ("blur", "noise", "unreadable", "unclear")):
        extraction.review_reasons.append("document readability concerns noted in analysis remarks")
        extraction.needs_review = True

    hard_row_confidence_floor = min(0.5, settings.review_confidence_threshold - 0.2)
    if any(item.row_confidence is not None and item.row_confidence < hard_row_confidence_floor for item in extraction.items):
        extraction.review_reasons.append("low-confidence rows detected")
    if any(item.needs_review for item in extraction.items):
        extraction.review_reasons.append("visual ambiguity detected in row")
        extraction.needs_review = True
    if len(extraction.items) == 0:
        extraction.review_reasons.append("extraction structure is incomplete")

    canonical_finalization = apply_canonical_finalization_to_extraction(extraction)
    preprocessing_meta.setdefault("extraction_finalization", {})
    preprocessing_meta["extraction_finalization"].update(canonical_finalization.to_dict())

    confidence_assessment = normalize_confidence(
        extraction=extraction,
        preprocessing_meta=preprocessing_meta,
        raw_reported_total_items=raw_reported_total_items,
        review_confidence_threshold=settings.review_confidence_threshold,
    )
    extraction.confidence_score = confidence_assessment.final_confidence
    extraction.review_reasons = confidence_assessment.review_reasons
    extraction.needs_review = confidence_assessment.status == "NEEDS_REVIEW"
    extraction.status = confidence_assessment.status
    extraction.confidence_breakdown = confidence_assessment.metrics.get("confidence_breakdown")
    preprocessing_meta["confidence_assessment"] = confidence_assessment.metrics
    logger.info("Confidence assessment: %s", json.dumps(confidence_assessment.metrics, ensure_ascii=False))

    review_decision = apply_review_policy(
        extraction,
        profile=profile,
        preprocessing_meta=preprocessing_meta,
        review_confidence_threshold=settings.review_confidence_threshold,
    )
    preprocessing_meta["review_policy"] = review_decision.to_dict()
    logger.info(
        "Review policy: needs_review=%s structured_reasons=%s",
        review_decision.review_required,
        review_decision.structured_reasons,
    )

    decision_tokens = finalize_decision(extraction)
    if decision_tokens:
        preprocessing_meta["extraction_finalization"]["decision_tokens"] = decision_tokens

    high_conf_floor = max(0.90, settings.review_confidence_threshold)
    policy_auto_cleared = (
        not review_decision.review_required
        and extraction.confidence_score >= high_conf_floor
        and not review_decision.structured_reasons
    )
    extraction.outcome = aggregate_document_outcome(
        [item.validation.outcome for item in extraction.items if item.validation]
    )
    if extraction.outcome == NEEDS_REVIEW and not policy_auto_cleared:
        extraction.needs_review = True
    if extraction.needs_review:
        extraction.status = "NEEDS_REVIEW"
    else:
        extraction.status = "COMPLETED"
    sanitize_unverified_traceability_for_user(extraction)
    post_sanitize_confidence = normalize_confidence(
        extraction=extraction,
        preprocessing_meta=preprocessing_meta,
        raw_reported_total_items=raw_reported_total_items,
        review_confidence_threshold=settings.review_confidence_threshold,
    )
    extraction.confidence_breakdown = post_sanitize_confidence.metrics.get("confidence_breakdown")
    extraction.review_reasons = sorted(
        set([*extraction.review_reasons, *post_sanitize_confidence.review_reasons])
    )
    if (
        (extraction.needs_review or post_sanitize_confidence.status == "NEEDS_REVIEW")
        and not policy_auto_cleared
    ):
        extraction.needs_review = True
        extraction.status = "NEEDS_REVIEW"
    elif policy_auto_cleared:
        extraction.needs_review = False
        extraction.status = "COMPLETED"
    post_decision_tokens = finalize_decision(extraction)
    if post_decision_tokens:
        preprocessing_meta["extraction_finalization"]["post_sanitize_decision_tokens"] = post_decision_tokens
    preprocessing_meta["confidence_assessment"] = post_sanitize_confidence.metrics
    extraction.explanation = _build_explanation_payload(
        extraction=extraction,
        profile=profile,
        route_decision=route_decision,
        review_policy=review_decision.to_dict(),
        preprocessing_meta=preprocessing_meta,
    )
    apply_reconcile_to_extraction(extraction)
    apply_review_invariants(
        extraction,
        canonical_review_tokens=review_tokens_from_meta(preprocessing_meta),
        raw_reported_total_items=raw_reported_total_items,
    )
    return extraction
