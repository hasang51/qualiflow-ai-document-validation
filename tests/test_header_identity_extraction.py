from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.domain.labeled_identifier_extractor import extract_labeled_identifiers_from_text
from app.services.document_profiler import DocumentProfile
from app.services.extraction_pipeline import run_multi_stage_extraction
from app.services.identity_field_mapper import apply_identity_field_mapping, promote_unique_header_fields
from app.services.row_shape_normalizer import backfill_single_item_context


def _profile() -> DocumentProfile:
    return DocumentProfile(
        document_id="syn-header-identity",
        filename="synthetic-header-identity.pdf",
        page_count=1,
        has_text_layer=True,
        text_density=0.8,
        blur_score=420.0,
        noise_score=2.0,
        table_presence_hint=True,
        quality_class="digital_clean",
        reasons=["text_layer_present"],
    )


def _page() -> MagicMock:
    page = MagicMock()
    page.page_number = 1
    page.table_crop_available = True
    return page


def _run_pipeline(stage_b_payload: dict, stage_a_payload: dict):
    with (
        patch("app.services.extraction_pipeline._run_metadata_extraction") as mock_meta,
        patch("app.services.extraction_pipeline._run_row_extraction") as mock_row,
    ):
        mock_meta.return_value = (stage_a_payload, {"input_tokens": 10, "output_tokens": 5})
        mock_row.return_value = (stage_b_payload, {"input_tokens": 20, "output_tokens": 10})
        return run_multi_stage_extraction(
            [_page()],
            {"pages": [{"table_detection": {"table_found": True}}]},
            profile=_profile(),
        )


def test_family_combined_header_line_keeps_cert_po_batch_independent():
    line = "Certificate No: SYN-IC-1001 PO: PO-77001 Batch: SYN41001"
    matches = {field: value for field, _label, value in extract_labeled_identifiers_from_text(line)}
    assert matches["certificate_number"] == "SYN-IC-1001"
    assert matches["order_number"] == "PO-77001"
    assert matches["batch_number"] == "SYN41001"

    from app.domain.labeled_identifier_extractor import apply_labeled_identifiers

    metadata = {"ai_analysis_remarks": line}
    rows = [{}]
    apply_labeled_identifiers(metadata, rows)
    mapped = apply_identity_field_mapping(rows, metadata=metadata)
    item = mapped.rows[0]
    assert item["certificate_number"] == "SYN-IC-1001"
    assert item["order_number"] == "PO-77001"
    assert item["batch_number"] == "SYN41001"
    assert item.get("product_name") is None
    assert item["certificate_number"] != item["order_number"] != item["batch_number"]


def test_family_product_name_separate_from_grade():
    result = apply_identity_field_mapping(
        [{"product_name": "WELDWIRE SG2 / BRONZEWIRE SG2", "grade": "SG2"}],
        metadata={},
    )
    item = result.rows[0]
    assert item["product_name"] == "WELDWIRE SG2 / BRONZEWIRE SG2"
    assert item["grade"] == "SG2"


def test_family_standards_mixed_with_product_text():
    result = apply_identity_field_mapping(
        [{"product_name": "EN ISO 14341-A", "grade": "M21 / C1"}],
        metadata={"product_description": "WELDWIRE SG2", "header_grade": "SG2", "standards": ["EN ISO 14341-A"]},
    )
    item = result.rows[0]
    assert item["product_name"] == "WELDWIRE SG2"
    assert item["grade"] == "SG2"
    assert "EN ISO 14341-A" in (item["standards"] or [])
    assert "M21" in (item["classifications"] or [])
    assert "C1" in (item["classifications"] or [])


def test_family_multilingual_header_aliases():
    result = apply_identity_field_mapping(
        [
            {
                "Certificato N": "CERT-2044",
                "Ordine": "PO-88",
                "Colata": "410537",
                "Prodotto": "Filo SG2",
                "Qualità": "SG2",
                "Diametro": "0.80 mm",
                "Norma": "EN ISO 14341-A",
            }
        ],
        metadata={},
    )
    item = result.rows[0]
    assert item["certificate_number"] == "CERT-2044"
    assert item["order_number"] == "PO-88"
    assert item["colata_number"] == "410537"
    assert item.get("batch_number") is None
    assert item["product_name"] == "Filo SG2"
    assert item["grade"] == "SG2"
    assert item["dimensions"] == "0.80 mm"
    assert item["standards"] == ["EN ISO 14341-A"]


def test_family_repeated_row_certificates_stay_independent():
    extraction = _run_pipeline(
        {
            "extraction_audit": "Repeating product-row table with two heats.",
            "total_items_detected": 2,
            "items": [
                {
                    "item_id": "1",
                    "heat_number": "H-100",
                    "grade": "S355J2",
                    "dimensions": "12 mm",
                    "mechanical_properties": {
                        "yield_strength_mpa": 360.0,
                        "tensile_strength_mpa": 510.0,
                        "elongation_percentage": 24.0,
                    },
                    "row_confidence": 0.9,
                    "field_confidence": {"heat_number": 0.95, "item_id": 0.95},
                },
                {
                    "item_id": "2",
                    "heat_number": "H-200",
                    "grade": "S355J2",
                    "dimensions": "16 mm",
                    "mechanical_properties": {
                        "yield_strength_mpa": 355.0,
                        "tensile_strength_mpa": 520.0,
                        "elongation_percentage": 23.0,
                    },
                    "row_confidence": 0.9,
                    "field_confidence": {"heat_number": 0.95, "item_id": 0.95},
                },
            ],
        },
        stage_a_payload={
            "supplier_name": "Structural Mill",
            "document_type": "Mill Test Certificate",
            "product_category": "STRUCTURAL_STEEL",
            "certificate_number": "MTC-55",
            "order_number": "PO-900",
            "header_grade": "S355J2",
            "confidence_score": 0.9,
            "field_confidence": {"certificate_number": 0.95, "order_number": 0.95},
        },
    )
    assert len(extraction.items) == 2
    assert extraction.items[0].heat_number == "H-100"
    assert extraction.items[1].heat_number == "H-200"
    assert extraction.items[0].dimensions == "12 mm"
    assert extraction.items[1].dimensions == "16 mm"
    assert extraction.certificate_number == "MTC-55"
    assert extraction.order_number == "PO-900"
    assert extraction.items[0].product_name is None


def test_family_single_product_chemistry_mechanical_preserves_header_fields():
    extraction = _run_pipeline(
        {
            "extraction_audit": "Identity block plus chemistry and mechanical matrices.",
            "total_items_detected": 1,
            "items": [
                {
                    "batch_number": "SYN41001",
                    "certificate_number": "SYN-IC-1001",
                    "order_number": "PO-77001",
                    "product_name": "WELDWIRE SG2 / BRONZEWIRE SG2",
                    "grade": "SG2",
                    "dimensions": "0.80 mm",
                    "standards": ["EN ISO 14341-A", "M21", "C1"],
                    "chemical_composition": {"C": 0.08},
                    "mechanical_properties": {
                        "yield_strength_mpa": 470.0,
                        "tensile_strength_mpa": 560.0,
                        "elongation_percentage": 26.0,
                    },
                    "row_confidence": 0.92,
                    "field_confidence": {
                        "batch_number": 0.95,
                        "certificate_number": 0.95,
                        "order_number": 0.94,
                    },
                    "source_page": 1,
                }
            ],
        },
        stage_a_payload={
            "supplier_name": "Synthetic Metals GmbH",
            "document_type": "Inspection Certificate",
            "product_category": "OTHER",
            "batch_number": "SYN41001",
            "certificate_number": "SYN-IC-1001",
            "order_number": "PO-77001",
            "product_description": "WELDWIRE SG2 / BRONZEWIRE SG2",
            "header_grade": "SG2",
            "dimensions": "0.80 mm",
            "standards": ["EN ISO 14341-A", "M21", "C1"],
            "confidence_score": 0.9,
            "field_confidence": {
                "batch_number": 0.96,
                "certificate_number": 0.95,
                "order_number": 0.94,
            },
        },
    )
    item = extraction.items[0]
    assert item.certificate_number == "SYN-IC-1001"
    assert item.order_number == "PO-77001"
    assert item.batch_number == "SYN41001"
    assert item.product_name == "WELDWIRE SG2 / BRONZEWIRE SG2"
    assert item.grade == "SG2"
    assert item.dimensions == "0.80 mm"
    assert item.standards == ["EN ISO 14341-A"]
    assert item.classifications == ["M21", "C1"]
    assert item.chemical_composition is not None
    assert item.chemical_composition.get("C") == 0.08


def test_family_ambiguous_or_missing_identity_fields_fail_safe():
    extraction = _run_pipeline(
        {
            "extraction_audit": "Product and certificate labels are missing or mixed.",
            "total_items_detected": 1,
            "items": [
                {
                    "grade": "M21",
                    "standards": ["C1"],
                    "mechanical_properties": {
                        "yield_strength_mpa": 470.0,
                        "tensile_strength_mpa": 560.0,
                        "elongation_percentage": 26.0,
                    },
                    "row_confidence": 0.4,
                    "needs_review": True,
                }
            ],
        },
        stage_a_payload={
            "supplier_name": "Unknown Mill",
            "document_type": "Inspection Certificate",
            "product_category": "OTHER",
            "product_description": "EN ISO 14341-A",
            "header_grade": "Unknown Mill",
            "confidence_score": 0.4,
        },
    )
    item = extraction.items[0]
    assert item.product_name is None
    assert item.grade not in {"M21", "C1", "Unknown Mill"}
    assert item.certificate_number is None
    assert item.order_number is None
    assert extraction.needs_review is True
    assert extraction.status == "NEEDS_REVIEW"
    assert extraction.auto_accept_evidence is None


def test_colata_does_not_backfill_into_batch_number():
    result = backfill_single_item_context(
        [{"mechanical_properties": {"yield_strength_mpa": 470.0}}],
        metadata={"colata_number": "410537", "field_confidence": {"colata_number": 0.96}},
    )
    assert result.rows[0]["colata_number"] == "410537"
    assert not result.rows[0].get("batch_number")


def test_explicit_row_identifiers_outrank_metadata_and_aliases():
    result = apply_identity_field_mapping(
        [
            {
                "certificate_number": "ROW-CERT",
                "Cert No": "ALIAS-CERT",
                "order_number": "ROW-PO",
            }
        ],
        metadata={"certificate_number": "META-CERT", "order_number": "META-PO"},
    )
    item = result.rows[0]
    assert item["certificate_number"] == "ROW-CERT"
    assert item["order_number"] == "ROW-PO"


def test_metadata_fills_missing_row_identifiers_without_copying_across_fields():
    result = apply_identity_field_mapping(
        [{"grade": "SG2"}],
        metadata={
            "certificate_number": "SYN-IC-1001",
            "order_number": "PO-77001",
            "batch_number": "SYN41001",
            "lot_number": "LOT-9",
            "colata_number": "410537",
            "supplier_name": "Synthetic Metals GmbH",
            "header_grade": "Synthetic Metals GmbH",
        },
    )
    item = result.rows[0]
    assert item["certificate_number"] == "SYN-IC-1001"
    assert item["order_number"] == "PO-77001"
    assert item["batch_number"] == "SYN41001"
    assert item["lot_number"] == "LOT-9"
    assert item["colata_number"] == "410537"
    assert item["grade"] == "SG2"
    assert item.get("product_name") is None


def test_promote_unique_header_fields_is_consensus_only():
    metadata: dict = {}
    promote_unique_header_fields(
        metadata,
        [
            {"certificate_number": "CERT-A", "order_number": "PO-1"},
            {"certificate_number": "CERT-A", "order_number": "PO-2"},
        ],
    )
    assert metadata["certificate_number"] == "CERT-A"
    assert "order_number" not in metadata


def test_does_not_infer_product_from_standards_or_grade_from_supplier():
    result = apply_identity_field_mapping(
        [{}],
        metadata={
            "supplier_name": "ACME Steel",
            "header_grade": "ACME Steel",
            "product_description": "EN ISO 14341-A",
            "standards": ["EN ISO 14341-A", "M21"],
        },
    )
    item = result.rows[0]
    assert item["product_name"] is None
    assert item["grade"] is None
    assert item["standards"] == ["EN ISO 14341-A"]
    assert item["classifications"] == ["M21"]
