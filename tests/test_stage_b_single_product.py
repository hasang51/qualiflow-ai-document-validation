from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.services.document_profiler import DocumentProfile
from app.services.extraction_pipeline import run_multi_stage_extraction
from app.services.extraction_providers.output_schemas import StageBLineItems


SYNTHETIC_BATCH = "SYN41001"
SYNTHETIC_CERT = "SYN-IC-1001"
SYNTHETIC_PO = "PO-77001"
SYNTHETIC_PRODUCT = "WELDWIRE SG2 / BRONZEWIRE SG2"
SYNTHETIC_GRADE = "SG2"


def _profile() -> DocumentProfile:
    return DocumentProfile(
        document_id="syn-single-product",
        filename="synthetic-inspection-certificate.pdf",
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


def build_synthetic_inspection_certificate_pdf(path: Path) -> None:
    """Draw a one-product identity + chemistry + mechanical certificate layout."""

    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "INSPECTION CERTIFICATE",
        "Supplier: Synthetic Metals GmbH",
        f"Certificate No: {SYNTHETIC_CERT}",
        f"Purchase Order: {SYNTHETIC_PO}",
        f"Batch / Heat: {SYNTHETIC_BATCH}",
        "Date: 15/09/2024",
        f"Product: {SYNTHETIC_PRODUCT}",
        f"Grade: {SYNTHETIC_GRADE}",
        "Diameter: 0.80 mm",
        "Standard: EN ISO 14341-A",
        "Classification: M21 / C1",
        "",
        "Chemical composition",
        "Element   Specified     Results",
        "C         0.06-0.14     0.08",
        "Si        0.50-1.00     0.85",
        "Mn        1.00-1.80     1.45",
        "P         <=0.025       0.015",
        "S         <=0.025       0.012",
        "Cu        <=0.30        0.15",
        "",
        "Mechanical properties",
        "Property              Specified     Results",
        "Yield strength Re     >=420         470",
        "Tensile strength Rm   500-640       560",
        "Elongation A5         >=22          26",
        "Classification        M21 / C1",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 28
    image.save(path, "PDF", resolution=150.0)


def _stage_a_payload() -> dict:
    return {
        "supplier_name": "Synthetic Metals GmbH",
        "document_type": "Inspection Certificate",
        "product_category": "OTHER",
        "certificate_date": "15/09/2024",
        "batch_number": SYNTHETIC_BATCH,
        "certificate_number": SYNTHETIC_CERT,
        "order_number": SYNTHETIC_PO,
        "product_description": SYNTHETIC_PRODUCT,
        "header_grade": SYNTHETIC_GRADE,
        "dimensions": "0.80 mm",
        "standards": ["EN ISO 14341-A", "M21", "C1"],
        "weight_or_length": "1.080 Kg",
        "confidence_score": 0.9,
        "field_confidence": {
            "batch_number": 0.96,
            "certificate_number": 0.95,
            "order_number": 0.94,
        },
    }


def _empty_items_with_tables_payload() -> dict:
    return {
        "extraction_audit": (
            "Identity block with batch, product, and diameter. "
            "Chemistry matrix C/Si/Mn/P/S/Cu. Mechanical property-per-row table. "
            "No repeating product-row table."
        ),
        "total_items_detected": 0,
        "items": [],
        "chemical_table_rows": [
            {"property": "C", "Specified": "0.06-0.14", "Results": 0.08, "source_page": 1},
            {"property": "Si", "Specified": "0.50-1.00", "Results": 0.85},
            {"property": "Mn", "Results": 1.45},
            {"property": "P", "Results": 0.015},
            {"property": "S", "Results": 0.012},
            {"property": "Cu", "Results": 0.15},
        ],
        "mechanical_table_rows": [
            {"property": "Yield strength Re", "Specified": ">=420", "Results": 470},
            {"property": "Tensile Strength Rm", "Specified": "500-640", "Results": 560},
            {"property": "Elongation A5", "Specified": ">=22", "Results": 26},
        ],
    }


def _aggregated_item_payload() -> dict:
    return {
        "extraction_audit": "Single product identity plus chemistry and mechanical matrices.",
        "total_items_detected": 1,
        "chemical_table_rows": [{"property": "C", "Results": 0.08}],
        "mechanical_table_rows": [{"property": "Tensile Strength Rm", "Results": 560}],
        "items": [
            {
                "item_id": None,
                "heat_number": None,
                "batch_number": SYNTHETIC_BATCH,
                "certificate_number": SYNTHETIC_CERT,
                "order_number": SYNTHETIC_PO,
                "product_name": SYNTHETIC_PRODUCT,
                "grade": SYNTHETIC_GRADE,
                "weight_or_length": "1.080 Kg",
                "dimensions": "0.80 mm",
                "standards": ["EN ISO 14341-A", "M21", "C1"],
                "chemical_composition": {"C": 0.08, "Si": 0.85, "Mn": 1.45},
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.92,
                "needs_review": False,
                "field_confidence": {
                    "batch_number": 0.95,
                    "certificate_number": 0.95,
                    "order_number": 0.94,
                    "yield_strength_mpa": 0.95,
                    "tensile_strength_mpa": 0.95,
                    "elongation_percentage": 0.95,
                },
                "source_page": 1,
            }
        ],
    }


def _run_pipeline(stage_b_payload: dict, stage_a_payload: dict | None = None):
    with (
        patch("app.services.extraction_pipeline._run_metadata_extraction") as mock_meta,
        patch("app.services.extraction_pipeline._run_row_extraction") as mock_row,
    ):
        mock_meta.return_value = (stage_a_payload or _stage_a_payload(), {"input_tokens": 10, "output_tokens": 5})
        mock_row.return_value = (stage_b_payload, {"input_tokens": 20, "output_tokens": 10})
        return run_multi_stage_extraction(
            [_page()],
            {"pages": [{"table_detection": {"table_found": True}}]},
            profile=_profile(),
        )


def test_synthetic_pdf_matches_single_product_layout(tmp_path: Path):
    pdf_path = tmp_path / "synthetic-inspection-certificate.pdf"
    build_synthetic_inspection_certificate_pdf(pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
    text = pdf_path.read_bytes()
    assert text.startswith(b"%PDF")


def test_empty_items_with_tables_synthesize_one_product():
    extraction = _run_pipeline(_empty_items_with_tables_payload())
    assert len(extraction.items) >= 1
    item = extraction.items[0]
    assert item.batch_number == SYNTHETIC_BATCH or item.traceability_identifier_value == SYNTHETIC_BATCH
    assert item.chemical_composition is not None
    assert item.chemical_composition.get("C") == 0.08
    assert item.chemical_composition.get("Si") == 0.85
    assert item.mechanical_properties is not None
    assert item.mechanical_properties.yield_strength_mpa == 470.0
    assert item.mechanical_properties.tensile_strength_mpa == 560.0
    assert item.mechanical_properties.elongation_percentage == 26.0
    assert item.product_name == SYNTHETIC_PRODUCT
    assert item.grade == SYNTHETIC_GRADE
    assert item.dimensions == "0.80 mm"
    assert item.certificate_number == SYNTHETIC_CERT
    assert item.order_number == SYNTHETIC_PO
    if extraction.status == "AUTO_ACCEPT":
        assert extraction.auto_accept_evidence is not None
    else:
        assert extraction.auto_accept_evidence is None


def test_aggregated_item_preserves_chemistry_mechanics_and_traceability():
    extraction = _run_pipeline(_aggregated_item_payload())
    assert len(extraction.items) >= 1
    item = extraction.items[0]
    assert item.batch_number == SYNTHETIC_BATCH
    assert item.certificate_number == SYNTHETIC_CERT
    assert item.order_number == SYNTHETIC_PO
    assert item.product_name == SYNTHETIC_PRODUCT
    assert item.grade == SYNTHETIC_GRADE
    assert item.dimensions == "0.80 mm"
    assert item.standards == ["EN ISO 14341-A"]
    assert item.classifications == ["M21", "C1"]
    assert item.chemical_composition is not None
    assert item.chemical_composition.get("C") == 0.08
    assert item.mechanical_properties is not None
    assert item.mechanical_properties.tensile_strength_mpa == 560.0
    assert item.source_page == 1
    if extraction.status == "AUTO_ACCEPT":
        assert extraction.auto_accept_evidence is not None
    else:
        assert extraction.auto_accept_evidence is None


def test_empty_items_and_empty_tables_still_need_review():
    extraction = _run_pipeline({"total_items_detected": 0, "items": []})
    assert extraction.items == []
    assert extraction.needs_review is True
    assert extraction.status == "NEEDS_REVIEW"
    assert extraction.auto_accept_evidence is None
    assert "no_items_extracted" in extraction.review_reasons


def test_stage_b_schema_accepts_synthetic_layout_payload():
    parsed = StageBLineItems.model_validate(_aggregated_item_payload())
    assert len(parsed.items) == 1
    assert parsed.items[0].chemical_composition is not None
    assert parsed.items[0].dimensions == "0.80 mm"
    assert parsed.items[0].product_name == SYNTHETIC_PRODUCT
    assert parsed.items[0].grade == SYNTHETIC_GRADE


def test_repeated_product_rows_remain_separate():
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
            "confidence_score": 0.9,
        },
    )
    assert len(extraction.items) == 2
    assert extraction.items[0].heat_number == "H-100"
    assert extraction.items[1].heat_number == "H-200"
    assert extraction.items[0].dimensions == "12 mm"
    assert extraction.items[1].dimensions == "16 mm"
    assert extraction.items[0].grade == "S355J2"
    assert extraction.items[0].product_name is None


def test_ambiguous_product_identity_stays_null_and_needs_review():
    extraction = _run_pipeline(
        {
            "extraction_audit": "Only classification labels are visible; product name is not labeled.",
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
            "confidence_score": 0.4,
        },
    )
    item = extraction.items[0]
    assert item.product_name is None
    assert item.grade not in {"M21", "C1"}
    assert extraction.needs_review is True
    assert extraction.status == "NEEDS_REVIEW"
    assert extraction.auto_accept_evidence is None

