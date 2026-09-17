from __future__ import annotations

from app.domain.spec_registry import (
    MaterialSpec,
    clear_classification_specs,
    register_classification_spec,
    resolve_spec_from_standard_classification,
)
from app.schemas.extraction import ExtractedItem, MechanicalProperties, UniversalDocumentExtraction
from app.services.canonical_reconciler import apply_canonical_reconciliation
from app.services.validator import validate_document


def test_po_number_and_date_on_one_label_line():
    result = apply_canonical_reconciliation(
        {"ai_analysis_remarks": "PO: ABC123 / 25/06/2024"},
        [{}],
    )
    item = result.rows[0]
    assert item["order_number"] == "ABC123"
    assert item["order_date"] == "25/06/2024"
    assert item["order_number"] != item["order_date"]
    assert result.metadata["order_number"] == "ABC123"
    assert result.metadata["order_date"] == "25/06/2024"


def test_certificate_number_and_date_combined():
    result = apply_canonical_reconciliation(
        {"ai_analysis_remarks": "Certificate No: SYN-IC-1001 / 19.06.2020"},
        [{}],
    )
    item = result.rows[0]
    assert item["certificate_number"] == "SYN-IC-1001"
    assert item["certificate_date"] == "19.06.2020"


def test_product_identity_stays_distinct_from_product_details():
    result = apply_canonical_reconciliation(
        {},
        [{"product_name": "WELDWIRE SG2 - DIAM 0.80 mm", "grade": "SG2"}],
    )
    item = result.rows[0]
    assert item["product_name"] == "WELDWIRE SG2"
    assert item["product_details"]
    assert "DIAM" in item["product_details"].upper() or "0.80" in item["product_details"]
    assert item["grade"] == "SG2"
    assert item["product_name"] != item["product_details"]


def test_classification_without_grade_is_not_invented_as_grade():
    result = apply_canonical_reconciliation(
        {},
        [{"grade": "ER70S-6", "standards": ["AWS A5.18"]}],
    )
    item = result.rows[0]
    assert item["grade"] is None
    assert "ER70S-6" in (item["classifications"] or [])
    assert "AWS A5.18" in (item["standards"] or [])
    assert item["product_name"] is None


def test_standard_and_classification_pair_stays_split():
    result = apply_canonical_reconciliation(
        {},
        [{"standards": ["EN ISO 14341-A", "M21", "C1"], "product_name": "WELDWIRE SG2", "grade": "SG2"}],
    )
    item = result.rows[0]
    assert item["product_name"] == "WELDWIRE SG2"
    assert item["grade"] == "SG2"
    assert item["standards"] == ["EN ISO 14341-A"]
    assert "M21" in (item["classifications"] or [])
    assert "C1" in (item["classifications"] or [])


def test_multilingual_labels_resolve_independently():
    result = apply_canonical_reconciliation(
        {},
        [
            {
                "Certificato N": "CERT-2044",
                "Ordine": "PO-88",
                "Colata": "410537",
                "Prodotto": "Filo SG2",
                "Qualità": "SG2",
                "Diametro": "0.80 mm",
                "Norma": "EN ISO 14341-A",
                "Classificazione": "ER70S-6",
            }
        ],
    )
    item = result.rows[0]
    assert item["certificate_number"] == "CERT-2044"
    assert item["order_number"] == "PO-88"
    assert item["colata_number"] == "410537"
    assert item.get("batch_number") in (None, "")
    assert item["product_name"] == "Filo SG2"
    assert item["grade"] == "SG2"
    assert item["dimensions"] == "0.80 mm"
    assert item["standards"] == ["EN ISO 14341-A"]
    assert "ER70S-6" in (item["classifications"] or [])


def test_conflicting_labeled_candidates_go_null_and_review():
    result = apply_canonical_reconciliation(
        {"ai_analysis_remarks": "PO: PO-111 PO Number: PO-222"},
        [{}],
    )
    assert result.metadata.get("order_number") in (None, "")
    assert result.rows[0].get("order_number") in (None, "")
    assert any(token.startswith("conflicting_labeled_candidates:order_number") for token in result.review_tokens)


def test_repeated_rows_stay_independent():
    result = apply_canonical_reconciliation(
        {"header_grade": "S355J2"},
        [
            {"heat_number": "H-100", "grade": "S355J2", "dimensions": "12 mm"},
            {"heat_number": "H-200", "grade": "S355J2", "dimensions": "16 mm"},
        ],
    )
    assert len(result.rows) == 2
    assert result.rows[0]["heat_number"] == "H-100"
    assert result.rows[1]["heat_number"] == "H-200"
    assert result.rows[0]["dimensions"] == "12 mm"
    assert result.rows[1]["dimensions"] == "16 mm"


def test_single_product_chemistry_mechanical_matrix():
    result = apply_canonical_reconciliation(
        {},
        [
            {
                "product_name": "WELDWIRE SG2 / BRONZEWIRE SG2",
                "grade": "SG2",
                "dimensions": "0.80 mm",
                "standards": ["EN ISO 14341-A", "M21", "C1"],
                "chemical_composition": {"C": 0.08},
                "mechanical_properties": {"yield_strength_mpa": 470.0},
            }
        ],
    )
    item = result.rows[0]
    assert item["product_name"] == "WELDWIRE SG2 / BRONZEWIRE SG2"
    assert item["grade"] == "SG2"
    assert item["standards"] == ["EN ISO 14341-A"]
    assert item["classifications"] == ["M21", "C1"]
    assert item["chemical_composition"]["C"] == 0.08


def test_unmapped_classification_pair_does_not_invent_grade():
    item = ExtractedItem(
        classifications=["ER70S-6"],
        standards=["AWS A5.18"],
        mechanical_properties=MechanicalProperties(
            yield_strength_mpa=470.0,
            tensile_strength_mpa=560.0,
            elongation_percentage=26.0,
        ),
    )
    extraction = UniversalDocumentExtraction(
        supplier_name="Mill",
        document_type="Inspection Certificate",
        total_items_detected=1,
        items=[item],
        confidence_score=0.9,
    )
    validated = validate_document(extraction)
    assert validated.items[0].grade is None
    assert validated.items[0].classifications == ["ER70S-6"]
    assert "missing_critical_field:grade" in validated.review_reasons
    assert validated.items[0].validation is not None
    assert validated.items[0].validation.is_compliant is None


def test_mapped_standard_classification_pair_validates_without_inventing_grade():
    spec = MaterialSpec("PAIR-SPEC", 400.0, 500.0, 700.0, 20.0)
    register_classification_spec("AWS A5.18", "ER70S-6", spec)
    try:
        resolution = resolve_spec_from_standard_classification(["AWS A5.18"], ["ER70S-6"])
        assert resolution.status == "resolved"
        item = ExtractedItem(
            classifications=["ER70S-6"],
            standards=["AWS A5.18"],
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=470.0,
                tensile_strength_mpa=560.0,
                elongation_percentage=26.0,
            ),
        )
        extraction = UniversalDocumentExtraction(
            supplier_name="Mill",
            document_type="Inspection Certificate",
            total_items_detected=1,
            items=[item],
            confidence_score=0.9,
        )
        validated = validate_document(extraction)
        assert validated.items[0].grade is None
        assert validated.items[0].classifications == ["ER70S-6"]
        assert validated.items[0].validation is not None
        assert validated.items[0].validation.is_compliant is True
        assert "missing_critical_field:grade" not in validated.review_reasons
    finally:
        clear_classification_specs()
