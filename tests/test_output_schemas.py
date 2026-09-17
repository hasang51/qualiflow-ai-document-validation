from __future__ import annotations

import pytest

from app.services.extraction_pipeline import _validate_stage_a_payload, _validate_stage_b_payload
from app.services.extraction_providers.base import ExtractionOutputError
from app.services.extraction_providers.mock_provider import MockExtractionProvider


def test_mock_default_payloads_pass_stage_schemas():
    provider = MockExtractionProvider()
    metadata, _usage = provider.extract_metadata([])
    items, _usage_b = provider.extract_line_items([], metadata)
    assert _validate_stage_a_payload(metadata)["supplier_name"] == "Mock Supplier"
    assert _validate_stage_b_payload(items)["total_items_detected"] == 0


def test_stage_a_validation_rejects_missing_required_field():
    with pytest.raises(ExtractionOutputError, match="Stage A"):
        _validate_stage_a_payload({"document_type": "Mill Test Certificate", "confidence_score": 0.4})


def test_stage_a_validation_rejects_conformity_keys():
    with pytest.raises(ExtractionOutputError, match="Stage A"):
        _validate_stage_a_payload(
            {
                "supplier_name": "Acme",
                "document_type": "Mill Test Certificate",
                "confidence_score": 0.9,
                "is_compliant": True,
            }
        )


def test_stage_b_validation_rejects_auto_accept_keys():
    with pytest.raises(ExtractionOutputError, match="Stage B"):
        _validate_stage_b_payload(
            {
                "total_items_detected": 0,
                "items": [],
                "auto_accept": True,
            }
        )


def test_stage_b_accepts_single_product_aggregated_item():
    payload = _validate_stage_b_payload(
        {
            "extraction_audit": "Identity block plus chemistry and mechanical matrices.",
            "total_items_detected": 1,
            "chemical_table_rows": [{"property": "C", "Results": 0.08, "Specified": "0.06-0.14"}],
            "mechanical_table_rows": [{"property": "Tensile Strength Rm", "Results": 560, "Specified": "500-640"}],
            "items": [
                {
                    "item_id": None,
                    "heat_number": None,
                    "batch_number": "SYN41001",
                    "certificate_number": "SYN-IC-1001",
                    "order_number": "PO-77001",
                    "grade": "WELDWIRE SG2",
                    "product_name": "WELDWIRE SG2",
                    "weight_or_length": "1.080 Kg",
                    "dimensions": "0.80 mm",
                    "standards": ["EN ISO 14341-A", "M21"],
                    "chemical_composition": {"C": 0.08, "Si": 0.85, "Mn": 1.45},
                    "mechanical_properties": {
                        "yield_strength_mpa": 470.0,
                        "tensile_strength_mpa": 560.0,
                        "elongation_percentage": 26.0,
                    },
                    "row_confidence": 0.9,
                    "needs_review": False,
                    "source_page": 1,
                }
            ],
        }
    )
    assert payload["total_items_detected"] == 1
    item = payload["items"][0]
    assert item["batch_number"] == "SYN41001"
    assert item["dimensions"] == "0.80 mm"
    assert item["chemical_composition"]["C"] == 0.08
    assert item["mechanical_properties"]["tensile_strength_mpa"] == 560.0
    assert item["standards"] == ["EN ISO 14341-A", "M21"]


def test_stage_a_accepts_lot_and_colata_independently():
    payload = _validate_stage_a_payload(
        {
            "supplier_name": "Acme",
            "document_type": "Mill Test Certificate",
            "confidence_score": 0.9,
            "lot_number": "LOT-24-01",
            "colata_number": "410537",
            "certificate_number": "SYN-IC-1001",
            "order_number": "PO-77001",
        }
    )
    assert payload["lot_number"] == "LOT-24-01"
    assert payload["colata_number"] == "410537"
    assert payload["certificate_number"] == "SYN-IC-1001"
    assert payload["order_number"] == "PO-77001"


def test_stage_b_rejects_compliance_keys_on_aggregated_item():
    with pytest.raises(ExtractionOutputError, match="Stage B"):
        _validate_stage_b_payload(
            {
                "total_items_detected": 1,
                "items": [
                    {
                        "grade": "SG2",
                        "is_compliant": True,
                    }
                ],
            }
        )
