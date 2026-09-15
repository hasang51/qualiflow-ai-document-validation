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
