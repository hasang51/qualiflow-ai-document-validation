from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.document_profiler import DocumentProfile
from app.services.extraction_pipeline import SCHEMA_FAILURE_TOKENS, run_multi_stage_extraction
from app.services.extraction_providers.base import ExtractionOutputError, ExtractionProviderError


def _page() -> MagicMock:
    page = MagicMock()
    page.page_number = 1
    page.table_crop_available = False
    return page


def _profile() -> DocumentProfile:
    return DocumentProfile(
        document_id="schema-fail",
        filename="schema-fail.pdf",
        page_count=1,
        has_text_layer=True,
        text_density=0.4,
        blur_score=300.0,
        noise_score=5.0,
        table_presence_hint=False,
        quality_class="digital_clean",
        reasons=["text_layer_present"],
    )


@patch("app.services.extraction_pipeline._run_row_extraction")
@patch("app.services.extraction_pipeline._run_metadata_extraction")
def test_schema_output_error_routes_to_review(mock_meta, mock_row):
    mock_meta.side_effect = ExtractionOutputError("Stage A metadata failed schema validation.")
    extraction = run_multi_stage_extraction([_page()], {}, profile=_profile())
    mock_row.assert_not_called()
    assert extraction.needs_review is True
    assert extraction.status == "NEEDS_REVIEW"
    assert extraction.auto_accept_evidence is None
    assert extraction.confidence_score < 0.75
    for token in SCHEMA_FAILURE_TOKENS:
        assert token in extraction.review_reasons


@patch("app.services.extraction_pipeline._run_row_extraction")
@patch("app.services.extraction_pipeline._run_metadata_extraction")
def test_provider_transport_error_still_returns_502(mock_meta, mock_row):
    mock_meta.side_effect = ExtractionProviderError("Bedrock Mantle request failed.")
    with pytest.raises(HTTPException) as exc_info:
        run_multi_stage_extraction([_page()], {}, profile=_profile())
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Vision extraction failed."
    mock_row.assert_not_called()
