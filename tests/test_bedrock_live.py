from __future__ import annotations

import os

import pytest

from app.services.extraction_providers.bedrock_provider import BedrockGemmaProvider
from config.settings import reset_settings_cache

pytestmark = pytest.mark.bedrock_live

_LIVE = os.getenv("QUALIFLOW_BEDROCK_LIVE") == "1"


@pytest.mark.skipif(not _LIVE, reason="Set QUALIFLOW_BEDROCK_LIVE=1 to run live Bedrock smoke tests")
def test_live_bedrock_json_schema_metadata_smoke(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    monkeypatch.setenv("BEDROCK_REGION", "eu-central-1")
    reset_settings_cache()
    metadata, usage = BedrockGemmaProvider().extract_metadata([])
    assert "supplier_name" in metadata
    assert "document_type" in metadata
    assert usage["input_tokens"] >= 0
    assert usage["output_tokens"] >= 0
