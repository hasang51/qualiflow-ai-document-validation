from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.services.extraction_providers.base import ExtractionProviderError
from app.services.extraction_providers.bedrock_provider import BedrockGemmaProvider
from app.services.extraction_providers.factory import get_extraction_provider, reset_extraction_provider_cache
from app.services.extraction_providers.mock_provider import MockExtractionProvider
from config.settings import get_settings, reset_settings_cache


def test_factory_rejects_anthropic(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "anthropic")
    reset_settings_cache()
    reset_extraction_provider_cache()
    with pytest.raises(ValidationError):
        get_settings()


def test_factory_returns_mock_when_configured(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "mock")
    reset_settings_cache()
    reset_extraction_provider_cache()
    provider = get_extraction_provider()
    assert provider.name == "mock"
    metadata, usage = provider.extract_metadata([])
    assert metadata["supplier_name"] == "Mock Supplier"
    assert usage["input_tokens"] == 0


def test_mock_provider_can_fail_without_remote_call():
    provider = MockExtractionProvider(fail_with="boom")
    with pytest.raises(ExtractionProviderError, match="boom"):
        provider.extract_metadata([])
    with pytest.raises(ExtractionProviderError, match="boom"):
        provider.extract_line_items([], {})


def test_mock_provider_returns_items_payload():
    provider = MockExtractionProvider()
    payload, usage = provider.extract_line_items([], {})
    assert payload["total_items_detected"] == 0
    assert usage["output_tokens"] == 0


def test_factory_returns_bedrock_when_configured(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    reset_settings_cache()
    reset_extraction_provider_cache()
    provider = get_extraction_provider()
    assert provider.name == "bedrock"
    assert isinstance(provider, BedrockGemmaProvider)


def test_bedrock_provider_requires_model_id(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "")
    reset_settings_cache()
    reset_extraction_provider_cache()
    provider = BedrockGemmaProvider()
    with pytest.raises(ExtractionProviderError, match="BEDROCK_MODEL_ID"):
        provider.extract_metadata([])


def test_bedrock_boundary_is_not_wired_even_when_model_set(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    reset_settings_cache()
    provider = BedrockGemmaProvider()
    with pytest.raises(ExtractionProviderError, match="boundary only"):
        provider.extract_line_items([], {})
