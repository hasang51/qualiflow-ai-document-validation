from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.services.extraction_providers.base import ExtractionOutputError, ExtractionProviderError
from app.services.extraction_providers.bedrock_provider import BedrockGemmaProvider
from app.services.extraction_providers.output_schemas import StageAMetadata, StageBLineItems
from config.settings import reset_settings_cache


def _completion(content: dict, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def test_stage_a_schema_rejects_compliance_keys():
    with pytest.raises(ValidationError):
        StageAMetadata.model_validate(
            {
                "supplier_name": "Acme",
                "document_type": "Mill Test Certificate",
                "confidence_score": 0.9,
                "is_compliant": True,
            }
        )


def test_stage_b_schema_rejects_auto_accept_keys():
    with pytest.raises(ValidationError):
        StageBLineItems.model_validate(
            {
                "total_items_detected": 0,
                "items": [],
                "auto_accept": True,
            }
        )


def test_bedrock_provider_requires_model_id(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "")
    reset_settings_cache()
    provider = BedrockGemmaProvider()
    with pytest.raises(ExtractionProviderError, match="BEDROCK_MODEL_ID"):
        provider.extract_metadata([])


def test_bedrock_metadata_happy_path(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    monkeypatch.setenv("BEDROCK_REGION", "eu-central-1")
    reset_settings_cache()
    captured: dict = {}

    def fake_chat(*, region: str, payload: dict) -> dict:
        captured["region"] = region
        captured["payload"] = payload
        return _completion(
            {
                "supplier_name": "Acme Steel",
                "document_type": "Mill Test Certificate",
                "confidence_score": 0.8,
            }
        )

    monkeypatch.setattr(
        "app.services.extraction_providers.bedrock_provider.chat_completions",
        fake_chat,
    )
    provider = BedrockGemmaProvider()
    metadata, usage = provider.extract_metadata([])
    assert metadata["supplier_name"] == "Acme Steel"
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert usage["latency_ms"] >= 0
    assert captured["region"] == "eu-central-1"
    assert captured["payload"]["model"] == "google.gemma-4-26b-a4b"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["response_format"]["json_schema"]["name"] == "submit_document_metadata"
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True


def test_bedrock_line_items_happy_path(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    reset_settings_cache()

    def fake_chat(*, region: str, payload: dict) -> dict:
        del region
        assert payload["response_format"]["json_schema"]["strict"] is False
        return _completion({"total_items_detected": 0, "items": []})

    monkeypatch.setattr(
        "app.services.extraction_providers.bedrock_provider.chat_completions",
        fake_chat,
    )
    provider = BedrockGemmaProvider()
    payload, usage = provider.extract_line_items([], {"document_type": "Mill Test Certificate"})
    assert payload["total_items_detected"] == 0
    assert payload["items"] == []
    assert usage["input_tokens"] == 11


def test_bedrock_invalid_json_raises_output_error(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    reset_settings_cache()

    def fake_chat(*, region: str, payload: dict) -> dict:
        del region, payload
        return {"choices": [{"message": {"content": "not-json"}}], "usage": {}}

    monkeypatch.setattr(
        "app.services.extraction_providers.bedrock_provider.chat_completions",
        fake_chat,
    )
    provider = BedrockGemmaProvider()
    with pytest.raises(ExtractionOutputError, match="non-JSON"):
        provider.extract_metadata([])


def test_bedrock_schema_mismatch_raises_output_error(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-4-26b-a4b")
    reset_settings_cache()

    def fake_chat(*, region: str, payload: dict) -> dict:
        del region, payload
        return _completion({"document_type": "Mill Test Certificate", "confidence_score": 0.1})

    monkeypatch.setattr(
        "app.services.extraction_providers.bedrock_provider.chat_completions",
        fake_chat,
    )
    provider = BedrockGemmaProvider()
    with pytest.raises(ExtractionOutputError, match="schema"):
        provider.extract_metadata([])


def test_mantle_client_requires_credentials(monkeypatch):
    class FakeSession:
        def get_credentials(self):
            return None

    monkeypatch.setattr(
        "app.services.extraction_providers.mantle_client.Session",
        FakeSession,
    )
    from app.services.extraction_providers.mantle_client import chat_completions

    with pytest.raises(ExtractionProviderError, match="credentials"):
        chat_completions(region="eu-central-1", payload={"model": "google.gemma-4-26b-a4b"})


def test_mantle_client_http_error_is_generic(monkeypatch):
    class Frozen:
        access_key = "AKIAFAKE"
        secret_key = "secret"
        token = None

    class Creds:
        def get_frozen_credentials(self):
            return Frozen()

    class FakeSession:
        def get_credentials(self):
            return Creds()

    class FakeResponse:
        status_code = 403

        def json(self):
            return {"error": "denied"}

    monkeypatch.setattr(
        "app.services.extraction_providers.mantle_client.Session",
        FakeSession,
    )
    monkeypatch.setattr(
        "app.services.extraction_providers.mantle_client.SigV4Auth.add_auth",
        lambda self, request: None,
    )
    monkeypatch.setattr(
        "app.services.extraction_providers.mantle_client.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    from app.services.extraction_providers.mantle_client import chat_completions

    with pytest.raises(ExtractionProviderError, match="Bedrock Mantle request failed"):
        chat_completions(region="eu-central-1", payload={"model": "x"})
