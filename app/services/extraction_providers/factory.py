from __future__ import annotations

from functools import lru_cache

from app.services.extraction_providers.base import ExtractionProvider
from config.settings import get_settings


@lru_cache
def get_extraction_provider() -> ExtractionProvider:
    settings = get_settings()
    name = settings.extraction_provider
    if name == "mock":
        from app.services.extraction_providers.mock_provider import MockExtractionProvider

        return MockExtractionProvider()
    if name == "bedrock":
        from app.services.extraction_providers.bedrock_provider import BedrockExtractionProvider

        return BedrockExtractionProvider()
    from app.services.extraction_providers.anthropic_provider import AnthropicExtractionProvider

    return AnthropicExtractionProvider()


def reset_extraction_provider_cache() -> None:
    get_extraction_provider.cache_clear()
