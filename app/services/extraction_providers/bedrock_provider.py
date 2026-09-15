from __future__ import annotations

from typing import Any

from app.services.extraction_providers.base import ExtractionProviderError, PayloadDict, UsageDict
from app.services.preprocessing import ProcessedPage
from config.settings import get_settings


class BedrockGemmaProvider:
    """Amazon Bedrock Gemma extraction boundary.

    Invoke is wired in a later M2 commit. This class remains the factory target.
    """

    name = "bedrock"

    def extract_metadata(self, pages: list[ProcessedPage]) -> tuple[PayloadDict, UsageDict]:
        del pages
        self._ensure_configured()
        raise ExtractionProviderError(
            "BedrockGemmaProvider is a boundary only; invoke_model is not wired in M1."
        )

    def extract_line_items(
        self,
        pages: list[ProcessedPage],
        metadata: dict[str, Any],
    ) -> tuple[PayloadDict, UsageDict]:
        del pages, metadata
        self._ensure_configured()
        raise ExtractionProviderError(
            "BedrockGemmaProvider is a boundary only; invoke_model is not wired in M1."
        )

    def _ensure_configured(self) -> None:
        settings = get_settings()
        if not settings.bedrock_model_id.strip():
            raise ExtractionProviderError(
                "BEDROCK_MODEL_ID is required when EXTRACTION_PROVIDER=bedrock."
            )


BedrockExtractionProvider = BedrockGemmaProvider
