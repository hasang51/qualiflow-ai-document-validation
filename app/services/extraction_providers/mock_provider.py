from __future__ import annotations

from typing import Any

from app.services.extraction_providers.base import ExtractionProvider, ExtractionProviderError, PayloadDict, UsageDict
from app.services.preprocessing import ProcessedPage


class MockExtractionProvider:
    """Deterministic Stage A/B stand-in for unit tests. Does not call a remote LLM."""

    name = "mock"

    def __init__(
        self,
        *,
        metadata: PayloadDict | None = None,
        items_payload: PayloadDict | None = None,
        fail_with: str | None = None,
    ) -> None:
        self._metadata = metadata or {
            "supplier_name": "Mock Supplier",
            "document_type": "Mill Test Certificate",
            "product_category": "OTHER",
            "confidence_score": 0.5,
        }
        self._items_payload = items_payload or {
            "total_items_detected": 0,
            "items": [],
        }
        self._fail_with = fail_with

    def extract_metadata(self, pages: list[ProcessedPage]) -> tuple[PayloadDict, UsageDict]:
        del pages
        if self._fail_with:
            raise ExtractionProviderError(self._fail_with)
        return dict(self._metadata), {"input_tokens": 0, "output_tokens": 0}

    def extract_line_items(
        self,
        pages: list[ProcessedPage],
        metadata: dict[str, Any],
    ) -> tuple[PayloadDict, UsageDict]:
        del pages, metadata
        if self._fail_with:
            raise ExtractionProviderError(self._fail_with)
        return dict(self._items_payload), {"input_tokens": 0, "output_tokens": 0}
