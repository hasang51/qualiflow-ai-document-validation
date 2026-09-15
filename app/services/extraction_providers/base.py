from __future__ import annotations

from typing import Any, Protocol

from app.services.preprocessing import ProcessedPage

UsageDict = dict[str, int]
PayloadDict = dict[str, Any]


class ExtractionProviderError(RuntimeError):
    """Raised when a vision extraction backend fails. Callers must not leak ``str(self)`` to API clients."""


class ExtractionProvider(Protocol):
    """Stage A/B vision extraction only. Domain validation stays in the pipeline."""

    name: str

    def extract_metadata(self, pages: list[ProcessedPage]) -> tuple[PayloadDict, UsageDict]:
        """Return ``(metadata_dict, usage_dict)``."""
        ...

    def extract_line_items(
        self,
        pages: list[ProcessedPage],
        metadata: PayloadDict,
    ) -> tuple[PayloadDict, UsageDict]:
        """Return ``(item_payload_dict, usage_dict)``."""
        ...
