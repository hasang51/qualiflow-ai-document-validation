from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.anthropic_client import get_anthropic_client
from app.services.extraction_providers.base import ExtractionProviderError, PayloadDict, UsageDict
from app.services.preprocessing import ProcessedPage


class AnthropicExtractionProvider:
    """Claude Messages API Stage A/B backend."""

    name = "anthropic"

    def extract_metadata(self, pages: list[ProcessedPage]) -> tuple[PayloadDict, UsageDict]:
        from app.services.extraction_pipeline import (
            METADATA_PROMPT,
            METADATA_TOOL,
            _extract_tool_input,
            _extract_usage,
            _metadata_blocks,
        )

        try:
            client = get_anthropic_client()
            metadata_resp = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=2048,
                temperature=0.0,
                system=METADATA_PROMPT,
                messages=[{"role": "user", "content": _metadata_blocks(pages)}],  # type: ignore[arg-type]
                tools=[METADATA_TOOL],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "submit_document_metadata"},
            )
            return _extract_tool_input(metadata_resp, "submit_document_metadata"), _extract_usage(metadata_resp)
        except ExtractionProviderError:
            raise
        except Exception as exc:
            raise ExtractionProviderError("Anthropic metadata extraction failed.") from exc

    def extract_line_items(
        self,
        pages: list[ProcessedPage],
        metadata: dict[str, Any],
    ) -> tuple[PayloadDict, UsageDict]:
        from app.services.extraction_pipeline import (
            ITEM_PROMPT,
            ITEM_TOOL,
            _extract_tool_input,
            _extract_usage,
            _items_blocks,
        )

        document_type = metadata.get("document_type", "Unknown document")
        supplier_name = metadata.get("supplier_name", "Unknown supplier")
        product_category = metadata.get("product_category", "OTHER")
        row_prompt_context = (
            f"Document type identified in Stage A: {document_type}. "
            f"Supplier identified in Stage A: {supplier_name}. "
            f"Product category identified in Stage A: {product_category}. "
            "Use that only as context. Do not invent missing line items or unreadable cell values."
        )
        try:
            client = get_anthropic_client()
            blocks = _items_blocks(pages)
            blocks.append({"type": "text", "text": row_prompt_context})
            item_resp = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4096,
                temperature=0.0,
                system=ITEM_PROMPT,
                messages=[{"role": "user", "content": blocks}],  # type: ignore[arg-type]
                tools=[ITEM_TOOL],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "submit_line_items"},
            )
            return _extract_tool_input(item_resp, "submit_line_items"), _extract_usage(item_resp)
        except ExtractionProviderError:
            raise
        except Exception as exc:
            raise ExtractionProviderError("Anthropic line-item extraction failed.") from exc
