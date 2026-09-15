from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from app.services.evidence import (
    pack_evidence,
    select_metadata_evidence,
    select_row_evidence,
    to_openai_image_content,
)
from app.services.extraction_providers.base import (
    ExtractionOutputError,
    ExtractionProviderError,
    PayloadDict,
    UsageDict,
)
from app.services.extraction_providers.mantle_client import chat_completions
from app.services.extraction_providers.output_schemas import (
    StageAMetadata,
    StageBLineItems,
    response_format_for_model,
)
from app.services.prompt_safety import with_untrusted_preamble
from app.services.preprocessing import ProcessedPage
from config.settings import get_settings

_PROMPT_OVERHEAD_BYTES = 80_000
_STAGE_A_MAX_TOKENS = 2048
_STAGE_B_MAX_TOKENS = 4096


class BedrockGemmaProvider:
    """Stage A/B extraction via Amazon Bedrock Mantle (Gemma 4 26B-A4B)."""

    name = "bedrock"

    def extract_metadata(self, pages: list[ProcessedPage]) -> tuple[PayloadDict, UsageDict]:
        from app.services.extraction_pipeline import METADATA_PROMPT

        self._ensure_configured()
        settings = get_settings()
        packed = pack_evidence(
            select_metadata_evidence(pages),
            max_request_bytes=settings.bedrock_max_request_bytes,
            prompt_overhead_bytes=_PROMPT_OVERHEAD_BYTES,
        )
        content = to_openai_image_content(packed, "Extract document-level metadata only.")
        payload = self._chat_payload(
            system=with_untrusted_preamble(METADATA_PROMPT),
            content=content,
            max_tokens=_STAGE_A_MAX_TOKENS,
            response_format=response_format_for_model(StageAMetadata, "submit_document_metadata", strict=True),
        )
        raw, usage = self._invoke(payload)
        try:
            parsed = StageAMetadata.model_validate(raw)
        except ValidationError as exc:
            raise ExtractionOutputError("Stage A metadata failed schema validation.") from exc
        return parsed.model_dump(), usage

    def extract_line_items(
        self,
        pages: list[ProcessedPage],
        metadata: dict[str, Any],
    ) -> tuple[PayloadDict, UsageDict]:
        from app.services.extraction_pipeline import ITEM_PROMPT

        self._ensure_configured()
        settings = get_settings()
        document_type = metadata.get("document_type", "Unknown document")
        supplier_name = metadata.get("supplier_name", "Unknown supplier")
        product_category = metadata.get("product_category", "OTHER")
        row_prompt_context = (
            f"Document type identified in Stage A: {document_type}. "
            f"Supplier identified in Stage A: {supplier_name}. "
            f"Product category identified in Stage A: {product_category}. "
            "Use that only as context. Do not invent missing line items or unreadable cell values."
        )
        packed = pack_evidence(
            select_row_evidence(pages),
            max_request_bytes=settings.bedrock_max_request_bytes,
            prompt_overhead_bytes=_PROMPT_OVERHEAD_BYTES,
        )
        content = to_openai_image_content(
            packed,
            "Step 1: Fill extraction_audit — list every column header you can physically see "
            "in the table (1-2 sentences). "
            "Step 2: Extract line items strictly from those visible columns. "
            "Any field without a corresponding visible column must be null. "
            f"{row_prompt_context}",
        )
        payload = self._chat_payload(
            system=with_untrusted_preamble(ITEM_PROMPT),
            content=content,
            max_tokens=_STAGE_B_MAX_TOKENS,
            # mechanical_table_rows allows extra keys, so Stage B cannot use strict json_schema.
            response_format=response_format_for_model(StageBLineItems, "submit_line_items", strict=False),
        )
        raw, usage = self._invoke(payload)
        try:
            parsed = StageBLineItems.model_validate(raw)
        except ValidationError as exc:
            raise ExtractionOutputError("Stage B line items failed schema validation.") from exc
        return parsed.model_dump(), usage

    def _chat_payload(
        self,
        *,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        settings = get_settings()
        return {
            "model": settings.bedrock_model_id,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }

    def _invoke(self, payload: dict[str, Any]) -> tuple[PayloadDict, UsageDict]:
        settings = get_settings()
        started = time.perf_counter()
        try:
            response = chat_completions(region=settings.bedrock_region, payload=payload)
        except ExtractionProviderError:
            raise
        except Exception as exc:
            raise ExtractionProviderError("Bedrock Gemma extraction failed.") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = _parse_json_content(response)
        usage = _extract_usage(response, latency_ms=latency_ms)
        return parsed, usage

    def _ensure_configured(self) -> None:
        settings = get_settings()
        if not settings.bedrock_model_id.strip():
            raise ExtractionProviderError(
                "BEDROCK_MODEL_ID is required when EXTRACTION_PROVIDER=bedrock."
            )


def _parse_json_content(response: dict[str, Any]) -> PayloadDict:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ExtractionOutputError("Bedrock Mantle response missing choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ExtractionOutputError("Bedrock Mantle response missing message.")
    if message.get("refusal"):
        raise ExtractionOutputError("Bedrock Mantle refused the extraction request.")
    raw_content = message.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ExtractionOutputError("Bedrock Mantle returned empty content.")
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ExtractionOutputError("Bedrock Mantle returned non-JSON content.") from exc
    if not isinstance(parsed, dict):
        raise ExtractionOutputError("Bedrock Mantle JSON was not an object.")
    return parsed


def _extract_usage(response: dict[str, Any], *, latency_ms: int) -> UsageDict:
    raw_usage = response.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "latency_ms": int(latency_ms),
    }


BedrockExtractionProvider = BedrockGemmaProvider
