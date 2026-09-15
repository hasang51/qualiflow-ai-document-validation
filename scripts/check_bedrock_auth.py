"""Minimal Bedrock Mantle auth/smoke check using the default AWS credential chain.

Exit codes:
* 0 when the signed Chat Completions call succeeds
* 2 when credentials or model id are missing
* 4 when Mantle returns an error
"""

from __future__ import annotations

import json
import sys

from app.services.extraction_providers.base import ExtractionProviderError
from app.services.extraction_providers.mantle_client import chat_completions
from app.services.extraction_providers.output_schemas import StageAMetadata, response_format_for_model
from config.settings import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.bedrock_model_id.strip():
        print("FAIL: BEDROCK_MODEL_ID is empty.")
        return 2
    print(f"Region: {settings.bedrock_region}")
    print(f"Model: {settings.bedrock_model_id}")
    payload = {
        "model": settings.bedrock_model_id,
        "temperature": 0,
        "max_tokens": 32,
        "response_format": response_format_for_model(StageAMetadata, "submit_document_metadata", strict=True),
        "messages": [
            {"role": "system", "content": "Return empty unknown metadata."},
            {
                "role": "user",
                "content": "supplier_name Unknown supplier, document_type Unknown document, confidence_score 0",
            },
        ],
    }
    try:
        response = chat_completions(region=settings.bedrock_region, payload=payload, timeout_s=60)
    except ExtractionProviderError:
        print("FAIL: Bedrock Mantle request failed.")
        return 4
    content = (((response.get("choices") or [{}])[0].get("message") or {}).get("content"))
    print("OK: Bedrock Mantle auth succeeded.")
    if isinstance(content, str) and content.strip():
        print(f"Content bytes: {len(content.encode('utf-8'))}")
        try:
            json.loads(content)
            print("json_schema content parsed.")
        except json.JSONDecodeError:
            print("WARN: content was not JSON; required-tool fallback is out of M2 scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
