from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session

from app.services.extraction_providers.base import ExtractionProviderError

_MANTLE_SERVICE = "bedrock-mantle"
_CHAT_COMPLETIONS_PATH = "/openai/v1/chat/completions"


def mantle_chat_completions_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws{_CHAT_COMPLETIONS_PATH}"


def chat_completions(
    *,
    region: str,
    payload: dict[str, Any],
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """POST Chat Completions to Bedrock Mantle using the default AWS credential chain.

    Never logs the request body (it may contain page images).
    """
    credentials = Session().get_credentials()
    if credentials is None:
        raise ExtractionProviderError("AWS credentials were not found for Bedrock Mantle.")
    frozen = credentials.get_frozen_credentials()
    url = mantle_chat_completions_url(region)
    body = json.dumps(payload)
    host = urlparse(url).netloc
    request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json", "Host": host},
    )
    SigV4Auth(frozen, _MANTLE_SERVICE, region).add_auth(request)
    prepared = request.prepare()
    try:
        response = requests.post(
            prepared.url,
            data=prepared.body,
            headers=dict(prepared.headers),
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise ExtractionProviderError("Bedrock Mantle request failed.") from exc
    if response.status_code >= 400:
        raise ExtractionProviderError("Bedrock Mantle request failed.")
    try:
        parsed = response.json()
    except ValueError as exc:
        raise ExtractionProviderError("Bedrock Mantle returned a non-JSON response.") from exc
    if not isinstance(parsed, dict):
        raise ExtractionProviderError("Bedrock Mantle returned an unexpected payload.")
    return parsed
