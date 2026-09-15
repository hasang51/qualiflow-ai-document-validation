from __future__ import annotations

import os

from fastapi.testclient import TestClient

from config.settings import reset_settings_cache


def test_rate_limit_can_be_disabled(client: TestClient, auth_headers: dict[str, str], minimal_pdf_bytes: bytes):
    from unittest.mock import patch

    with patch("app.api.v1.routes.uploads.enqueue_job", return_value=None):
        for _ in range(3):
            response = client.post(
                "/api/v1/uploads",
                files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 202


def test_health_endpoints_not_rate_limited_heavily(client: TestClient):
    for _ in range(5):
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code in {200, 503}
