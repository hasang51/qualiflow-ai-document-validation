from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_upload_returns_202_with_job_id(client: TestClient, auth_headers: dict[str, str], minimal_pdf_bytes: bytes):
    with patch("app.api.v1.routes.uploads.enqueue_job", return_value="rq-1"):
        response = client.post(
            "/api/v1/uploads",
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
            headers=auth_headers,
        )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]
    assert body["poll_url"] == f"/api/v1/jobs/{body['job_id']}"


def test_upload_rejects_non_pdf(client: TestClient, auth_headers: dict[str, str]):
    response = client.post(
        "/api/v1/uploads",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 415
