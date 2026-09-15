from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.security import decode_access_token
from app.services.jobs.service import (
    create_queued_job,
    get_job,
    mark_job_failed,
    mark_job_succeeded,
    store_job_result,
)
from db.session import SessionLocal


def test_job_status_and_result_flow(client: TestClient, auth_headers: dict[str, str], minimal_pdf_bytes: bytes):
    with patch("app.api.v1.routes.uploads.enqueue_job", return_value=None):
        upload = client.post(
            "/api/v1/uploads",
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
            headers=auth_headers,
        )
    assert upload.status_code == 202, upload.text
    job_id = upload.json()["job_id"]

    status_resp = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "queued"
    assert status_resp.json()["analysis_id"] is None

    result_resp = client.get(f"/api/v1/jobs/{job_id}/result", headers=auth_headers)
    assert result_resp.status_code == 425

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        assert job is not None
        result_key = store_job_result(job_id, {"status": "needs_review", "items": []})
        mark_job_succeeded(db, job, result_key, analysis_run_id=42)
    finally:
        db.close()

    done = client.get(f"/api/v1/jobs/{job_id}/result", headers=auth_headers)
    assert done.status_code == 200
    assert done.json()["status"] == "succeeded"
    assert done.json()["analysis_id"] == 42
    assert done.json()["result"]["status"] == "needs_review"


def test_job_not_found(client: TestClient, auth_headers: dict[str, str]):
    assert client.get("/api/v1/jobs/does-not-exist", headers=auth_headers).status_code == 404


def test_job_requires_auth(client: TestClient):
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 401


def test_failed_and_cancelled_job_results(client: TestClient, auth_headers: dict[str, str]):
    db = SessionLocal()
    try:
        token = auth_headers["Authorization"].split(" ", 1)[1]
        user_id = int(decode_access_token(token)["sub"])
        failed = create_queued_job(
            db, input_object_key="uploads/f.pdf", original_filename="f.pdf", user_id=user_id
        )
        mark_job_failed(db, failed, "Document processing failed.")
        cancelled = create_queued_job(
            db, input_object_key="uploads/c.pdf", original_filename="c.pdf", user_id=user_id
        )
        cancelled.status = "cancelled"
        db.commit()
        failed_id, cancelled_id = failed.id, cancelled.id
    finally:
        db.close()

    failed_resp = client.get(f"/api/v1/jobs/{failed_id}/result", headers=auth_headers)
    assert failed_resp.status_code == 200
    assert failed_resp.json()["status"] == "failed"
    assert failed_resp.json()["error_message"] == "Document processing failed."

    cancelled_resp = client.get(f"/api/v1/jobs/{cancelled_id}/result", headers=auth_headers)
    assert cancelled_resp.status_code == 409
