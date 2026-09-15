from __future__ import annotations

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.services.jobs.service import create_queued_job, mark_job_succeeded, store_job_result
from db.models import User
from db.session import SessionLocal


def _register(client: TestClient) -> dict[str, str]:
    email = f"bola-{uuid.uuid4().hex}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_upload_requires_authentication(client: TestClient, minimal_pdf_bytes: bytes):
    response = client.post(
        "/api/v1/uploads",
        files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 401


def test_user_cannot_read_another_users_job(client: TestClient, minimal_pdf_bytes: bytes):
    owner = _register(client)
    other = _register(client)

    with patch("app.api.v1.routes.uploads.enqueue_job", return_value=None):
        upload = client.post(
            "/api/v1/uploads",
            files={"file": ("sample.pdf", minimal_pdf_bytes, "application/pdf")},
            headers=owner,
        )
    assert upload.status_code == 202
    job_id = upload.json()["job_id"]

    status = client.get(f"/api/v1/jobs/{job_id}", headers=other)
    result = client.get(f"/api/v1/jobs/{job_id}/result", headers=other)
    assert status.status_code == 404
    assert result.status_code == 404

    own_status = client.get(f"/api/v1/jobs/{job_id}", headers=owner)
    assert own_status.status_code == 200


def test_unauthenticated_cannot_read_existing_job(client: TestClient):
    db = SessionLocal()
    try:
        user = User(email=f"anon-job-{uuid.uuid4().hex}@example.com", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        job = create_queued_job(
            db,
            input_object_key="uploads/x.pdf",
            original_filename="x.pdf",
            user_id=user.id,
        )
        job_id = job.id
        result_key = store_job_result(job_id, {"secret": "should-not-leak"})
        mark_job_succeeded(db, job, result_key)
    finally:
        db.close()

    status = client.get(f"/api/v1/jobs/{job_id}")
    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert status.status_code == 401
    assert result.status_code == 401
    assert "should-not-leak" not in status.text
    assert "should-not-leak" not in result.text
