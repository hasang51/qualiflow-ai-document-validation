from __future__ import annotations

from unittest.mock import patch

from app.services.jobs.service import create_queued_job, get_job
from app.workers.tasks import process_document_job
from db.models import AnalysisRun, User
from db.session import SessionLocal


def _minimal_extraction_result() -> dict:
    return {
        "supplier_name": "Test Supplier",
        "document_type": "MTC",
        "total_items_detected": 0,
        "items": [],
        "confidence_score": 0.8,
        "needs_review": True,
        "status": "NEEDS_REVIEW",
        "preprocessing_meta": {
            "file_sha256": "abc123",
            "page_count": 1,
        },
    }


def test_worker_success_path(minimal_pdf_bytes: bytes):
    from app.services.storage.factory import get_storage_backend, reset_storage_backend_cache
    from config.settings import reset_settings_cache

    reset_settings_cache()
    reset_storage_backend_cache()
    db = SessionLocal()
    try:
        from app.services.storage.factory import get_storage_backend

        storage = get_storage_backend()
        key = "uploads/test/worker.pdf"
        storage.put_bytes(key, minimal_pdf_bytes, content_type="application/pdf")
        job = create_queued_job(db, input_object_key=key, original_filename="worker.pdf")
        job_id = job.id
    finally:
        db.close()

    fake_result = _minimal_extraction_result()
    with patch("app.workers.tasks.process_pdf_bytes", return_value=fake_result):
        process_document_job(job_id)

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result_object_key
    finally:
        db.close()


def test_worker_persists_analysis_for_authenticated_user(minimal_pdf_bytes: bytes):
    from app.services.storage.factory import reset_storage_backend_cache
    from config.settings import reset_settings_cache

    reset_settings_cache()
    reset_storage_backend_cache()
    user_id: int
    job_id: str
    db = SessionLocal()
    try:
        from app.services.storage.factory import get_storage_backend

        user = User(email="worker-test@example.com", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        user_id = user.id

        storage = get_storage_backend()
        key = "uploads/test/authenticated.pdf"
        storage.put_bytes(key, minimal_pdf_bytes, content_type="application/pdf")
        job = create_queued_job(
            db,
            input_object_key=key,
            original_filename="authenticated.pdf",
            user_id=user_id,
        )
        job_id = job.id
    finally:
        db.close()

    fake_result = _minimal_extraction_result()
    with patch("app.workers.tasks.process_pdf_bytes", return_value=fake_result):
        process_document_job(job_id)

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.analysis_run_id is not None

        run = db.get(AnalysisRun, job.analysis_run_id)
        assert run is not None
        assert run.user_id == user_id
        assert run.supplier_name == "Test Supplier"
        assert run.document is not None
        assert run.document.stored_pdf_path == key
    finally:
        db.close()


def test_worker_failure_path(minimal_pdf_bytes: bytes):
    from app.services.storage.factory import reset_storage_backend_cache
    from config.settings import reset_settings_cache

    reset_settings_cache()
    reset_storage_backend_cache()
    db = SessionLocal()
    try:
        from app.services.storage.factory import get_storage_backend

        storage = get_storage_backend()
        key = "uploads/test/fail.pdf"
        storage.put_bytes(key, minimal_pdf_bytes, content_type="application/pdf")
        job = create_queued_job(db, input_object_key=key, original_filename="fail.pdf")
        job_id = job.id
    finally:
        db.close()

    with patch("app.workers.tasks.process_pdf_bytes", side_effect=RuntimeError("boom")):
        try:
            process_document_job(job_id)
        except RuntimeError:
            pass

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error_message
    finally:
        db.close()


def test_worker_schema_failure_payload_is_succeeded_review():
    from app.services.storage.factory import reset_storage_backend_cache
    from config.settings import reset_settings_cache

    reset_settings_cache()
    reset_storage_backend_cache()
    db = SessionLocal()
    try:
        from app.services.storage.factory import get_storage_backend

        storage = get_storage_backend()
        key = "uploads/test/schema-fail.pdf"
        storage.put_bytes(key, b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n", content_type="application/pdf")
        job = create_queued_job(db, input_object_key=key, original_filename="schema-fail.pdf")
        job_id = job.id
    finally:
        db.close()

    result = _minimal_extraction_result()
    result["review_reasons"] = ["extraction_schema_invalid", "model_output_unusable"]
    with patch("app.workers.tasks.process_pdf_bytes", return_value=result):
        process_document_job(job_id)

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        assert job is not None
        assert job.status == "succeeded"
    finally:
        db.close()
