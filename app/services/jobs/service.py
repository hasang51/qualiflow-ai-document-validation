from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.services.jobs.schemas import JobCreateResponse, JobResultResponse, JobStatusResponse
from app.services.storage.factory import get_storage_backend
from config.settings import get_settings
from db.models import Job

logger = logging.getLogger("qualiflow.jobs")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_queued_job(
    db: Session,
    *,
    input_object_key: str,
    original_filename: str,
    user_id: int | None = None,
    trace_id: str | None = None,
) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        status="queued",
        input_object_key=input_object_key,
        original_filename=original_filename,
        user_id=user_id,
        trace_id=trace_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Job | None:
    return db.get(Job, job_id)


def get_owned_job(db: Session, job_id: str, user_id: int) -> Job | None:
    job = get_job(db, job_id)
    if job is None or job.user_id != user_id:
        return None
    return job


def job_status_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,  # type: ignore[arg-type]
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        trace_id=job.trace_id,
        analysis_id=job.analysis_run_id,
    )


def job_create_response(job: Job) -> JobCreateResponse:
    settings = get_settings()
    poll_url = f"{settings.api_v1_prefix}/jobs/{job.id}"
    return JobCreateResponse(job_id=job.id, status=job.status, poll_url=poll_url)  # type: ignore[arg-type]


def mark_job_processing(db: Session, job: Job) -> Job:
    job.status = "processing"
    job.started_at = utcnow()
    job.updated_at = utcnow()
    db.commit()
    db.refresh(job)
    return job


def mark_job_succeeded(
    db: Session,
    job: Job,
    result_object_key: str,
    *,
    analysis_run_id: int | None = None,
) -> Job:
    job.status = "succeeded"
    job.result_object_key = result_object_key
    job.analysis_run_id = analysis_run_id
    job.finished_at = utcnow()
    job.updated_at = utcnow()
    job.error_message = None
    db.commit()
    db.refresh(job)
    return job


def mark_job_failed(db: Session, job: Job, error_message: str) -> Job:
    job.status = "failed"
    job.error_message = error_message[:2000]
    job.finished_at = utcnow()
    job.updated_at = utcnow()
    db.commit()
    db.refresh(job)
    return job


def load_job_result(job: Job) -> JobResultResponse:
    if job.status == "failed":
        return JobResultResponse(
            job_id=job.id,
            status="failed",
            error_message=job.error_message,
            analysis_id=job.analysis_run_id,
        )
    if job.status != "succeeded" or not job.result_object_key:
        return JobResultResponse(
            job_id=job.id,
            status=job.status,  # type: ignore[arg-type]
            analysis_id=job.analysis_run_id,
        )
    storage = get_storage_backend()
    raw = storage.get_bytes(job.result_object_key)
    result = json.loads(raw.decode("utf-8"))
    if job.analysis_run_id is not None and isinstance(result, dict):
        result.setdefault("analysis_id", job.analysis_run_id)
    return JobResultResponse(
        job_id=job.id,
        status="succeeded",
        result=result,
        analysis_id=job.analysis_run_id,
    )


def store_job_result(job_id: str, payload: dict) -> str:
    key = f"results/{job_id}.json"
    storage = get_storage_backend()
    storage.put_bytes(key, json.dumps(payload, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    return key


def enqueue_job(job_id: str) -> str | None:
    settings = get_settings()
    if not settings.worker_enabled:
        logger.info("Worker disabled; job %s left queued.", job_id)
        return None
    try:
        from redis import Redis
        from rq import Queue

        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(settings.rq_queue_name, connection=redis_conn)
        job = queue.enqueue("app.workers.tasks.process_document_job", job_id, job_timeout="30m")
        return job.id
    except Exception:
        logger.exception("Failed to enqueue job %s", job_id)
        return None
