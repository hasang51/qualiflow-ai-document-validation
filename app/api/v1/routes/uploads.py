from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.deps import get_current_user
from app.observability.logging import request_id_ctx, trace_id_ctx
from app.observability.metrics import record_job_status
from app.rate_limit import limiter
from app.services.document_processor import sha256_bytes, validate_pdf_upload
from app.services.jobs.service import create_queued_job, enqueue_job, job_create_response
from app.services.storage.factory import get_storage_backend
from config.settings import get_settings
from db.models import User
from db.session import get_db

logger = logging.getLogger("qualiflow.uploads")
router = APIRouter(prefix="/uploads", tags=["uploads"])
_settings = get_settings()


async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    content = await file.read()
    validate_pdf_upload(content, file.content_type, file.filename)

    file_hash = sha256_bytes(content)
    object_key = f"uploads/{file_hash[:16]}/{uuid.uuid4().hex}.pdf"
    storage = get_storage_backend()
    storage.put_bytes(object_key, content, content_type="application/pdf")

    trace_id = trace_id_ctx.get() or request_id_ctx.get()
    job = create_queued_job(
        db,
        input_object_key=object_key,
        original_filename=file.filename or "document.pdf",
        user_id=current_user.id,
        trace_id=trace_id,
    )
    record_job_status("queued")
    enqueue_job(job.id)
    logger.info("Queued job %s object_key=%s user_id=%s", job.id, object_key, current_user.id)

    response = job_create_response(job)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=response.model_dump(),
        headers={"Location": response.poll_url},
    )


_upload_handler: Callable[..., Any] = upload_document
if _settings.rate_limit_enabled:
    _upload_handler = limiter.limit(_settings.rate_limit_upload)(upload_document)

router.add_api_route(
    "",
    _upload_handler,
    methods=["POST"],
    status_code=status.HTTP_202_ACCEPTED,
)
