from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_current_user
from app.models_db import User
from app.services.jobs.schemas import JobResultResponse, JobStatusResponse
from app.services.jobs.service import get_owned_job, job_status_response, load_job_result
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = get_owned_job(db, job_id, current_user.id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job_status_response(job)


@router.get("/{job_id}/result", response_model=JobResultResponse)
def get_job_result(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = get_owned_job(db, job_id, current_user.id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if job.status in {"queued", "processing"}:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Job is still processing.",
        )
    if job.status == "cancelled":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job was cancelled.")
    if job.status == "failed":
        return JobResultResponse(job_id=job.id, status="failed", error_message=job.error_message)
    return load_job_result(job)
