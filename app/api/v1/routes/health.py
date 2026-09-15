from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from app.config import settings
from app.observability.metrics import render_prometheus_metrics
from db.session import engine

router = APIRouter(tags=["ops"])


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    checks: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    if settings.worker_enabled:
        try:
            from redis import Redis

            Redis.from_url(settings.redis_url).ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "error"

    ready = all(value == "ok" for value in checks.values())
    status_code = 200 if ready else 503
    from fastapi.responses import JSONResponse

    return JSONResponse({"ready": ready, "checks": checks}, status_code=status_code)


@router.get("/metrics")
async def metrics():
    return PlainTextResponse(render_prometheus_metrics(), media_type="text/plain; version=0.0.4")
