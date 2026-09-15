from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_v1_router
from app.api.v1.routes.health import router as ops_router
from app.config import settings
from app.observability.logging import RequestIdMiddleware, configure_logging
from app.observability.middleware import MetricsMiddleware
from app.rate_limit import limiter
from app.routes.analyses import router as analyses_router
from app.routes.auth import router as auth_router
from app.routes.extraction import router as extraction_router
from app.routes.health import router as legacy_health_router
from app.services.storage import ensure_storage_dirs
from db.session import Base, engine

logger = logging.getLogger("qualiflow")


def create_app() -> FastAPI:
    configure_logging(production=settings.is_production)

    app = FastAPI(
        title="QualiFlow - Document Analysis API",
        version="7.0.0",
        debug=settings.debug,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, StarletteHTTPException):
            return await http_exception_handler(request, exc)
        if isinstance(exc, RequestValidationError):
            return await request_validation_exception_handler(request, exc)
        logger.exception("Unhandled error on %s", request.url.path, exc_info=exc)
        request_id = getattr(request.state, "request_id", None)
        from app.observability.logging import request_id_ctx

        rid = request_id or request_id_ctx.get()
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": rid},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIdMiddleware)

    ensure_storage_dirs()
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)

    app.include_router(ops_router)
    app.include_router(legacy_health_router)
    app.include_router(auth_router)
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
    app.include_router(extraction_router)
    app.include_router(analyses_router)

    logger.info("QualiFlow started app_env=%s storage=%s", settings.app_env, settings.object_storage_backend)
    return app


app = create_app()
