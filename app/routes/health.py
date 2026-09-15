from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.services.validator import MATERIAL_SPECS

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "QualiFlow Document Extraction API",
        "version": "6.0.0",
        "model": settings.bedrock_model_id,
        "engine": settings.extraction_provider,
        "known_grades": list(MATERIAL_SPECS.keys()),
        "supported_documents": [
            "Certificate of Analysis",
            "Mill Test Certificate",
            "Packing List",
            "Inspection Certificate",
        ],
    }
