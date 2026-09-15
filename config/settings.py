from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent

_INSECURE_JWT_SECRETS = frozenset(
    {"", "change-me", "change-me-in-production", "changeme", "secret", "test-secret"}
)


def _load_dotenv_files() -> None:
    from dotenv import load_dotenv

    root_env = _REPO_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
    docs_env = _REPO_ROOT / "docs" / ".env"
    if docs_env.exists():
        load_dotenv(docs_env, override=False)
    load_dotenv(override=False)


_load_dotenv_files()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg2://qualiflow:qualiflow@localhost:5432/qualiflow"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 240

    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    rate_limit_enabled: bool = False
    rate_limit_default: str = "60/minute"
    rate_limit_upload: str = "10/minute"

    object_storage_backend: Literal["local", "s3"] = "s3"
    storage_dir: Path = Path("./data/storage")
    s3_endpoint_url: str | None = None
    s3_bucket_name: str = "qualiflow"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"

    anthropic_api_key: str = ""
    anthropic_model: str = Field(default="", description="Anthropic model id; must be set via env for LLM calls")
    extraction_provider: Literal["anthropic", "mock", "bedrock"] = "anthropic"
    bedrock_region: str = "eu-central-1"
    bedrock_model_id: str = ""

    pdf_dpi: int = 400
    poppler_path: str = ""
    review_confidence_threshold: float = 0.75
    max_upload_mb: int = 20
    max_pages_for_llm: int = 8
    llm_image_max_edge: int = 1600
    llm_image_target_bytes: int = 4_500_000
    llm_image_min_edge: int = 900
    llm_jpeg_quality: int = 82
    ocr_backend: str = "pytesseract"
    tesseract_cmd: str = ""
    ocr_fail_loudly: bool = False
    debug_field_provenance: bool = False

    rq_queue_name: str = "qualiflow"
    worker_enabled: bool = True

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _coerce_storage_dir(cls, value: str | Path) -> Path:
        return Path(value)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_allow_origins.strip():
            return []
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @model_validator(mode="after")
    def _validate_production_safety(self) -> Settings:
        if not self.is_production:
            return self
        errors: list[str] = []
        secret = self.jwt_secret_key.strip().lower()
        if secret in _INSECURE_JWT_SECRETS or len(self.jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY must be set to a secure value (>=32 chars) in production.")
        if not self.cors_origins_list or "*" in self.cors_origins_list:
            errors.append("CORS_ALLOW_ORIGINS must be an explicit origin list (no '*') in production.")
        if self.database_url.startswith("sqlite"):
            errors.append("DATABASE_URL must use PostgreSQL in production (SQLite is not allowed).")
        if self.debug:
            errors.append("DEBUG must be false in production.")
        if not self.rate_limit_enabled:
            errors.append("RATE_LIMIT_ENABLED must be true in production.")
        if errors:
            raise ValueError(" ".join(errors))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


# Eager validation in production so misconfiguration fails at import/startup.
if os.getenv("APP_ENV", "local").lower() == "production":
    get_settings()
