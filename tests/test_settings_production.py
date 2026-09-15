from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_production_rejects_insecure_jwt():
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        Settings(
            app_env="production",
            jwt_secret_key="change-me",
            cors_allow_origins="https://app.example.com",
            database_url="postgresql+psycopg2://user:pass@db:5432/qualiflow",
        )


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="CORS_ALLOW_ORIGINS"):
        Settings(
            app_env="production",
            jwt_secret_key="x" * 32,
            cors_allow_origins="*",
            database_url="postgresql+psycopg2://user:pass@db:5432/qualiflow",
        )


def test_production_rejects_sqlite():
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(
            app_env="production",
            jwt_secret_key="x" * 32,
            cors_allow_origins="https://app.example.com",
            database_url="sqlite:///./data/qualiflow.db",
        )


def test_production_rejects_debug():
    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(
            app_env="production",
            jwt_secret_key="x" * 32,
            cors_allow_origins="https://app.example.com",
            database_url="postgresql+psycopg2://user:pass@db:5432/qualiflow",
            debug=True,
            rate_limit_enabled=True,
        )


def test_production_rejects_disabled_rate_limit():
    with pytest.raises(ValidationError, match="RATE_LIMIT_ENABLED"):
        Settings(
            app_env="production",
            jwt_secret_key="x" * 32,
            cors_allow_origins="https://app.example.com",
            database_url="postgresql+psycopg2://user:pass@db:5432/qualiflow",
            rate_limit_enabled=False,
        )


def test_production_accepts_secure_config():
    settings = Settings(
        app_env="production",
        jwt_secret_key="x" * 32,
        cors_allow_origins="https://app.example.com",
        database_url="postgresql+psycopg2://user:pass@db:5432/qualiflow",
        debug=False,
        rate_limit_enabled=True,
    )
    assert settings.is_production
    assert settings.rate_limit_enabled
    settings = Settings(
        app_env="local",
        database_url="postgresql+psycopg2://qualiflow:qualiflow@localhost:5432/qualiflow",
        object_storage_backend="s3",
    )
    assert settings.database_url.startswith("postgresql")
    assert settings.object_storage_backend == "s3"
