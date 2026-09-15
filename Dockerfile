FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 qualiflow

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=qualiflow:qualiflow app ./app
COPY --chown=qualiflow:qualiflow config ./config
COPY --chown=qualiflow:qualiflow db ./db
COPY --chown=qualiflow:qualiflow alembic ./alembic
COPY --chown=qualiflow:qualiflow alembic.ini ./
COPY --chown=qualiflow:qualiflow scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod +x /app/scripts/docker-entrypoint.sh \
    && mkdir -p /app/data/storage/pdfs \
                 /app/data/storage/artifacts \
                 /app/data/storage/outputs \
    && chown -R qualiflow:qualiflow /app/data /app/.venv

USER qualiflow

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
