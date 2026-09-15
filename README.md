## Project Leadership

QualiFlow was developed as a two-person Computer Engineering graduation project.

I initiated the project after researching industrial document-verification workflows and led the product definition, technical planning, and system architecture.

My contributions included:

- market and problem research
- product scope and architecture
- backend services and API workflows
- PostgreSQL data design
- asynchronous processing
- AI extraction and deterministic validation
- human-review policy
- testing, documentation, and deployment setup

The project was developed collaboratively with one teammate.
---

## Architecture

```text
Browser (React, :5173)
        │
        ▼
API (FastAPI, :8000)
        │  POST /api/v1/uploads
        ▼
Redis queue ──► Worker (RQ)
        │              │
        │              ├── Bedrock Gemma extraction (or mock in CI)
        │              ├── validation + review policy
        │              └── persist analysis
        ▼
PostgreSQL (:5432)     MinIO S3 (:9000)
  users                  uploads/*.pdf
  analysis_runs          results/*.json
  documents
  jobs
```

### Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Worker | Redis + RQ |
| Database | PostgreSQL 16 (Alembic migrations) |
| Object storage | MinIO (S3-compatible) |
| Frontend | React, TypeScript, Vite, Tailwind |
| Extraction | Amazon Bedrock Gemma 4 26B-A4B (`eu-central-1`, Mantle) or `MockExtractionProvider` for CI |

---

## Quick start (Docker — recommended)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Node.js 20+ (frontend only)
- AWS credentials via the default chain (env, shared config, or instance/task role) for live Bedrock
- IAM permission `bedrock-mantle:CreateInference` (AWS managed policy `AmazonBedrockMantleInferenceAccess`)

### 1. Environment

```powershell
copy .env.example .env
```

For **live** extraction set:

```env
EXTRACTION_PROVIDER=bedrock
BEDROCK_REGION=eu-central-1
BEDROCK_MODEL_ID=google.gemma-4-26b-a4b
```

Do not put AWS access keys in `.env`. Use the default AWS credential chain. Local CI/tests use `EXTRACTION_PROVIDER=mock`.

Estimated Gemma 4 26B-A4B Standard rates in `eu-central-1` (not billing-grade): `$0.16` / 1M input tokens and `$0.48` / 1M output tokens. Override with `BEDROCK_INPUT_COST_PER_MILLION` / `BEDROCK_OUTPUT_COST_PER_MILLION`.

### 2. Start backend services

```powershell
docker compose up --build -d
```

This starts: **api**, **worker**, **postgres**, **redis**, **minio** (+ one-shot `minio-init` bucket setup).

Migrations run automatically on API/worker startup.

### 3. Start frontend

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open **http://localhost:5173**

Or use the helper script from the repo root:

```powershell
.\scripts\start.ps1
```

---

## Auto-start on boot

Services use `restart: unless-stopped` in `docker-compose.yml`. After the first `docker compose up -d`:

1. Open **Docker Desktop**
2. Enable **Settings → General → Start Docker Desktop when you sign in**
3. Containers restart automatically when Docker Desktop starts

You only need to run `npm run dev` in `frontend/` for the UI.

---

## Service URLs

| Service | URL | Credentials |
| --- | --- | --- |
| API | http://localhost:8000 | — |
| API health | http://localhost:8000/healthz | — |
| Frontend | http://localhost:5173 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| PostgreSQL | `localhost:5432` | `qualiflow` / `qualiflow` / db `qualiflow` |

---

## Daily workflow

```powershell
# 1. Ensure Docker Desktop is running (whale icon green)
# 2. Backend (skip if containers already up)
docker compose up -d

# 3. Frontend
cd frontend
npm run dev
```

---

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/auth/register` | Register user |
| `POST` | `/api/v1/auth/login` | Login |
| `GET` | `/api/v1/auth/me` | Current user |
| `POST` | `/api/v1/uploads` | Upload PDF → async job (primary flow) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status |
| `GET` | `/api/v1/jobs/{job_id}/result` | Get extraction result |
| `GET` | `/api/v1/analyses` | List saved analyses |
| `GET` | `/api/v1/analyses/{id}` | Analysis detail |
| `GET` | `/api/v1/documents/{id}/download` | Download source PDF |
| `GET` | `/healthz` | Liveness check |
| `GET` | `/readyz` | Readiness (DB + Redis) |

`POST /api/v1/extract` remains for synchronous/debug use; the UI uses the async upload pipeline.

### Example async flow

```powershell
curl -F "file=@sample.pdf" http://localhost:8000/api/v1/uploads
curl http://localhost:8000/api/v1/jobs/<job_id>
curl http://localhost:8000/api/v1/jobs/<job_id>/result
```

---

## Viewing the database

**Terminal:**

```powershell
docker compose exec postgres psql -U qualiflow -d qualiflow
```

```sql
\dt
SELECT id, supplier_name, status, created_at FROM analysis_runs;
SELECT id, status, original_filename FROM jobs;
```

**GUI:** Connect DBeaver or pgAdmin to `localhost:5432` with user `qualiflow`, password `qualiflow`, database `qualiflow`.

**PDF / JSON files:** MinIO console at http://localhost:9001 → bucket `qualiflow`.

---

## Docker commands

```powershell
docker compose ps                  # status
docker compose logs -f api worker  # live logs
docker compose down                # stop (data volumes kept)
docker compose down -v             # stop + delete DB/MinIO volumes (full reset)
docker compose up --build -d       # rebuild after code changes
```

---

## Local development (without Docker)

Requires PostgreSQL, Redis, and MinIO running locally (or via Docker for infra only):

```powershell
uv sync
alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
uv run python -m app.workers.run_worker
```

Use `.env` with `localhost` URLs as in `.env.example`.

---

## Tests

```powershell
uv run pytest tests/ -q
```

Offline by default (`EXTRACTION_PROVIDER=mock`). Optional live Bedrock smoke:

```powershell
$env:QUALIFLOW_BEDROCK_LIVE=1
uv run pytest tests/test_bedrock_live.py -m bedrock_live
```

Safety-critical subset (see `AGENTS.md`):

```powershell
python -m pytest tests/test_review_policy.py tests/test_extraction_finalizer.py tests/test_conservative_decision_confidence.py tests/test_auto_accept_safety_regression.py -q
```

---

## Project layout

```text
app/                  FastAPI app, services, workers
db/                   SQLAlchemy models
alembic/              Database migrations
frontend/             React UI
config/               Settings
tests/                Unit and regression tests
scripts/              Utilities (start.ps1, eval scripts)
docker-compose.yml    Full local stack
```

---

## Safety model

QualiFlow follows a conservative review policy:

- Never auto-accept `severe_scan` documents
- Never auto-accept missing/unverified traceability identifiers
- Never auto-accept uncertain table alignment
- Invalid or unusable model output fails toward review (`extraction_schema_invalid`)
- The LLM extracts evidence only; it does not make the final conformity decision
- `auto_accept` requires structured `auto_accept_evidence`

See `AGENTS.md` for agent/developer safety rules.

---

## Known limitations

- Academic research prototype, not production SaaS
- Deterministic validation limited to supported grade/spec families
- Degraded scans may require human review
- Final certification approval remains a human responsibility
