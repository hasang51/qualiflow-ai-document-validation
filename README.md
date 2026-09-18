# QualiFlow

QualiFlow is a human-in-the-loop system that extracts structured fields from Mill Test Certificates (MTCs), validates them with deterministic rules, and routes uncertain cases to review instead of silently approving them.

This is an academic / portfolio prototype, **not** a production certification product.

---

## Why this project exists

Industrial buyers still check mill certificates by hand. The costly failure mode is not “the model missed a field.” It is **silently accepting** a wrong heat number, grade, date, or mechanical value.

QualiFlow is built around that failure mode: when evidence is incomplete, ambiguous, or unverified, the document goes to a human. Auto-accept is a narrow, auditable exception — not the default.

---

## Architecture

```text
Browser (React)
        │
        ▼
API (FastAPI)
        │  POST /api/v1/uploads
        ▼
Redis + RQ worker
        │
        ├── Amazon Bedrock extraction (or mock in CI)
        ├── deterministic validation + review policy
        └── persist analysis
        ▼
PostgreSQL          MinIO / local object storage
```

| Layer | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Worker | Redis + RQ |
| Database | PostgreSQL 16 (Alembic) |
| Object storage | MinIO or local disk |
| Frontend | React, TypeScript, Vite, Tailwind |
| Extraction | Amazon Bedrock Gemma 4 26B-A4B (`eu-central-1`) or `MockExtractionProvider` in CI |

The LLM is an **evidence extractor**. It does not decide conformity.

1. **Stage A/B extraction** returns candidate fields, tables, and confidences.
2. **Deterministic services** parse identifiers, map mechanical rows, resolve grades/specs, and compare values to allowed ranges.
3. **Review policy** consumes structured blocking reasons. Confidence-only reasons cannot override a real block.
4. **`auto_accept` requires** an explainable `auto_accept_evidence` object. The finalizer must not strip blocking reasons to force acceptance.

Safety-critical code lives in `app/services/review_policy.py`, `extraction_finalizer.py`, `traceability.py`, and `validator.py`. Agent/developer rules are in `AGENTS.md`.

---

## Safety model

Hard rules (also regression-tested):

- Never auto-accept `severe_scan` documents.
- Never auto-accept if grade is missing, unresolved, ambiguous, or unmapped.
- Never auto-accept if heat number or another accepted traceability identifier is missing, unverified, OCR-uncertain, or conflicting.
- Never auto-accept if row/table alignment is uncertain.
- Invalid model JSON becomes `extraction_schema_invalid` and goes to review.

**Extraction miss** = a verified gold field the model got wrong or left empty.  
**Fail-closed review** = live policy sent the document to a human rather than auto-accepting. That is a safety outcome, not an unsafe accept.  
**Unsafe auto-accept** = verified gold requires review, and the prediction auto-accepted anyway.

When in doubt, the system escalates. Auto-accept is a narrow exception that must be explainable.

---

## Live Bedrock Evaluation

Nineteen real Mill Test Certificate PDFs were processed through the live Amazon Bedrock pipeline (`google.gemma-4-26b-a4b` in `eu-central-1`) with the existing adapter and fresh predictions. Private PDFs, supplier names, identifiers, filenames, and certificate contents are **not committed** and are not listed here.

**Human-verified ground truth covers all 19 documents (n = 19).** Candidate annotations were not treated as truth. Ambiguous fields were left null. Accuracy is scored only on those verified files.

Quality mix of the evaluation set:

- 4 clean
- 4 blurry
- 4 low-quality
- 4 different-layout
- 3 watermarked

String fields are exact-match after lowercase/whitespace normalisation. Numeric fields use a 1.0 tolerance. Missing predictions count as incorrect when verified gold has a value.

| Metric | Value |
| --- | --- |
| Documents | 19 real MTC PDFs |
| Human-verified ground truth | 19 / 19 |
| Processed / failed / skipped | 19 / 0 / 0 |
| Field accuracy | 44.1% |
| Critical-field accuracy | 57.5% |
| Traceability exact-match | 7/15 (46.7%) |
| Processing-decision accuracy | 73.7% |
| Review rate | 19/19 (100%) |
| Unsafe auto-accept | 0 / 0% |
| Schema failures | 0 |
| Latency p50 / p95 | 26.1 s / 43.9 s |
| Estimated cost | $0.0305 total, $0.0016 / document |

**All 19 documents were routed to review.** There were no auto-accepts. **0 unsafe auto-accepts** means fail-closed safety was preserved: extraction errors did not become silent approvals.

**100% review rate also means automation is currently conservative.** The system is not claiming production-ready auto-accept volume. Processing-decision accuracy of 73.7% is consistent with a policy that prefers review over release.

Main extraction weaknesses on this set:

- **Grade** resolution / mapping
- **Traceability** typing (identifier kind and exact match)
- **Dates**
- **Mechanical-table mapping** (row/column alignment)

Those misses stayed in review. They did not become auto-accepts.

---

## What the evaluation proves

On this 19-document real-PDF set, the live pipeline ran end to end: profiling → routing → preprocessing → Bedrock extraction → deterministic validation → review policy → scoring against human-verified ground truth.

- Schema-valid outputs for all 19 documents (0 schema failures).
- Extraction misses were contained by review rather than auto-accept.
- Fail-closed behaviour held: **0 unsafe auto-accepts**.

---

## What it does NOT prove

- It does **not** prove 100% extraction accuracy (field accuracy here is 44.1%).
- It is **not** production certification or a substitute for mill / material certification.
- **n = 19** is a small academic evaluation set. Results should **not** be generalized beyond this dataset.
- 100% review is evidence of conservative gating, not of a mature auto-accept product.
- Estimated Bedrock cost is not an AWS invoice.

Final material certification remains a human responsibility.

---

## Engineering highlights

- **Split of concerns:** multimodal extraction is separated from deterministic validation and review gating so model variability cannot silently certify a document.
- **Fail-closed policy:** blocking reasons (grade, traceability, table alignment, scan quality, invalid schema) cannot be overridden by confidence-only signals.
- **Auditable auto-accept path:** `auto_accept` requires structured `auto_accept_evidence`; the finalizer must not strip real blocks.
- **Regression safety net:** review policy, finalizer, traceability, conservative confidence, and auto-accept safety tests run in CI with `EXTRACTION_PROVIDER=mock` (no live Bedrock).
- **Measured live behaviour:** one verified 19-PDF Bedrock run with explicit accuracy, latency, cost, and unsafe-accept metrics — reported without private certificate contents.

---

## Project leadership / my contributions

This is a **two-person Computer Engineering graduation project**. Work was collaborative with one teammate.

I led product definition, system architecture, backend/API, extraction plus deterministic validation, review policy, evaluation methodology, and deployment setup. Architecture, acceptance criteria, safety constraints, test design, evaluation interpretation, and what may be claimed publicly remained human-owned.

I do not claim to have built the project alone, and I do not claim that the 19-PDF run is production qualification.

### AI-assisted development

I used AI-assisted coding tools during development, but architecture, acceptance criteria, safety constraints, test design, evaluation methodology, and final engineering decisions remained human-owned and independently verified.

---

## Limitations

- Research / portfolio prototype, not certified production SaaS.
- Small academic evaluation set (**n = 19**). Do not generalize these numbers to other mills, layouts, or languages.
- Not a claim of 100% real-world accuracy.
- Deterministic validation covers supported grade/spec families only.
- Degraded scans, watermarks, and unusual layouts usually require a human — and on this run, **every** document did.
- Final material certification remains a human responsibility.
- Estimated Bedrock cost is not an AWS invoice.

---

## Quick start

```powershell
copy .env.example .env
docker compose up --build -d
cd frontend
copy .env.example .env
npm install
npm run dev
```

UI: http://localhost:5173 — API: http://localhost:8000.

For live extraction set `EXTRACTION_PROVIDER=bedrock` and use the default AWS credential chain. Do not put access keys in `.env`. Local tests stay on `EXTRACTION_PROVIDER=mock`.

Windows rasterisation needs Poppler (`POPPLER_PATH`). Encrypted PDFs need the `cryptography` extra pulled in with `uv sync`.

```powershell
uv run pytest tests/ -q
```

---

## API

Upload a PDF and poll the job until analysis is ready:

- `POST /api/v1/uploads`
- `GET /api/v1/jobs/{id}`

The worker runs extraction, deterministic validation, and review policy. Uncertain documents are stored for human review rather than auto-accepted.

---

## CI and tests

GitHub Actions (`.github/workflows/ci.yml`) runs:

- Ruff and Pyright
- A deterministic **safety subset**: review policy, validator, finalizer, traceability, conservative confidence, auto-accept regressions, unsafe auto-accept rate
- Full backend pytest with coverage (`fail_under=65`)
- Frontend lint, unit tests, and production build

Offline tests use `EXTRACTION_PROVIDER=mock`. They do **not** call Bedrock.

The `data/gold/m2/` set is a **legacy synthetic regression fixture** (6 generated certificates). It is useful for matcher/eval plumbing and CI gates. It is **not** real-world accuracy.

Optional live smoke:

```powershell
$env:QUALIFLOW_BEDROCK_LIVE=1
uv run pytest tests/test_bedrock_live.py -m bedrock_live
```
