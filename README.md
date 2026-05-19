# ML Guard — AI Governance & Observability

This repository contains ML Guard: an enterprise-focused AI governance platform combining a Python backend, a Next.js dashboard, a Python SDK, and asynchronous workers for heavy ML analysis (explainability, drift, robustness, and governance policy enforcement).

Highlights
- Modern FastAPI backend with async SQLAlchemy and OpenAPI docs
- Next.js dashboard for operations and visualization
- Celery + Redis workers for long-running explainability and audit tasks
- S3-compatible storage using MinIO for artifacts
- Designed for local development and production (Postgres/Neon, Redis, MinIO)

Repository layout (top-level)
- [ml_guard/backend](ml_guard/backend) — FastAPI backend and core governance engine
- [ml_guard/frontend](ml_guard/frontend) — Next.js dashboard (App Router)
- [ml_guard/sdk](ml_guard/sdk) and [sdk](sdk) — Python SDK(s) for in-app enforcement and programmatic usage
- [ml_guard/docker](ml_guard/docker) — Docker compose and deployment helpers
- docs/ — architecture diagrams and assets

What I found (detected tech and where it lives)
- Backend: FastAPI (see [ml_guard/backend/pyproject.toml](ml_guard/backend/pyproject.toml)) with `uvicorn` and async SQL tooling.
- Workers: Celery with Redis (see [ml_guard/backend/.env.example](ml_guard/backend/.env.example) and `.env` templates).
- Frontend: Next.js app with a Dockerfile in [ml_guard/frontend/Dockerfile](ml_guard/frontend/Dockerfile) — Node.js 20+ recommended.
- Storage: MinIO (Dockerfile: `Dockerfile.minio`) for S3-compatible artifacts.
- Persistence: SQLite for local dev and PostgreSQL/Neon for production (see [\.env.example](.env.example) and [ml_guard/.env.production.template](ml_guard/.env.production.template)).
- Explainability / ML libs: SHAP, fairlearn, scikit-learn, XGBoost (declared in `pyproject.toml` files).

Architecture (short)
- The Next.js dashboard talks to the FastAPI backend (REST + WebSockets for telemetry). The backend persists metadata to SQL, enqueues heavy tasks to Celery (Redis broker), and stores artifacts in MinIO/S3. Celery workers run explainability (SHAP), fairness scans, drift calculations (PSI/KS/MMD), and security audits.

Quick start — local development
1) Requirements
- Python 3.10+ (3.11 recommended)
- Node.js 20+
- Redis, PostgreSQL (or SQLite for local), MinIO (or S3)

2) Backend (local)
```powershell
cd ml_guard/backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# run the API
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment files and examples live in [ml_guard](ml_guard) — copy `.env.example` or use [ml_guard/.env.production.template](ml_guard/.env.production.template) for production values.

3) Celery worker (in a new terminal)
```powershell
cd ml_guard/backend
.\.venv\Scripts\activate
celery -A app.core.celery_app worker --loglevel=info
```

4) Frontend (dev)
```powershell
cd ml_guard/frontend
npm install
npm run dev
# visit http://localhost:3000
```

Docker / Local compose
- `ml_guard/docker-compose.yml` defines frontend, backend, worker, database, Redis, and MinIO for a single-command local environment. Use Docker Compose in that folder for quick reproducible dev clusters.

CI and tests
- GitHub Actions workflows lint and run backend/frontend tests. See `.github/workflows` for CI jobs (FastAPI tests, frontend lint, and governance gate runs).

Where to look next (useful entry points)
- Backend main app: [ml_guard/backend/app/main.py](ml_guard/backend/app/main.py)
- Celery setup: [ml_guard/backend/app/core/celery_app.py](ml_guard/backend/app/core/celery_app.py)
- Frontend Dockerfile: [ml_guard/frontend/Dockerfile](ml_guard/frontend/Dockerfile)
- SDK packaging: [sdk/pyproject.toml](sdk/pyproject.toml)

Notes & recommendations
- Local dev commonly uses SQLite; switch to a Postgres-compatible DB (Neon/RDS) in production (templates provided).
- Celery can be run in `solo` mode for local debugging but should use prefork/uvloop worker pools in production where appropriate.
- The project contains several architecture docs and explainers in `docs/` and `ml_guard/ARCHITECTURE.md` that detail design decisions.

Contributing
- See `CONTRIBUTING.md` if present, otherwise open issues or pull requests. For gated changes, add tests to `ml_guard/backend/app/tests` and frontend checks under `ml_guard/frontend`.

License
- The repository contains mixed licensing info across modules; check individual package manifests (for example, [ml_guard/backend/pyproject.toml](ml_guard/backend/pyproject.toml) and [sdk/pyproject.toml](sdk/pyproject.toml)) for licensing details.

Questions or next steps
- I updated this README to reflect detected structure, tech stack, and a concise quickstart. Do you want me to:
  - Add a short troubleshooting section (common errors and fixes)?
  - Add a `docker-compose` quick command example in this README?
  - Open a PR and run the CI checks locally?

---
Updated: automatic analysis and README consolidation.
