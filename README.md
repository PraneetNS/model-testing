# ML Guard v7.2 — The Enterprise AI Governance Platform

[![ML Guard](https://img.shields.io/badge/Version-7.2.0%20(Agentic)-blueviolet)](https://github.com/FireFlink/ml_guard)
[![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20PostgreSQL%20%7C%20Next.js-blue)](https://github.com/FireFlink/ml_guard)

**ML Guard** is a state-of-the-art AI governance and observability platform designed to bring accountability, security, and behavioral compliance to machine learning models. Beyond simple monitoring, ML Guard implements a **Governance-as-Code** philosophy through its novel Behavioral Contract system, integrating directly into enterprise CI/CD workflows and production runtimes.

---

## 🚀 The Feature Universe (v7.2)

ML Guard is designed to evaluate, monitor, and enforce policy across the entire ML lifecycle. From local development to production pipelines, ML Guard protects your enterprise.

### 1. CI/CD Governance Gates (The Model Sentinel)
No bad models ever make it to production.
- **Automated Gates**: A CLI tool (`ml_guard_ci.py`) that executes a multipart submission to the backend API during your GitHub Actions pipeline.
- **Strict Mode Enforcement**: Models must meet scoring criteria (e.g. `> 60/100`) and receive a `CERTIFIED` or `CONDITIONAL` verdict to pass.
- **Job Asynchrony**: Background Celery workers execute the checks without blocking CI, responding via poll hooks with deterministic evaluation results.

### 2. Model Behavior Contracts (Novel)
Define behavioral promises that your model must keep. Validated in real-time during every prediction via our Python SDK.
- **Promise Types**: Output confidence ranges, latency SLAs, probabilistic thresholds, and fairness parity bounds.
- **Breach Management**: Automated recording of violations classified by severity (CRITICAL, HIGH, LOW).
- **Governance Impact**: Real-world contract breaches directly and automatically penalize the model's live governance score.

### 3. Certified Governance Report Cards
Bring peace of mind to stakeholders with audit-ready documentation.
- **Automated Generation**: Produces professional PDF report cards for any model version at an audited timestamp.
- **Executive Summary**: Contains the ultimate verdict (CERTIFIED, CONDITIONAL, FAILED) alongside multi-dimensional radar charts.
- **Tamper-Proof & Cryptographic**: Every certificate includes a unique SHA-256 hash mathematically locking the state of the model when the certificate was issued.

### 4. Real-Time Drift Sentinel
High-performance sliding-window drift detection for production models.
- **Statistical Distance Calculation**: Monitors Population Stability Index (PSI), Kolmogorov-Smirnov (KS-Test), and Jensen-Shannon Divergence.
- **Asynchronous Telemetry Ingestion**: Handles high-throughput production request logging using fire-and-forget ingestion powered by Celery queues.
- **Feature Level Granularity**: Analyzes concept drift on target labels, and data drift on specific input features.

### 5. Multi-dimensional Composite Scoring Engine
Calculates the "health" of an AI model using heuristic weighting:
- **Performance**: Statistical accuracy metrics (F1, Accuracy, Brier Score, Precision, Recall).
- **Security & Vulnerability**: Protection against Data Poisoning, Data Extraction, and Membership Inference attacks. 
- **Fairness & Bias**: Parity assessment across sensitive demographic features.
- **Live Decay**: Automatically rots a deployed model's governance score based on real-time SLA/contract breaches.

### 6. LLM Security & Red Teaming (Generative AI)
A dedicated suite tailored specifically to Large Language Model governance.
- **Heuristic Toxicity & Hallucination Guardrails**: Analyzes model outputs systematically.
- **Adversarial Resiliency**: Active jailbreak vector detection and prompt-injection mitigations.
- **PII Leakage Scanning**: Guarantees generative assets do not exfiltrate sensitive data.

---

## 🏗️ System Architecture

ML Guard operates on an asynchronous microservices architecture meant to run at scale:

- **Core ML Engine (`ml_guard/core`)**: Pure Python libraries housing the deterministic statistics, risk algorithms, and heuristics.
- **Backend API (`ml_guard/backend`)**: Built with **FastAPI** utilizing **SQLAlchemy** for asynchronous persistence into **PostgreSQL**.
- **Asynchronous Workers**: Redis-backed **Celery** tasks specifically tasked to compute heavy algorithms (Fairness/Drift computations) outside of the HTTP cycle.
- **Dashboard (`ml_guard/frontend`)**: A blazing fast **Next.js 14** application styled with Tailwind and Shadcn UI.
- **Data & Artifact Layer**: S3-compatible generic storage (MinIO interface ready).

---

## 📖 Module Catalog

The backend exposes a highly separated Domain Driven structure comprising 16 top-level routers:

- `audit.py` - Initiates multipart scanning and metadata extraction for uploaded models.
- `gate.py` - The fast, synchronous execution path for CI pipeline verdicts.
- `governance.py` - Retrieves complex scoring distributions across the platform.
- `contracts.py` - Evaluates strict behavioral constraints and prediction bounds.
- `report.py` & `history.py` - Historical audit trails and PDF certificate generation.
- `sentinel.py` / `alerts.py` - Streaming prediction bounds check and notification systems.
- `policies.py` - Configurable enterprise-wide compliance limits.
- `drift.py`, `performance.py`, `fairness.py`, `llm_eval.py` - Granular sub-domain analysis endpoints.
- `ingest.py` & `auth.py` - Security and high-volume payload consumption.

---

## 🛠️ Complete Setup Guide

### 1. Requirements

- Node.js 20+
- Python 3.11+
- Redis (For Celery Task Queues)

### 2. Backend & Core Dependencies

```bash
cd ml_guard/backend
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
pip install celery redis
```

Set up your `.env` file (you can copy `.env.example`):
```ini
SECRET_KEY=your_secure_hash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mlguard
# Redis connection for Celery
REDIS_URL=redis://localhost:6379/0
```

Start the primary API:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Celery Worker (In a new terminal, `ml_guard/backend`):
```bash
# Activate your venv first
celery -A app.core.celery_app worker --loglevel=info -P solo
```
*(Note for Windows users: `-P solo` is required for Celery execution on Windows)*

### 3. Frontend Dashboard

```bash
cd ml_guard/frontend
npm install
npm run dev
```

The ML Guard dashboard will be accessible via `http://localhost:3000`.

---

## 🛡️ Setting up the Github Actions CI Gate

Ensure your deployment stays un-corrupted by integrating the governance pipeline directly into your CI!
Read the specific markdown instructions at [`.github/SETUP.md`](./.github/SETUP.md).

**Local Test Drive of the CI Tool**:
Ensure your `uvicorn` instance is running, then execute from the repository root:
```bash
python .github/scripts/ml_guard_ci.py \
  --api-url http://127.0.0.1:8000 \
  --api-key mlg_1Ai7zfmfsB_GLaoNuKjOOopFh12xLzGy7SDqh7Kho1U \
  --model-name TestChurnModel-v1 \
  --model-path ml_guard/backend/fair_loan_model.pkl \
  --data-path ml_guard/backend/fair_loan_test.csv \
  --label-col target \
  --min-score 60
```
This script will:
1. Run a health check against the local API.
2. Submit your model and dataset as a multi-part payload.
3. Determine job status and asynchronously poll Celery for completion.
4. Obtain the `model_id` via heuristic lookup.
5. Provide you an automated Score & Verdict.

---

## 💡 Governance Philosophy
ML Guard transforms subjective "AI Ethics" into objective, measurable, and enforceable technical contracts. By bridging the gap between data science and compliance, we ensure that AI remains a secure, predictable, and fair asset for the enterprise.

---
© 2026 FireFlink ML Research. Proprietary & Confidential.
