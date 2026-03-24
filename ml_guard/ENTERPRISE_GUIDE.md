# ML Guard: Enterprise Governance Platform 🚀

Welcome to the upgraded **ML Guard Enterprise**. The platform has been transformed from a prototype into a production-ready, industrial-grade governance solution for Mission-Critical AI.

## 🏗️ New Enterprise Architecture
The system now follows a **Domain-Driven Design (DDD)** pattern with a hardened infrastructure:
- **Multi-Tenant SaaS Core**: Secure data isolation and tenant-based project management.
- **Enterprise Persistence**: PostgreSQL for immutable audit trails and time-series drift logs.
- **Asynchronous Engine**: Redis-backed Celery workers for heavy model evaluations.
- **Security First**: JWT Authentication, RBAC (Admin/Auditor/Developer), and Rate Limiting.

## ✨ Key Features
1.  **Scriptless ML Testing Framework**: JUnit-style structured tests with PASS/FAIL results, severity levels, and automated remediation.
2.  **Regression Suite**: Automated Model v1 vs Model v2 comparison to block performance degradation.
3.  **Industrial Quality Gate**: Enforce strict passing thresholds with a weighted Quality Index.
4.  **Audit Trail & Compliance**: Immutable history of governance scans for legal and regulatory compliance.
5.  **Live Drift Telemetry**: Persistent monitoring of feature stability with automatic alert triggering.
6.  **ML Guard CLI & CI/CD**: Integrate audits directly into GitHub Actions or GitLab CI with our standalone CLI tool.
7.  **Rich Reporting**: Automated generation of Human-Readable HTML and Machine-Readable JSON audit reports.

## 🚀 Getting Started (Production Mode)

### 1. Requirements
Ensure you have **Docker** and **Docker Compose** installed.

### 2. Launch Stack
Run the following command to start PostgreSQL, Redis, Backend, Worker, and Frontend:
```bash
docker-compose up --build
```

### 3. Access the Platform
- **Frontend Dashboard**: `http://localhost:5173`
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Monitoring (Prometheus)**: `http://localhost:9090` (Optional)

## 🛠️ Developer Setup (Local)
If you prefer running without Docker:
1.  **Backend**:
    - Install dependencies: `pip install -r backend/requirements.txt`
    - Start Redis & Postgres (local or Docker sidecar)
    - Run: `uvicorn app.main:app --reload`
2.  **Frontend**:
    - `npm install`
    - `npm run dev`

## 🔒 Security Configuration
All secrets are managed via `.env` files. Ensure you update `SECRET_KEY` for production deployments.

---
**Fireflink - Empowering Responsible AI Development**
