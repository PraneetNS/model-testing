import os
import sys
from contextlib import asynccontextmanager

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from app.core.config import settings
from app.db.session import engine, Base, SessionLocal

# ─── Grouped Router Imports ───
from app.routers import (
    # ── Group 1: Already imported (working) ──
    streaming,
    advisory,
    monitoring,
    jobs,
    auth,
    gate,
    forecast,
    sentinel,
    red_team,
    # ── Group 2: Lifecycle extension (working) ──
    model_registry,
    datasets,
    experiments,
    explainability,
    data_quality,
    deployments,
    predictions,
    # ── Group 3: Core analysis ──
    audit,
    behavior,
    init_scan,
    preflight,
    drift,
    performance,
    fairness,
    llm_eval,
    governance,
    # ── Group 4: Enterprise ──
    enterprise,
    policies,
    history,
    alerts,
    ci,
    # ── Reports ──
    reports
)

# ─── Lifespan Management ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    # FIX 5: Startup env validation
    required_env = [
        "DATABASE_URL", 
        "SECRET_KEY",
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"STARTUP FAILED — missing required "
            f"environment variables: {missing}\n"
            f"Check your .env file."
        )

    # Database Initialization
    Base.metadata.create_all(bind=engine)

    # Object Storage Initialization (Optional/Warning only)
    from app.services.storage_service import _get_s3_client, _ensure_bucket_exists
    try:
        client = _get_s3_client()
        _ensure_bucket_exists(client)
        structlog.get_logger().info("storage_initialized", bucket=settings.MINIO_BUCKET)
    except Exception as e:
        structlog.get_logger().warning("storage_init_failed", error=str(e))

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise AI Governance Platform v7.2 — CI/CD Integration, Fairness, LLM Guard",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# FIX 4: CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# FIX 2: Debug Traceback Gating
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if settings.DEBUG:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={
                "detail": str(exc), 
                "traceback": traceback.format_exc()
            }
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ─── Core Health & Info Endpoints ───
@app.get("/")
async def root():
    # FIX 1: Version consistency
    return {
        "status": "running", 
        "service": "ML Guard API", 
        "version": settings.APP_VERSION, 
        "architecture": "enterprise-ai-governance"
    }

@app.get("/health")
async def health():
    # FIX 1: Version consistency
    return {"status": "ok", "version": settings.APP_VERSION}

# FIX 3: Real Celery health check
@app.get("/health/worker")
async def health_worker():
    """Verify Celery worker connectivity and task processing status."""
    try:
        from app.core.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active()
        online = active is not None
        worker_count = len(active) if active else 0
    except Exception as e:
        online = False
        worker_count = 0
        
    return {
        "status": "healthy" if online else "offline",
        "workers_online": online,
        "worker_count": worker_count,
    }

# ─── API Router Registration ───

# Core Analysis
app.include_router(audit.router,       prefix="/api/v1", tags=["audit"])
app.include_router(behavior.router,    prefix="/api/v1", tags=["behavior"])
app.include_router(init_scan.router,   prefix="/api/v1/scan", tags=["init"])
app.include_router(preflight.router,   prefix="/api/v1", tags=["preflight"])
app.include_router(drift.router,       prefix="/api/v1", tags=["drift"])
app.include_router(performance.router, prefix="/api/v1", tags=["performance"])
app.include_router(fairness.router,    prefix="/api/v1", tags=["fairness"])
app.include_router(llm_eval.router,    prefix="/api/v1", tags=["llm"])
app.include_router(governance.router,  prefix="/api/v1", tags=["governance"])

# Operations & Platform
app.include_router(model_registry.router, prefix="/api/v1", tags=["model-registry"])
app.include_router(datasets.router,       prefix="/api/v1", tags=["datasets"])
app.include_router(enterprise.router,     prefix="/api/v1", tags=["enterprise"])
app.include_router(policies.router,       prefix="/api/v1", tags=["policies"])
app.include_router(jobs.router,           prefix="/api/v1", tags=["jobs"])
app.include_router(auth.router,           prefix="/api/v1", tags=["auth"])
app.include_router(history.router,        prefix="/api/v1", tags=["history"])
app.include_router(alerts.router,         prefix="/api/v1", tags=["alerts"])
app.include_router(ci.router,             prefix="/api/v1", tags=["ci"])

# Specialized Modules (v7.2)
app.include_router(gate.router,        prefix="/api/v1/gate", tags=["gate"])
app.include_router(forecast.router,    prefix="/api/v1/forecast", tags=["forecast"])
app.include_router(sentinel.router,    prefix="/api/v1/sentinel", tags=["sentinel"])
app.include_router(red_team.router,    prefix="/api/v1/redteam", tags=["red_team"])
app.include_router(reports.router,     prefix="/api/v1/reports", tags=["reports"])

# Telemetry & Extension
app.include_router(streaming.router,      prefix="/api/v1", tags=["streaming"])
app.include_router(advisory.router,       prefix="/api/v1", tags=["advisory"])
app.include_router(monitoring.router,     prefix="/api/v1", tags=["monitoring"])
app.include_router(experiments.router,    prefix="/api/v1", tags=["experiments"])
app.include_router(explainability.router, prefix="/api/v1", tags=["explainability"])
app.include_router(data_quality.router,   prefix="/api/v1", tags=["data-quality"])
app.include_router(deployments.router,    prefix="/api/v1", tags=["deployments"])
app.include_router(predictions.router,    prefix="/api/v1", tags=["predictions"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=settings.DEBUG)
