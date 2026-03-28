import os
import sys
from contextlib import asynccontextmanager

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from app.core.config import settings
from app.db.session import engine, Base, SessionLocal, get_db
from app.db.models import Job
from sqlalchemy import desc
from sqlalchemy.orm import Session

# ─── Router Imports (Fixed & Optimized) ───
from app.routers import (
    streaming, advisory, monitoring, jobs, auth, gate, forecast, sentinel, red_team, reports,
    audit, behavior, init_scan, preflight, drift, performance, fairness, llm_eval, governance, 
    enterprise, policies, history, alerts, ci,
    model_registry, datasets, experiments, explainability, data_quality, deployments, predictions
)

# ─── Lifespan Management ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup Validation
    required = ["DATABASE_URL", "SECRET_KEY", "MINIO_ENDPOINT"]
    missing = [k for k in required if not os.getenv(k)]
    if missing and not settings.DEBUG:
        # In DEBUG mode we might fallback to SQLite, so only raise in non-debug or if strictly required
        from structlog import get_logger
        get_logger().error("startup_validation_failed", missing_vars=missing)
        # raise RuntimeError(f"Missing required env vars: {missing}")

    # 2. Database Initialization
    Base.metadata.create_all(bind=engine)

    # 3. Object Storage Initialization
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

# ─── CORS Configuration ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS if settings.DEBUG else [o for o in settings.BACKEND_CORS_ORIGINS if "localhost" not in o],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"] if not settings.DEBUG else ["*"],
    allow_headers=["Content-Type", "Authorization"] if not settings.DEBUG else ["*"],
)

# ─── Exception Handling ───
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    if settings.DEBUG:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={"detail": str(exc), "traceback": traceback.format_exc()}
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please contact the administrator."}
    )

# ─── Core Health & Info Endpoints ───
@app.get("/")
async def root():
    return {
        "status": "running", 
        "service": "ML Guard API", 
        "version": settings.APP_VERSION, 
        "architecture": "enterprise-ai-governance"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}

@app.get("/health/worker")
async def health_worker():
    """Verify Celery worker connectivity by inspecting active nodes."""
    from app.core.celery_app import celery_app
    try:
        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active()
        worker_online = active is not None
        
        # Cross-reference with DB for job processing stats
        db = SessionLocal()
        try:
            last_job = db.query(Job).order_by(desc(Job.created_at)).first()
            return {
                "status": "healthy" if worker_online else "degraded",
                "worker_online": worker_online,
                "last_job_processed_at": last_job.created_at if last_job else None,
                "total_jobs_tracked": db.query(Job).count()
            }
        finally:
            db.close()
    except Exception as e:
        return {"status": "error", "detail": str(e), "worker_online": False}

# ─── API Router Registration ───
# Governance & Analysis
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
    uvicorn.run("app.main:app", host="0.127.0.0.1", port=8000, reload=settings.DEBUG)
