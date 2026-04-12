import os
import sys
from contextlib import asynccontextmanager

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from sqlalchemy import desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.db.session import engine, Base, AsyncSessionLocal, get_db
from app.db.models import Job

# ── Core Analysis Routers ──────────────────────────
from app.routers import audit
from app.routers import behavior
from app.routers import init_scan
from app.routers import preflight
from app.routers import drift
from app.routers import performance
from app.routers import fairness
from app.routers import llm_eval
from app.routers import rag_eval
from app.routers import governance

# ── Enterprise Platform Routers ────────────────────
from app.routers import enterprise
from app.routers import policies
from app.routers import history
from app.routers import alerts
from app.routers import ci
from app.routers import reports
from app.routers import ingest
from app.routers import observe

# ── Infrastructure Routers ─────────────────────────
from app.routers import jobs
from app.routers import auth
from app.routers import gate
from app.routers import forecast
from app.routers import sentinel
from app.routers import red_team
from app.routers import streaming
from app.routers import advisory
from app.routers import monitoring

# ── Lifecycle Extension Routers ────────────────────
from app.routers import model_registry
from app.routers import datasets
from app.routers import experiments
from app.routers import explainability
from app.routers import data_quality
from app.routers import deployments
from app.routers import predictions
from app.routers import contracts

# ── API/Tasks Router ───────────────────────────────
from app.api.routers import tasks

# ─── Lifespan Management ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    # FIX 5: Startup env validation
    if not settings.SECRET_KEY or "CHANGE_ME" in settings.SECRET_KEY:
        raise RuntimeError(
            "STARTUP FAILED — SECRET_KEY not configured. "
            "Set it in your environment or .env file."
        )

    # Database Initialization
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
    allow_origins=getattr(
        settings, 
        "ALLOWED_ORIGINS", 
        ["http://localhost:3000"]
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# FIX 2: Debug Traceback Gating
@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
):
    if getattr(settings, "DEBUG", False):
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

@app.get("/api/health/db")
async def health_database(db: AsyncSession = Depends(get_db)):
    """Heartbeat check for PostgreSQL async pool connectivity."""
    import time
    from app.db.session import engine
    try:
        start_time = time.time()
        await db.execute(text("SELECT 1"))
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        pool = getattr(engine, "pool", None)
        pool_size = getattr(pool, "size", 0) if pool else 0
        checked_out = getattr(pool, "checkedout", 0) if pool else 0
        
        return {
            "status": "ok", 
            "latency_ms": latency_ms,
            "pool_size": pool_size,
            "checked_out_connections": checked_out
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.get("/health/storage")
async def health_storage():
    """Heartbeat check for MinIO / S3 object storage."""
    from app.services.storage_service import _get_s3_client
    try:
        client = _get_s3_client()
        client.list_buckets()
        return {"status": "ok", "message": "Object storage connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# FIX 3: Real Celery health check
@app.get("/health/worker")
async def health_worker():
    try:
        from app.core.celery_app import celery_app
        inspect = celery_app.control.inspect(
            timeout=2.0
        )
        active = inspect.active()
        online = active is not None
        count = len(active) if active else 0
    except Exception:
        online = False
        count = 0
    return {
        "status": "healthy" if online else "offline",
        "workers_online": online,
        "worker_count": count
    }

# ─── API Router Registration ───

# ── Core Analysis ──────────────────────────────────
app.include_router(audit.router,
    prefix="/api/v1", tags=["audit"])
app.include_router(behavior.router,
    prefix="/api/v1", tags=["behavior"])
app.include_router(init_scan.router,
    prefix="/api/v1/scan", tags=["init"])
app.include_router(preflight.router,
    prefix="/api/v1", tags=["preflight"])
app.include_router(drift.router,
    prefix="/api/v1", tags=["drift"])
app.include_router(performance.router,
    prefix="/api/v1", tags=["performance"])
app.include_router(fairness.router,
    prefix="/api/v1", tags=["fairness"])
app.include_router(llm_eval.router,
    prefix="/api/v1", tags=["llm"])
app.include_router(governance.router,
    prefix="/api/v1", tags=["governance"])
app.include_router(rag_eval.router,
    prefix="/api/v1", tags=["rag_eval"])

# ── Enterprise Platform ────────────────────────────
app.include_router(enterprise.router,
    prefix="/api/v1", tags=["enterprise"])
app.include_router(policies.router,
    prefix="/api/v1", tags=["policies"])
app.include_router(history.router,
    prefix="/api/v1", tags=["history"])
app.include_router(alerts.router,
    prefix="/api/v1", tags=["alerts"])
app.include_router(ci.router,
    prefix="/api/v1", tags=["ci"])
app.include_router(reports.router,
    prefix="/api/v1", tags=["reports"])
app.include_router(ingest.router,
    prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(observe.router,
    prefix="/api/v1", tags=["observe"])

# ── Infrastructure ─────────────────────────────────
app.include_router(jobs.router,
    prefix="/api/v1", tags=["jobs"])
app.include_router(auth.router,
    prefix="/api/v1", tags=["auth"])
app.include_router(gate.router,
    prefix="/api/v1/gate", tags=["gate"])
app.include_router(forecast.router,
    prefix="/api/v1/forecast", tags=["forecast"])
app.include_router(sentinel.router,
    prefix="/api/v1/sentinel", tags=["sentinel"])
app.include_router(red_team.router,
    prefix="/api/v1/redteam", tags=["red_team"])
app.include_router(streaming.router,
    prefix="/api/v1", tags=["streaming"])
app.include_router(advisory.router,
    prefix="/api/v1", tags=["advisory"])
app.include_router(monitoring.router,
    prefix="/api/v1", tags=["monitoring"])

# ── Lifecycle Extensions ───────────────────────────
app.include_router(model_registry.router,
    prefix="/api/v1", tags=["model-registry"])
app.include_router(datasets.router,
    prefix="/api/v1", tags=["datasets"])
app.include_router(experiments.router,
    prefix="/api/v1", tags=["experiments"])
app.include_router(explainability.router,
    prefix="/api/v1", tags=["explainability"])
app.include_router(data_quality.router,
    prefix="/api/v1", tags=["data-quality"])
app.include_router(deployments.router,
    prefix="/api/v1", tags=["deployments"])
app.include_router(predictions.router,
    prefix="/api/v1", tags=["predictions"])
app.include_router(contracts.router,
    prefix="/api/v1", tags=["contracts"])

from app.api.routers import lineage
from app.routers import plugins
from app.routers import notifications
app.include_router(lineage.router)
app.include_router(tasks.router)
app.include_router(plugins.router, prefix="/api/plugins", tags=["plugins"])
app.include_router(notifications.router, prefix="/api/v1", tags=["notifications"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=getattr(settings, "DEBUG", False))
