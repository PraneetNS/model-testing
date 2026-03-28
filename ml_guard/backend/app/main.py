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
from app.db.models import Job
from sqlalchemy import desc

# Routers
from app.routers import (
    streaming, advisory, monitoring, jobs, auth, gate, forecast, sentinel, red_team
)

# Lifecycle Extension Routers
from app.routers import (
    model_registry, datasets, experiments,
    explainability, data_quality, deployments,
    predictions,
)

# Lifespan for DB setup and Storage Init
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Database Schema
    Base.metadata.create_all(bind=engine)

    # 2. Initialize Object Storage (MinIO)
    from app.services.storage_service import _get_s3_client, _ensure_bucket_exists
    try:
        client = _get_s3_client()
        _ensure_bucket_exists(client)
        structlog.get_logger().info("storage_initialized", bucket=settings.MINIO_BUCKET)
    except Exception as e:
        structlog.get_logger().warning("storage_init_failed", error=str(e))

    yield

app = FastAPI(
    title="ML Guard Enterprise",
    description="Enterprise AI Governance Platform v7.2 — CI/CD Integration, Fairness, LLM Guard",
    version="7.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "https://ml-guard.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Debug Exception Handler ───
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    import traceback
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": traceback.format_exc()})

# ─── Core Endpoints ───
@app.get("/")
async def root():
    return {"status": "running", "service": "ML Guard API", "version": "7.0.0", "architecture": "enterprise-ai-governance"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "7.0.0"}

@app.get("/health/storage")
async def health_storage():
    """Check Cloudflare R2 object storage connectivity."""
    try:
        from app.services.storage_service import check_storage_health
        return check_storage_health()
    except ImportError:
        return {"status": "not_configured", "provider": "cloudflare_r2", "detail": "storage_service not available"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/health/database")
async def health_database():
    """Check database connectivity (Neon PostgreSQL / SQLite)."""
    db = SessionLocal()
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT 1"))
        row = result.fetchone()
        db_uri = settings.SQLALCHEMY_DATABASE_URI or ""
        if "postgresql" in db_uri:
            db_type = "postgresql"
            provider = "neon" if "neon.tech" in db_uri else "standard"
        elif "sqlite" in db_uri:
            db_type = "sqlite"
            provider = "local"
        else:
            db_type = "unknown"
            provider = "unknown"
        return {
            "status": "healthy",
            "database_type": db_type,
            "provider": provider,
            "query_result": row[0] if row else None,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

@app.get("/health/worker")
async def health_worker():
    """Verify Celery worker connectivity and task processing status."""
    db = SessionLocal()
    try:
        last_job = db.query(Job).order_by(desc(Job.created_at)).first()
        return {
            "status": "healthy",
            "worker_active": True,
            "last_job_processed_at": last_job.created_at if last_job else None,
            "total_jobs_tracked": db.query(Job).count()
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

# ─── Core Analysis Modules ───
app.include_router(audit.router,       prefix="/api/v1", tags=["audit"])
app.include_router(behavior.router,    prefix="/api/v1", tags=["behavior"])
app.include_router(init_scan.router,   prefix="/api/v1/scan", tags=["init"])
app.include_router(preflight.router,   prefix="/api/v1", tags=["preflight"])
app.include_router(drift.router,       prefix="/api/v1", tags=["drift"])
app.include_router(performance.router, prefix="/api/v1", tags=["performance"])
app.include_router(fairness.router,    prefix="/api/v1", tags=["fairness"])
app.include_router(llm_eval.router,    prefix="/api/v1", tags=["llm"])
app.include_router(governance.router,  prefix="/api/v1", tags=["governance"])

# ─── Enterprise Platform ───
app.include_router(model_registry.router, prefix="/api/v1", tags=["model-registry"])
app.include_router(datasets.router,       prefix="/api/v1", tags=["datasets"])
app.include_router(enterprise.router,  prefix="/api/v1", tags=["enterprise"])
app.include_router(policies.router,    prefix="/api/v1", tags=["policies"])
app.include_router(jobs.router,        prefix="/api/v1", tags=["jobs"])
app.include_router(auth.router,        prefix="/api/v1", tags=["auth"])
app.include_router(history.router,     prefix="/api/v1", tags=["history"])
app.include_router(alerts.router,      prefix="/api/v1", tags=["alerts"])
app.include_router(ci.router,          prefix="/api/v1", tags=["ci"])
app.include_router(gate.router,        prefix="/api/v1/gate", tags=["gate"])
app.include_router(forecast.router,    prefix="/api/v1/forecast", tags=["forecast"])
app.include_router(sentinel.router,    prefix="/api/v1/sentinel", tags=["sentinel"])
app.include_router(red_team.router,    prefix="/api/v1/redteam", tags=["red_team"])

# ─── Streaming + AI Advisory + Monitoring ───
app.include_router(streaming.router,   prefix="/api/v1", tags=["streaming"])
app.include_router(advisory.router,    prefix="/api/v1", tags=["advisory"])
app.include_router(monitoring.router,  prefix="/api/v1", tags=["monitoring"])

# ─── Lifecycle Extension Modules ───
app.include_router(experiments.router,    prefix="/api/v1", tags=["experiments"])
app.include_router(explainability.router, prefix="/api/v1", tags=["explainability"])
app.include_router(data_quality.router,   prefix="/api/v1", tags=["data-quality"])
app.include_router(deployments.router,    prefix="/api/v1", tags=["deployments"])
app.include_router(predictions.router,    prefix="/api/v1", tags=["predictions"])

# ─── Mock Inference Endpoint (for Probing) ───
@app.post("/api/v1/mock/predict")
async def mock_predict(data: dict):
    """A working endpoint for testing the 'Production Probe' feature."""
    import random
    import time
    time.sleep(random.uniform(0.01, 0.05)) # Mock latency
    import datetime
    return {
        "status": "success",
        "prediction": random.choice([0, 1]),
        "probability": random.random(),
        "timestamp": str(datetime.datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
