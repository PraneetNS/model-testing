from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
from app.core.config import settings
from app.api.v1.api import api_router
from app.infrastructure.database import engine, Base
from app.infrastructure.persistence import models as sql_models

from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise ML Governance & Quality Gate Platform",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/api/v1/openapi.json"
)

# Startup event to create tables and init limiter
@app.on_event("startup")
async def on_startup():
    logger.info("Initializing Enterprise Services...")
    Base.metadata.create_all(bind=engine)
    
    # Initialize Rate Limiter with Redis
    try:
        r = redis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}", encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(r)
        logger.info("Rate Limiter initialized")
    except Exception as e:
        logger.error("Rate Limiter initialization failed", error=str(e))

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0", "platform": "Enterprise"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
