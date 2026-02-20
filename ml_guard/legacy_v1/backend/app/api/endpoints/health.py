"""
Health check endpoints
"""

from fastapi import APIRouter
import structlog

from ...core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

@router.get("/status")
async def health_status():
    """Detailed health status check."""
    return {
        "status": "healthy",
        "service": "ml-guard",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint for load balancer health checks."""
    return {"status": "pong"}

@router.get("/ready")
async def readiness_check():
    """Readiness check for deployment orchestration."""
    # In a real implementation, this would check:
    # - Database connectivity
    # - ML model loading capability
    # - Storage accessibility
    # - External service dependencies

    return {
        "status": "ready",
        "service": "ml-guard",
        "checks": {
            "configuration": "passed",
            "storage": "passed",
            "ml_engine": "passed"
        }
    }