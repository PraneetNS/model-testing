"""
Main API router for ML Guard
"""

from fastapi import APIRouter
from .endpoints import (
    projects,
    models,
    test_suites,
    test_runs,
    quality_gate,
    health
)

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    projects.router,
    prefix="/projects",
    tags=["projects"]
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"]
)

api_router.include_router(
    test_suites.router,
    prefix="/test-suites",
    tags=["test-suites"]
)

api_router.include_router(
    test_runs.router,
    prefix="/test-runs",
    tags=["test-runs"]
)

api_router.include_router(
    quality_gate.router,
    prefix="/quality-gate",
    tags=["quality-gate"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)