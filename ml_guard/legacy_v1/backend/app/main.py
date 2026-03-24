"""
ML Guard - FastAPI Backend Service
FireFlink-style ML Model Testing Platform
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from .api.routes import api_router
from .core.config import settings
from .core.logging import setup_logging

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting ML Guard service", version=settings.VERSION)
    yield
    logger.info("Shutting down ML Guard service")

# Create FastAPI application
app = FastAPI(
    title="ML Guard",
    description="FireFlink ML Model Testing Platform - Pre-deployment ML Quality Assurance",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ml-guard", "version": settings.VERSION}

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "ML Guard",
        "description": "FireFlink ML Model Testing Platform",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our custom logging
    )