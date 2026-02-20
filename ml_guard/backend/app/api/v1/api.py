
from fastapi import APIRouter
from app.api.v1.endpoints import models, quality_gate, test_suites, auth, governance

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(governance.router, prefix="/governance", tags=["governance"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(quality_gate.router, prefix="/quality-gate", tags=["quality-gate"])
api_router.include_router(test_suites.router, prefix="/test-suites", tags=["test-suites"])
