from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints import models, quality_gate, test_suites, governance, monitoring, compliance, llm, artifacts, retraining
from app.api.v1 import reports

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(governance.router, prefix="/governance", tags=["governance"])
api_router.include_router(monitoring.router, prefix="/monitor", tags=["monitoring"])
api_router.include_router(quality_gate.router, prefix="", tags=["quality-gate"])
api_router.include_router(artifacts.router, prefix="", tags=["artifacts"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(retraining.router, prefix="/models", tags=["retraining"])
