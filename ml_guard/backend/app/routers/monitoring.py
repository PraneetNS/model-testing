from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.core.auth import AuthContext, require_engineer, log_action
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class MonitorLogRequest(BaseModel):
    endpoint_url: str
    status: str # HEALTHY, DEGRADED, UNSTABLE
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate_pct: float
    probe_count: int

@router.post("/monitoring/log")
async def log_monitoring_event(
    body: MonitorLogRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_engineer)
):
    """Log a client-side monitoring probe to the enterprise audit trail."""
    log_action(
        db, auth, "monitor.probe", 
        resource_type="endpoint", 
        resource_id=body.endpoint_url[:64], 
        details=body.model_dump()
    )
    return {"status": "logged"}
