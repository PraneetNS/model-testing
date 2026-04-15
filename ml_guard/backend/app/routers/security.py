"""
security.py — Real-time Security Observability Router

Provides endpoints for:
- Fetching real-time security alerts (log of injection attempts)
- Real-time security scanning of prediction payloads
- Security dashboard stats
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.db.session import get_db
from app.db.models import SecurityAlert, APIKey
from app.core.auth import AuthContext, require_role

router = APIRouter()


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class SecurityScanRequest(BaseModel):
    model_id: str
    features: Dict[str, Any]
    prediction: Optional[Any] = None


class SecurityAlertSchema(BaseModel):
    id: str
    timestamp: datetime
    alert_type: str
    endpoint: Optional[str]
    ip: Optional[str]
    details: Optional[Dict[str, Any]]


# ─── Alert Retrieval ─────────────────────────────────────────────────────────

@router.get("/alerts", response_model=List[SecurityAlertSchema], tags=["security"])
async def get_security_alerts(
    limit: int = Query(default=50, le=500),
    alert_type: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    Retrieve the latest security alerts (injection attempts, blocked calls).
    These are populated by the SecurityHardeningMiddleware in real-time.
    """
    stmt = select(SecurityAlert)
    if alert_type:
        stmt = stmt.filter(SecurityAlert.alert_type == alert_type)
    
    result = await db.execute(stmt.order_by(SecurityAlert.timestamp.desc()).limit(limit))
    rows = result.scalars().all()

    return [
        SecurityAlertSchema(
            id=str(r.id),
            timestamp=r.timestamp,
            alert_type=r.alert_type,
            endpoint=r.endpoint,
            ip=r.ip,
            details=r.details
        )
        for r in rows
    ]


@router.get("/stats", tags=["security"])
async def get_security_stats(
    days: int = Query(default=7, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    Aggregated security statistics for the dashboard.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    total_alerts = (await db.execute(
        select(func.count(SecurityAlert.id)).filter(SecurityAlert.timestamp >= cutoff)
    )).scalar() or 0

    # Alerts by type
    type_stmt = select(SecurityAlert.alert_type, func.count(SecurityAlert.id)).filter(
        SecurityAlert.timestamp >= cutoff
    ).group_by(SecurityAlert.alert_type)
    type_results = (await db.execute(type_stmt)).all()
    
    type_breakdown = {t: c for t, c in type_results}

    return {
        "total_alerts": total_alerts,
        "days_window": days,
        "type_breakdown": type_breakdown,
        "status": "HEALTHY" if total_alerts < 10 else "ATTENTION_REQUIRED"
    }


# ─── Live Security Scanning ──────────────────────────────────────────────────

async def run_realtime_scan(model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal logic for real-time security scanning.
    """
    anomalies = []
    for feat, val in features.items():
        if isinstance(val, (int, float)):
            if val > 1_000_000 or val < -1_000_000:
                anomalies.append({"feature": feat, "value": val, "issue": "extreme_value"})

    import re
    PII_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    
    for feat, val in features.items():
        if isinstance(val, str) and re.search(PII_PATTERN, val):
            anomalies.append({"feature": feat, "issue": "pii_detected"})

    risk_score = min(len(anomalies) * 20, 100)
    risk_level = "LOW"
    if risk_score > 60: risk_level = "HIGH"
    elif risk_score > 20: risk_level = "MEDIUM"

    return {
        "model_id": model_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "anomalies": anomalies,
        "scanned_at": datetime.utcnow().isoformat(),
        "compliant": risk_score < 40
    }


@router.post("/scan-live", tags=["security"])
async def scan_payload_live(
    req: SecurityScanRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    Run a real-time security scan on a prediction payload.
    """
    return await run_realtime_scan(req.model_id, req.features)
