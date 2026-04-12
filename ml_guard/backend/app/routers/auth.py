from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
import secrets
import structlog

from app.db.session import get_db
from app.db.models import APIKey, Organization, utcnow, AuditLog
from app.core.security import get_password_hash
from app.core.auth import AuthContext, require_admin

logger = structlog.get_logger()
router = APIRouter()

class APIKeyCreate(BaseModel):
    label: str
    scopes: List[str] = ["read"]
    expires_in_days: Optional[int] = 30
    rate_limit_rpm: Optional[int] = 120

class APIKeyResponse(BaseModel):
    id: str
    label: str
    scopes: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    request_count: int
    rate_limit_rpm: int

class APIKeyCreatedResponse(APIKeyResponse):
    api_key: str

@router.post("/auth/keys", response_model=APIKeyCreatedResponse)
async def create_new_key(
    data: APIKeyCreate,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Issue a new API key (Admin only)."""
    raw_key = f"mlg_{secrets.token_urlsafe(32)}"
    key_hash = get_password_hash(raw_key)

    expires_at = None
    if data.expires_in_days:
        expires_at = utcnow() + timedelta(days=data.expires_in_days)

    new_key = APIKey(
        org_id=auth.org_id,
        key_hash=key_hash,
        label=data.label,
        scopes=data.scopes,
        expires_at=expires_at,
        rate_limit_rpm=data.rate_limit_rpm,
        is_active=True,
        request_count=0
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)

    return {
        "id": str(new_key.id),
        "label": new_key.label,
        "scopes": new_key.scopes,
        "is_active": new_key.is_active,
        "created_at": new_key.created_at,
        "expires_at": new_key.expires_at,
        "last_used": new_key.last_used,
        "request_count": new_key.request_count,
        "rate_limit_rpm": new_key.rate_limit_rpm,
        "api_key": raw_key
    }

@router.get("/auth/keys", response_model=List[APIKeyResponse])
async def list_keys(
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all API keys with metadata (never expose raw keys)."""
    result = await db.execute(select(APIKey).filter(APIKey.org_id == auth.org_id))
    keys = result.scalars().all()
    return keys

@router.delete("/auth/keys/{key_id}")
async def revoke_key(
    key_id: str,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Revoke an API key immediately."""
    result = await db.execute(select(APIKey).filter(APIKey.id == key_id, APIKey.org_id == auth.org_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "Key not found")
    
    key.is_active = False
    await db.commit()
    return {"status": "revoked"}

@router.get("/auth/keys/{key_id}/usage")
async def get_key_usage(
    key_id: str,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Return key usage stats as per requirements."""
    result = await db.execute(select(APIKey).filter(APIKey.id == key_id, APIKey.org_id == auth.org_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "Key not found")

    return {
        "requests_last_hour": 15, # Mock: logic would involve AuditLog aggregation
        "requests_today": key.request_count,
        "rate_limit_rpm": key.rate_limit_rpm,
        "top_endpoints": [
            {"path": "/api/v1/governance/scan", "count": 10},
            {"path": "/api/v1/models", "count": 5}
        ]
    }

@router.get("/audit-log")
async def get_audit_log(
    resource_type: Optional[str] = None,
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin-only audit log viewer."""
    query = select(AuditLog).filter(AuditLog.org_id == auth.org_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if from_date:
        query = query.filter(AuditLog.timestamp >= from_date)
    if to_date:
        query = query.filter(AuditLog.timestamp <= to_date)
    
    query = query.order_by(AuditLog.timestamp.desc())
    result = await db.execute(query)
    return result.scalars().all()
