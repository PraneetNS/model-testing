from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
import secrets
import structlog

from app.api.v1 import deps
from app.db.models import APIKey, Organization, utcnow
from app.core.security import get_password_hash
from app.core.auth import AuthContext, require_admin

logger = structlog.get_logger()
router = APIRouter()

class APIKeyCreate(BaseModel):
    label: str
    scopes: List[str] = ["read"]
    expires_in_days: Optional[int] = 30
    rate_limit_rpm: Optional[int] = 120
    org_id: Optional[str] = None # Admin can specify org

class APIKeyResponse(BaseModel):
    id: str
    label: str
    scopes: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    request_count: int
    rate_limit_rpm: int

class APIKeyCreatedResponse(APIKeyResponse):
    raw_key: str # Only returned once

@router.post("/keys", response_model=APIKeyCreatedResponse)
async def create_new_key(
    data: APIKeyCreate,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(deps.get_db)
):
    """Issue a new API key (Admin only)."""
    # Use specified org_id or current org_id
    target_org_id = data.org_id or auth.org_id
    if not target_org_id:
        raise HTTPException(400, "Organization ID required.")

    # Generate raw key
    raw_key = f"mlg_{secrets.token_urlsafe(32)}"
    key_hash = get_password_hash(raw_key)

    expires_at = None
    if data.expires_in_days:
        expires_at = utcnow() + timedelta(days=data.expires_in_days)

    new_key = APIKey(
        org_id=target_org_id,
        key_hash=key_hash,
        label=data.label,
        scopes=data.scopes,
        expires_at=expires_at,
        rate_limit_rpm=data.rate_limit_rpm,
        is_active=True
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)

    return {
        **new_key.__dict__,
        "id": str(new_key.id),
        "raw_key": raw_key
    }

@router.get("/keys", response_model=List[APIKeyResponse])
async def list_keys(
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(deps.get_db)
):
    """List all API keys for the current organization."""
    result = await db.execute(
        select(APIKey).filter(APIKey.org_id == auth.org_id)
    )
    keys = result.scalars().all()
    return keys

@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(deps.get_db)
):
    """Immediately revoke an API key."""
    result = await db.execute(
        select(APIKey).filter(APIKey.id == key_id, APIKey.org_id == auth.org_id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "API Key not found.")
    
    key.is_active = False
    await db.commit()
    return {"status": "revoked"}

@router.get("/keys/{key_id}/usage")
async def get_key_usage(
    key_id: str,
    auth: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(deps.get_db)
):
    """Get usage statistics for an API key."""
    # In a real app we'd query AuditLogs or a Metrics service
    # For now we'll return the request_count and some mock data as requested
    result = await db.execute(
        select(APIKey).filter(APIKey.id == key_id, APIKey.org_id == auth.org_id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "API Key not found.")

    return {
        "requests_last_hour": 15, # Mock
        "requests_today": key.request_count,
        "rate_limit_rpm": key.rate_limit_rpm,
        "top_endpoints": [
            {"path": "/api/v1/governance/scan", "count": 10},
            {"path": "/api/v1/models", "count": 5}
        ]
    }
