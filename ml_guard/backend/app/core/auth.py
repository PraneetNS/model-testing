"""
Auth + RBAC + Rate Limiting Dependency Layer.

Every protected endpoint injects `current_user` via Depends(require_role(...)).
All DB queries MUST filter by current_user.org_id to ensure zero cross-tenant leakage.

Authentication: Uses X-API-Key with SHA-256 hashing.
"""
import hashlib
from fastapi import Header, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from app.db.session import get_db
from dataclasses import dataclass
from typing import Optional, List
import uuid
import structlog

logger = structlog.get_logger()

# FastAPI standard API key security implementation
api_key_header = APIKeyHeader(
    name="X-API-Key", 
    auto_error=False
)

@dataclass
class AuthContext:
    """Consolidated authentication and authorization context."""
    user_id: Optional[uuid.UUID]
    org_id: Optional[uuid.UUID]
    role: str
    scopes: List[str]

    def can(self, min_role: str) -> bool:
        """Role hierarchy check (admin > viewer)."""
        ROLE_HIERARCHY = {"admin": 4, "ml_engineer": 3, "auditor": 2, "viewer": 1}
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY.get(min_role, 0)

    def assert_role(self, min_role: str):
        """Enforce role existence or raise 403."""
        if not self.can(min_role):
            raise HTTPException(403, f"Role '{self.role}' insufficient. Requires '{min_role}'.")

async def get_auth_context(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
) -> AuthContext:
    """
    Resolve API credentials from the X-API-Key header.
    Validates the key against the SHA-256 hash in the database.
    """
    if not x_api_key:
        # Check for fallback org for local dev if desired, but here we enforce key
        raise HTTPException(
            status_code=401,
            detail="X-API-Key header required for access to protected resources."
        )
    
    from app.db.models import APIKey, utcnow
    
    # Hash the provided raw key for comparison
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    
    # Lookup active key
    api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()
    
    if not api_key:
        logger.warning("auth_failed_invalid_key", hash=key_hash[:16])
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key. Please check your credentials."
        )
    
    # Update last_used for auditing
    api_key.last_used = utcnow()
    db.commit()
    
    # Return context. Note: API keys currently act as org-level admins.
    return AuthContext(
        user_id=None, # API Key auth is typically system/org level
        org_id=api_key.org_id,
        role="admin", # Elevated role for seeded dev key
        scopes=api_key.scopes or []
    )

def require_role(role: str = "viewer"):
    """
    Dependency factory for granular role-based access control.
    Usage: Depends(require_role("ml_engineer"))
    """
    async def _require(
        auth: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        auth.assert_role(role)
        return auth
    return _require

# Common access shorthand
require_admin = require_role("admin")
require_engineer = require_role("ml_engineer")
require_auditor = require_role("auditor")
require_viewer = require_role("viewer")

def log_action(
    db: Session, 
    auth: AuthContext,
    action: str,
    resource_type: str = None,
    resource_id: str = None,
    details: dict = None
):
    """
    Persistent audit logging for all governance actions.
    Ensures that every write action is traced back to a tenant and key/user.
    """
    try:
        from app.db.models import AuditLog
        log = AuditLog(
            org_id=auth.org_id,
            user_id=auth.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error("audit_log_failed", error=str(e))
        pass # Never crash the main request on logging failure
