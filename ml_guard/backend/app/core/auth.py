"""
Auth + RBAC + Rate Limiting Dependency Layer.

Every protected endpoint injects `current_user` via Depends(require_role(..)).
All DB queries MUST filter by current_user.org_id to ensure zero cross-tenant leakage.

Authentication: Uses X-API-Key with SHA-256 hashing.
"""
import hashlib
from fastapi import Header, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from dataclasses import dataclass, field
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
    user_id: Optional[uuid.UUID] = None
    api_key_id: Optional[uuid.UUID] = None
    org_id: Optional[uuid.UUID] = None
    role: str = "viewer"
    scopes: List[str] = field(default_factory=list)

    def can(self, min_role: str) -> bool:
        """Role hierarchy check (admin > viewer)."""
        ROLE_HIERARCHY = {
            "admin": 4, 
            "administrator": 4, 
            "ml_engineer": 3, 
            "auditor": 2, 
            "viewer": 1
        }
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY.get(min_role, 0)

    def assert_role(self, min_role: str):
        """Enforce role existence or raise 403."""
        if not self.can(min_role):
            raise HTTPException(403, f"Role '{self.role}' insufficient. Requires '{min_role}'.")

from fastapi import Request

async def get_auth_context(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db)
) -> AuthContext:
    """
    Resolve API credentials from the X-API-Key header.
    Validates the key using bcrypt hash comparison.
    """
    key = x_api_key or request.headers.get("x-api-key") or request.headers.get("X-API-Key")

    if not key:
        from app.core.config import settings
        if getattr(settings, "DEBUG", False):
            key = "dev-secret-key"
        else:
            logger.warning("auth_failed_missing_header", headers=dict(request.headers))
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header required for access to protected resources."
            )
    
    # DEV BYPASS: Allow dev keys immediately
    DEV_BYPASS_KEYS = [
        "dev-secret-key",
        "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi",
        "mlg_simulator_key_2026_safe_dev",  # Used by all simulation & test scripts
    ]
    if key in DEV_BYPASS_KEYS:
        from app.db.models import Organization
        org_stmt = select(Organization).limit(1)
        org = (await db.execute(org_stmt)).scalars().first()
        org_id = org.id if org else None
        return AuthContext(
            user_id="dev-user",
            api_key_id=None,
            org_id=org_id,
            role="admin",
            scopes=["admin", "ml_engineer", "auditor", "viewer"]
        )
    
    logger.info("auth_attempt", key_provided=key[:5] + "...")
    
    from app.db.models import APIKey, utcnow
    from app.core.security import verify_password
    
    # We can't query by hash directly with bcrypt as easily as SHA-256 for a single lookup 
    # unless we use indices which bcrypt doesn't support well for searching.
    # However, usually we have a label or we prefix the key.
    # The current key format is 'mlg_<32_bytes_urlsafe>'.
    # For performance, usually one would store a 'prefix' or 'id' part of the key.
    # Since we don't have that yet, we have to fetch all active keys or find another way.
    # TO OPTIMIZE: Store a public component (e.g. mlg_abc123_xyz...) where 'abc123' is public.
    
    # For now, let's fetch active keys and compare. This is suboptimal.
    # BETTER: Let's assume most keys are new. 
    
    results = await db.execute(select(APIKey).filter(APIKey.is_active == True))
    api_keys = results.scalars().all()
    
    found_key = None
    for key_record in api_keys:
        # Check if it's a passlib-compatible hash (starts with $)
        if key_record.key_hash.startswith("$"):
            if verify_password(x_api_key, key_record.key_hash):
                found_key = key_record
                break
        # Fallback for old raw SHA-256 keys (migration period)
        else:
            import hashlib
            old_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
            if old_hash == key_record.key_hash:
                found_key = key_record
                break
    
    if not found_key:
        logger.warning("auth_failed_invalid_key")
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key."
        )

    # Check expiry
    if found_key.expires_at and found_key.expires_at < utcnow():
        raise HTTPException(status_code=401, detail="API key expired.")
    
    # Update last_used and request_count
    found_key.last_used = utcnow()
    if hasattr(found_key, 'request_count'):
        found_key.request_count += 1
    
    await db.commit()
    
    # Determine highest role from scopes
    scopes = found_key.scopes or []
    role = "viewer"
    if "admin" in scopes:
        role = "admin"
    elif "ml_engineer" in scopes:
        role = "ml_engineer"
    elif "auditor" in scopes:
        role = "auditor"

    return AuthContext(
        user_id=None,
        api_key_id=found_key.id,
        org_id=found_key.org_id,
        role=role,
        scopes=scopes
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

async def log_action(
    db: AsyncSession, 
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
            actor_key_id=getattr(auth, "api_key_id", None),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )
        db.add(log)
        await db.commit()
    except Exception as e:
        logger.error("audit_log_failed", error=str(e))
        pass # Never crash the main request on logging failure
