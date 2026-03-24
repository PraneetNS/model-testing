"""
Auth + RBAC + Rate Limiting Dependency Layer.

Every protected endpoint injects `current_user` via Depends(require_role(...)).
All DB queries MUST filter by current_user.org_id → zero cross-tenant leakage.

Roles:
  admin       → full control
  ml_engineer → run scans, view results
  auditor     → read-only across all data
  viewer      → summary endpoints only
"""
import hashlib
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import Depends, HTTPException, Header, Request
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User, APIKey, Organization, AuditLog

# ─── Role hierarchy (higher includes lower) ───
ROLE_HIERARCHY = {
    "admin": 4,
    "ml_engineer": 3,
    "auditor": 2,
    "viewer": 1,
}

# ─── In-memory rate limiter (per org) ───
_rate_buckets: dict = defaultdict(list)  # org_id → [timestamp, ...]
RATE_LIMITS = {
    "free": 100,        # requests per minute
    "pro": 500,
    "enterprise": 5000,
}


def _check_rate_limit(org_id: str, plan: str):
    """Sliding window rate limit per org."""
    limit = RATE_LIMITS.get(plan, 100)
    now = time.time()
    window = 60.0  # 1 minute
    bucket = _rate_buckets[org_id]
    # Prune old entries
    _rate_buckets[org_id] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[org_id]) >= limit:
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded: {limit} requests/min for '{plan}' plan. Upgrade to increase."
        )
    _rate_buckets[org_id].append(now)


class AuthContext:
    """Injected into every protected endpoint."""
    def __init__(self, user: User, org: Organization):
        self.user = user
        self.org = org
        self.user_id = str(user.id)
        self.org_id = str(org.id)
        self.role = user.role
        self.plan = org.plan or "free"

    def can(self, min_role: str) -> bool:
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY.get(min_role, 0)

    def assert_role(self, min_role: str):
        if not self.can(min_role):
            raise HTTPException(403, f"Role '{self.role}' insufficient. Requires '{min_role}'.")


async def _resolve_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> AuthContext:
    """
    Resolve auth from:
    1. X-Api-Key header → look up hashed key → resolve org + synthetic user
    2. X-Org-Id + X-User-Email headers → direct lookup (dev mode / session-based)
    """
    # ─── Path 1: API Key auth ───
    if x_api_key:
        key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
        api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash, APIKey.is_active == True).first()
        if not api_key:
            raise HTTPException(401, "Invalid API key.")
        org = db.get(Organization, str(api_key.org_id))
        if not org:
            raise HTTPException(401, "Organization not found for this API key.")
        # Update last_used
        api_key.last_used = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        # Find an admin user for this org (API key acts as org-level admin)
        user = db.query(User).filter(User.org_id == str(org.id), User.role == "admin").first()
        if not user:
            # Create a synthetic system user
            user = User(org_id=str(org.id), email=f"api@{org.slug}", name="API System", role="ml_engineer")
            db.add(user)
            db.commit()
            db.refresh(user)
        _check_rate_limit(str(org.id), org.plan or "free")
        return AuthContext(user=user, org=org)

    # ─── Path 2: Header-based (dev / session) ───
    if x_org_id and x_user_email:
        user = db.query(User).filter(User.email == x_user_email, User.org_id == x_org_id).first()
        if not user:
            raise HTTPException(401, f"User '{x_user_email}' not found in org '{x_org_id}'.")
        if not user.is_active:
            raise HTTPException(403, "Account deactivated.")
        org = db.get(Organization, x_org_id)
        if not org:
            raise HTTPException(401, "Organization not found.")
        _check_rate_limit(str(org.id), org.plan or "free")
        return AuthContext(user=user, org=org)

    # ─── Path 3: Fallback — create a default org for easy local dev ───
    # This allows the existing UI to work without auth headers
    org = db.query(Organization).first()
    if not org:
        org = Organization(name="Default", slug="default", plan="enterprise")
        db.add(org)
        db.commit()
        db.refresh(org)
    user = db.query(User).filter(User.org_id == str(org.id)).first()
    if not user:
        user = User(org_id=str(org.id), email="admin@default.local", name="Local Admin", role="admin")
        db.add(user)
        db.commit()
        db.refresh(user)
    return AuthContext(user=user, org=org)


def require_role(min_role: str = "viewer"):
    """FastAPI dependency factory. Usage: Depends(require_role("ml_engineer"))"""
    async def _dep(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
        auth.assert_role(min_role)
        return auth
    return _dep


# ─── Convenience aliases ───
require_admin = require_role("admin")
require_engineer = require_role("ml_engineer")
require_auditor = require_role("auditor")
require_viewer = require_role("viewer")

# Public alias used by fairness.py, llm_eval.py routers
get_auth_context = _resolve_auth


def log_action(db: Session, auth: AuthContext, action: str, resource_type: str = None,
               resource_id: str = None, details: dict = None):
    """Write to audit_logs with org context."""
    db.add(AuditLog(
        org_id=auth.org_id,
        user_id=auth.user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    ))
    db.commit()
