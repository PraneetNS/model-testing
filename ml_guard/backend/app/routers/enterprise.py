"""
Multi-Tenant Organization + RBAC Router + Enterprise Summary.
Handles org creation, user management, API key generation, project management,
and enterprise-level summary/dashboard endpoints.
"""
import uuid
import hashlib
import secrets
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from app.db.session import get_db
from app.db.models import Organization, User, Project, APIKey, AuditLog, PolicyVersion, PolicyRule, utcnow
from app.services.enterprise_service import (
    get_enterprise_summary, get_scans_paginated,
    get_models_paginated, get_audit_logs_paginated,
)

router = APIRouter()


class OrgCreate(BaseModel):
    name: str
    slug: str
    plan: str = "free"


class UserCreate(BaseModel):
    email: str
    name: str
    role: str = "viewer"
    password: str = ""


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class APIKeyCreate(BaseModel):
    label: str
    scopes: List[str] = ["audit", "behavior", "monitor"]


class PolicyPatch(BaseModel):
    """Partial policy update."""
    config: Optional[dict] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


# ─── LOG HELPER ───
def _log(db, org_id, user_id, action, resource_type=None, resource_id=None, details=None, ip=None):
    db.add(AuditLog(
        org_id=org_id, user_id=user_id, action=action,
        resource_type=resource_type, resource_id=str(resource_id) if resource_id else None,
        details=details, ip_address=ip,
    ))
    db.commit()


# ══════════════════════════════════════════════════════
# ENTERPRISE SUMMARY (v7.0 — real data, zero dummy)
# ══════════════════════════════════════════════════════

@router.get("/enterprise/summary")
def enterprise_summary(org_id: str = "", db: Session = Depends(get_db)):
    """Real-time enterprise summary powered by actual database records."""
    return get_enterprise_summary(db, org_id or None)


@router.get("/enterprise/scans")
def enterprise_scans(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_dir: str = Query("desc"),
    db: Session = Depends(get_db),
):
    """Paginated, sortable scan history."""
    return get_scans_paginated(db, page, per_page, sort_by, sort_dir)


@router.get("/enterprise/models")
def enterprise_models(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Paginated model registry with latest scan data."""
    return get_models_paginated(db, page, per_page)


@router.get("/enterprise/policies")
def enterprise_policies(org_id: str = "", db: Session = Depends(get_db)):
    """All policies — both PolicyVersion and PolicyRule models."""
    from sqlalchemy import desc as sql_desc
    q_version = db.query(PolicyVersion).order_by(sql_desc(PolicyVersion.created_at))
    q_rule = db.query(PolicyRule).order_by(sql_desc(PolicyRule.created_at))
    if org_id:
        q_version = q_version.filter(PolicyVersion.org_id == org_id)
        q_rule = q_rule.filter(PolicyRule.org_id == org_id)
    versions = q_version.limit(50).all()
    rules = q_rule.limit(50).all()
    return [
        {
            "id": str(p.id), "name": p.name,
            "version": getattr(p, "version", "N/A"),
            "is_active": p.is_active,
            "config": getattr(p, "config", getattr(p, "rules_json", {})),
            "notes": getattr(p, "notes", ""),
            "created_at": str(p.created_at),
            "type": "version" if hasattr(p, "version") else "rule",
        }
        for p in versions + rules
    ]


@router.get("/enterprise/audit-logs")
def enterprise_audit_logs(
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    org_id: str = "",
    db: Session = Depends(get_db),
):
    """Paginated audit log trail."""
    return get_audit_logs_paginated(db, page, per_page, org_id or None)


@router.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    """Check database connection and type (Postgres/SQLite/Neon)."""
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        diag = str(db.bind.url)
        db_type = "postgres" if "postgres" in diag else "sqlite"
        is_neon = "neon" in diag
        return {
            "status": "connected",
            "db": db_type,
            "cloud": is_neon,
            "provider": "Neon" if is_neon else ("Local" if db_type == "postgres" else "File")
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}



# ══════════════════════════════════════════════════════
# POLICY PATCH (v7.0)
# ══════════════════════════════════════════════════════

@router.patch("/policies/{policy_id}")
def patch_policy(policy_id: str, body: PolicyPatch, db: Session = Depends(get_db)):
    """
    Partial update of a policy (config thresholds, name, notes, active status).
    Creates an audit log entry for traceability.
    """
    # Try PolicyVersion first, then PolicyRule
    policy = db.get(PolicyVersion, policy_id)
    policy_type = "version"
    if not policy:
        policy = db.get(PolicyRule, policy_id)
        policy_type = "rule"
    if not policy:
        raise HTTPException(404, "Policy not found.")

    changes = {}
    if body.name is not None:
        policy.name = body.name
        changes["name"] = body.name
    if body.notes is not None and hasattr(policy, "notes"):
        policy.notes = body.notes
        changes["notes"] = body.notes
    if body.is_active is not None:
        policy.is_active = body.is_active
        changes["is_active"] = body.is_active
    if body.config is not None:
        if policy_type == "version":
            # Merge with existing config
            existing = policy.config or {}
            merged = {**existing, **body.config}
            policy.config = merged
            changes["config"] = merged
        else:
            existing = policy.rules_json or {}
            merged = {**existing, **body.config}
            policy.rules_json = merged
            changes["config"] = merged

    db.commit()
    db.refresh(policy)

    # Audit log
    db.add(AuditLog(
        org_id=getattr(policy, "org_id", None),
        action="policy.update",
        resource_type="policy",
        resource_id=str(policy.id),
        details={"changes": changes, "policy_type": policy_type},
    ))
    db.commit()

    return {
        "id": str(policy.id), "name": policy.name,
        "is_active": policy.is_active,
        "config": getattr(policy, "config", getattr(policy, "rules_json", {})),
        "updated": True, "changes": changes,
    }


# ══════════════════════════════════════════════════════
# DB HEALTH CHECK
# ══════════════════════════════════════════════════════

@router.get("/health/db")
def db_health_check(db: Session = Depends(get_db)):
    """Check database connectivity and return status."""
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT 1")).fetchone()
        # Detect DB type
        uri = str(db.bind.url) if db.bind else "unknown"
        if "postgresql" in uri or "neon" in uri:
            db_type = "neon" if "neon" in uri else "postgresql"
        elif "sqlite" in uri:
            db_type = "sqlite"
        else:
            db_type = "unknown"
        return {"status": "connected", "db": db_type, "check": "ok"}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


# ══════════════════════════════════════════════════════
# ORGANIZATIONS (existing, unchanged)
# ══════════════════════════════════════════════════════
@router.post("/orgs")
def create_org(body: OrgCreate, db: Session = Depends(get_db)):
    existing = db.query(Organization).filter(Organization.slug == body.slug).first()
    if existing:
        raise HTTPException(400, "Organization slug already exists.")
    org = Organization(name=body.name, slug=body.slug, plan=body.plan)
    db.add(org)
    db.commit()
    db.refresh(org)
    _log(db, str(org.id), None, "org.create", "organization", org.id, {"name": body.name})
    return {"id": str(org.id), "name": org.name, "slug": org.slug, "plan": org.plan}


@router.get("/orgs")
def list_orgs(db: Session = Depends(get_db)):
    orgs = db.query(Organization).all()
    return [{"id": str(o.id), "name": o.name, "slug": o.slug, "plan": o.plan, "created_at": str(o.created_at)} for o in orgs]


@router.get("/orgs/{org_id}")
def get_org(org_id: str, db: Session = Depends(get_db)):
    org = db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    return {
        "id": str(org.id), "name": org.name, "slug": org.slug, "plan": org.plan,
        "user_count": len(org.users), "project_count": len(org.projects),
        "created_at": str(org.created_at),
    }


# ══════════════════════════════════════════════════════
# USERS (existing, unchanged)
# ══════════════════════════════════════════════════════
@router.post("/orgs/{org_id}/users")
def create_user(org_id: str, body: UserCreate, db: Session = Depends(get_db)):
    org = db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    if body.role not in ("admin", "ml_engineer", "auditor", "viewer"):
        raise HTTPException(400, "Invalid role. Must be: admin, ml_engineer, auditor, viewer.")
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(400, "Email already in use.")
    pw_hash = hashlib.sha256(body.password.encode()).hexdigest() if body.password else None
    user = User(
        org_id=org_id, email=body.email, name=body.name,
        role=body.role, password_hash=pw_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _log(db, org_id, str(user.id), "user.create", "user", user.id, {"email": body.email, "role": body.role})
    return {"id": str(user.id), "email": user.email, "name": user.name, "role": user.role}


@router.get("/orgs/{org_id}/users")
def list_users(org_id: str, db: Session = Depends(get_db)):
    users = db.query(User).filter(User.org_id == org_id).all()
    return [{"id": str(u.id), "email": u.email, "name": u.name, "role": u.role, "is_active": u.is_active} for u in users]


# ══════════════════════════════════════════════════════
# PROJECTS (existing, unchanged)
# ══════════════════════════════════════════════════════
@router.post("/orgs/{org_id}/projects")
def create_project(org_id: str, body: ProjectCreate, db: Session = Depends(get_db)):
    org = db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    proj = Project(org_id=org_id, name=body.name, description=body.description)
    db.add(proj)
    db.commit()
    db.refresh(proj)
    _log(db, org_id, None, "project.create", "project", proj.id, {"name": body.name})
    return {"id": str(proj.id), "name": proj.name, "description": proj.description}


@router.get("/orgs/{org_id}/projects")
def list_projects(org_id: str, db: Session = Depends(get_db)):
    projects = db.query(Project).filter(Project.org_id == org_id).all()
    return [{"id": str(p.id), "name": p.name, "model_count": len(p.models)} for p in projects]


# ══════════════════════════════════════════════════════
# API KEYS (existing, unchanged)
# ══════════════════════════════════════════════════════
@router.post("/orgs/{org_id}/api-keys")
def create_api_key(org_id: str, body: APIKeyCreate, db: Session = Depends(get_db)):
    org = db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    raw_key = f"mlg_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    api_key = APIKey(org_id=org_id, key_hash=key_hash, label=body.label, scopes=body.scopes)
    db.add(api_key)
    db.commit()
    _log(db, org_id, None, "api_key.create", "api_key", api_key.id, {"label": body.label})
    return {"id": str(api_key.id), "key": raw_key, "label": body.label, "scopes": body.scopes,
            "warning": "Store this key securely. It will not be shown again."}


@router.get("/orgs/{org_id}/api-keys")
def list_api_keys(org_id: str, db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.org_id == org_id).all()
    return [{"id": str(k.id), "label": k.label, "scopes": k.scopes, "is_active": k.is_active, "last_used": str(k.last_used)} for k in keys]
