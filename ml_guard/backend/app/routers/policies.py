"""
Versioned Governance Policy Router.
- Create, list, activate policies
- Attach to scans for reproducibility
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from pydantic import BaseModel
from typing import Optional
import structlog
from app.db.session import get_db
from app.db.models import PolicyVersion, PolicyRule, AuditLog, utcnow
from app.core.auth import AuthContext, get_auth_context, require_role

logger = structlog.get_logger()
router = APIRouter()

DEFAULT_POLICY_CONFIG = {
    "max_psi":              0.20,
    "max_jsd":              0.10,
    "min_accuracy":         0.80,
    "min_f1":               0.65,
    "max_overfit_gap":      0.08,
    "max_brier_score":      0.20,
    "min_stability_score":  0.90,
    "min_governance_score": 70.0,
    "max_latency_p95_ms":   300,
}


class PolicyCreate(BaseModel):
    name: str = "Default Policy"
    config: dict = {}
    notes: str = ""
    org_id: str = ""


@router.get("/policies")
async def list_policies(org_id: str = "", db: AsyncSession = Depends(get_db)):
    # Also include the new PolicyRule model in the list
    stmt_v = select(PolicyVersion).order_by(desc(PolicyVersion.created_at)).limit(25)
    stmt_r = select(PolicyRule).order_by(desc(PolicyRule.created_at)).limit(25)
    
    if org_id:
        stmt_v = stmt_v.filter(PolicyVersion.org_id == org_id)
        stmt_r = stmt_r.filter(PolicyRule.org_id == org_id)
        
    versions = (await db.execute(stmt_v)).scalars().all()
    rules = (await db.execute(stmt_r)).scalars().all()
    
    return [
        {
            "id": str(p.id), "name": p.name, "version": getattr(p, "version", "N/A"),
            "is_active": p.is_active, "config": getattr(p, "config", getattr(p, "rules_json", {})),
            "notes": getattr(p, "notes", ""), "created_at": str(p.created_at),
        }
        for p in versions + rules
    ]


@router.post("/policies")
async def create_policy(body: PolicyCreate, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    # Merge with defaults — any missing keys get default values
    full_config = {**DEFAULT_POLICY_CONFIG, **body.config}

    # Auto-version: count existing policies with the same name + org
    existing = (await db.execute(
        select(func.count(PolicyVersion.id)).filter(
            PolicyVersion.name == body.name,
            PolicyVersion.org_id == (body.org_id or None),
        )
    )).scalar() or 0

    policy = PolicyVersion(
        org_id=body.org_id or None,
        name=body.name,
        version=existing + 1,
        config=full_config,
        notes=body.notes,
        is_active=True,
    )
    db.add(policy)

    # Deactivate previous versions with same name
    old_stmt = select(PolicyVersion).filter(
        PolicyVersion.name == body.name,
        PolicyVersion.org_id == (body.org_id or None),
        PolicyVersion.id != policy.id,
    )
    old = (await db.execute(old_stmt)).scalars().all()
    for o in old:
        o.is_active = False

    await db.commit()
    await db.refresh(policy)

    db.add(AuditLog(
        org_id=body.org_id or None,
        action="policy.create",
        resource_type="policy",
        resource_id=str(policy.id),
        details={"name": body.name, "version": policy.version, "config": full_config},
    ))
    await db.commit()

    return {
        "id": str(policy.id), "name": policy.name, "version": policy.version,
        "config": policy.config, "is_active": policy.is_active,
    }


@router.get("/policies/active")
async def get_active_policy(db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)):
    try:
        org_id = auth.org_id
        logger.info("fetch_active_policy", org_id=str(org_id))
        
        # 1. First try the new PolicyRule model
        try:
            stmt_rule = select(PolicyRule).filter(PolicyRule.is_active == True)
            if org_id:
                stmt_rule = stmt_rule.filter(PolicyRule.org_id == org_id)
            
            active_rule = (await db.execute(stmt_rule.order_by(desc(PolicyRule.created_at)).limit(1))).scalar_one_or_none()
            
            if active_rule:
                merged_config = {**DEFAULT_POLICY_CONFIG, **active_rule.rules_json}
                return {
                    "id": str(active_rule.id), 
                    "name": active_rule.name,
                    "rules": active_rule.rules_json,
                    "config": merged_config,
                    "version": "active_rule",
                    "is_active": True,
                }
        except Exception as e:
            logger.error("policy_rule_fetch_error", error=str(e))

        # 2. Fallback to older PolicyVersion
        try:
            stmt_ver = select(PolicyVersion).filter(PolicyVersion.is_active == True)
            if org_id:
                stmt_ver = stmt_ver.filter(PolicyVersion.org_id == org_id)
            
            policy = (await db.execute(stmt_ver.order_by(desc(PolicyVersion.created_at)).limit(1))).scalar_one_or_none()
            
            if policy:
                return {
                    "id": str(policy.id), 
                    "name": policy.name, 
                    "version": policy.version,
                    "rules": policy.config, 
                    "config": policy.config, 
                    "is_active": True,
                }
        except Exception as e:
            logger.error("policy_version_fetch_error", error=str(e))
    except Exception as e:
        logger.error("critical_policy_fetch_failure", error=str(e))
    
    # 3. Final safety fallback
    return {
        "id": "default",
        "name": "Global Default Policy",
        "version": 0,
        "rules": DEFAULT_POLICY_CONFIG,
        "config": DEFAULT_POLICY_CONFIG,
        "is_active": True,
    }


class ActivePolicyUpdate(BaseModel):
    name: str = "Custom Governance Policy"
    rules: dict


@router.post("/policies/active")
async def update_active_policy(body: ActivePolicyUpdate, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)):
    """
    Tweak the active governance policy.
    Creates a new active PolicyRule and deactivates others.
    """
    org_id = auth.org_id
    
    # 1. Deactivate current active rules for this org
    from sqlalchemy import update
    stmt = update(PolicyRule).where(PolicyRule.is_active == True)
    if org_id:
        stmt = stmt.filter(PolicyRule.org_id == org_id)
    await db.execute(stmt.values(is_active=False))
    
    # Also deactivate PolicyVersion for consistency if we are switching to PolicyRule
    stmt_v = update(PolicyVersion).where(PolicyVersion.is_active == True)
    if org_id:
        stmt_v = stmt_v.filter(PolicyVersion.org_id == org_id)
    await db.execute(stmt_v.values(is_active=False))

    # 2. Create new active rule
    new_rule = PolicyRule(
        id=uuid.uuid4(),
        org_id=org_id,
        name=body.name,
        rules_json=body.rules,
        is_active=True,
    )
    db.add(new_rule)
    
    # 3. Log action
    from app.db.models import AuditLog
    db.add(AuditLog(
        org_id=org_id,
        actor_key_id=auth.api_key_id,
        action="policy.update",
        resource_type="policy",
        resource_id=str(new_rule.id),
        details={"name": body.name, "rules": body.rules}
    ))
    
    await db.commit()
    return {"status": "success", "id": str(new_rule.id)}


@router.get("/policies/{policy_id}")
async def get_policy(policy_id: str, db: AsyncSession = Depends(get_db)):
    p = await db.get(PolicyVersion, policy_id)
    if not p:
        # Try PolicyRule
        p = await db.get(PolicyRule, policy_id)
    if not p:
        raise HTTPException(404, "Policy not found.")
    return {
        "id": str(p.id), "name": p.name, "version": getattr(p, "version", "N/A"),
        "config": getattr(p, "config", getattr(p, "rules_json", {})), "is_active": p.is_active,
        "notes": getattr(p, "notes", ""), "created_at": str(p.created_at),
    }
