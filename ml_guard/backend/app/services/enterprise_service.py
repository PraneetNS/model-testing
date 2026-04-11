"""
Enterprise Service Layer.
Aggregates data across scans, models, policies, and audit logs
for the organization-level governance control center.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func
from typing import Optional
from app.db.models import (
    ScanRecord, Model, PolicyVersion, PolicyRule, AuditLog,
    AlertRule, AlertEvent, Organization, LLMScanRecord, StreamDriftRecord,
)


async def get_enterprise_summary(db: AsyncSession, org_id: Optional[str] = None) -> dict:
    """
    Compute real-time enterprise-level summary metrics.
    All numbers come from the database — zero dummy data.
    """
    from sqlalchemy import select, func

    # ─── Total Models ───
    total_models = (await db.execute(select(func.count(Model.id)))).scalar() or 0

    # ─── Total Scans ───
    total_scans = (await db.execute(select(func.count(ScanRecord.id)))).scalar() or 0

    # ─── High Risk Scans ───
    high_risk_scans = (await db.execute(
        select(func.count(ScanRecord.id))
        .filter(ScanRecord.risk_level.in_(["HIGH", "CRITICAL"]))
    )).scalar() or 0

    # ─── Unique high-risk model IDs ───
    high_risk_model_ids = (await db.execute(
        select(func.count(func.distinct(ScanRecord.model_id)))
        .filter(ScanRecord.risk_level.in_(["HIGH", "CRITICAL"]))
    )).scalar() or 0

    # ─── Active Policies ───
    active_policies_v = (await db.execute(
        select(func.count(PolicyVersion.id)).filter(PolicyVersion.is_active == True)
    )).scalar() or 0
    active_policies_r = (await db.execute(
        select(func.count(PolicyRule.id)).filter(PolicyRule.is_active == True)
    )).scalar() or 0
    active_policies = active_policies_v + active_policies_r

    # ─── Total Policies ───
    total_policies_v = (await db.execute(select(func.count(PolicyVersion.id)))).scalar() or 0
    total_policies_r = (await db.execute(select(func.count(PolicyRule.id)))).scalar() or 0
    total_policies = total_policies_v + total_policies_r

    # ─── Average Governance Score ───
    avg_score_result = (await db.execute(
        select(func.avg(ScanRecord.governance_score)).filter(ScanRecord.governance_score.isnot(None))
    )).scalar()
    avg_governance_score = round(float(avg_score_result), 2) if avg_score_result else 0.0

    # ─── Min / Max Governance Score ───
    min_score = (await db.execute(
        select(func.min(ScanRecord.governance_score)).filter(ScanRecord.governance_score.isnot(None))
    )).scalar()
    max_score = (await db.execute(
        select(func.max(ScanRecord.governance_score)).filter(ScanRecord.governance_score.isnot(None))
    )).scalar()

    # ─── Gate Status Distribution ───
    gate_passed = (await db.execute(
        select(func.count(ScanRecord.id)).filter(ScanRecord.gate_status == "PASSED")
    )).scalar() or 0
    gate_warning = (await db.execute(
        select(func.count(ScanRecord.id)).filter(ScanRecord.gate_status == "WARNING")
    )).scalar() or 0
    gate_critical = (await db.execute(
        select(func.count(ScanRecord.id)).filter(ScanRecord.gate_status == "CRITICAL")
    )).scalar() or 0

    # ─── Alert Stats ───
    total_alert_rules = (await db.execute(select(func.count(AlertRule.id)))).scalar() or 0
    total_alert_events = (await db.execute(select(func.count(AlertEvent.id)))).scalar() or 0
    undelivered_alerts = (await db.execute(
        select(func.count(AlertEvent.id)).filter(AlertEvent.delivered == False)
    )).scalar() or 0

    # ─── LLM Scans ───
    total_llm_scans = (await db.execute(select(func.count(LLMScanRecord.id)))).scalar() or 0

    # ─── Organizations ───
    total_orgs = (await db.execute(select(func.count(Organization.id)))).scalar() or 0

    # ─── Recent Activity (last 10 audit log entries) ───
    recent_logs = (await db.execute(
        select(AuditLog).order_by(desc(AuditLog.created_at)).limit(10)
    )).scalars().all()
    recent_activity = [
        {
            "id": str(l.id),
            "action": l.action,
            "resource_type": l.resource_type,
            "resource_id": l.resource_id,
            "details": l.details,
            "created_at": str(l.created_at),
        }
        for l in recent_logs
    ]

    return {
        "total_models": total_models,
        "total_scans": total_scans,
        "high_risk_models": high_risk_model_ids,
        "high_risk_scans": high_risk_scans,
        "active_policies": active_policies,
        "total_policies": total_policies,
        "average_governance_score": avg_governance_score,
        "min_governance_score": round(float(min_score), 2) if min_score is not None else None,
        "max_governance_score": round(float(max_score), 2) if max_score is not None else None,
        "gate_distribution": {
            "passed": gate_passed,
            "warning": gate_warning,
            "critical": gate_critical,
        },
        "total_alert_rules": total_alert_rules,
        "total_alert_events": total_alert_events,
        "undelivered_alerts": undelivered_alerts,
        "total_llm_scans": total_llm_scans,
        "total_organizations": total_orgs,
        "recent_activity": recent_activity,
    }


async def get_scans_paginated(
    db: AsyncSession, page: int = 1, per_page: int = 20,
    sort_by: str = "created_at", sort_dir: str = "desc",
) -> dict:
    """Paginated scan history with sorting."""
    from sqlalchemy import select, func, desc, asc
    
    total = (await db.execute(select(func.count(ScanRecord.id)))).scalar() or 0

    # Sorting
    sort_col = getattr(ScanRecord, sort_by, ScanRecord.created_at)
    stmt = select(ScanRecord)
    if sort_dir == "asc":
        stmt = stmt.order_by(asc(sort_col))
    else:
        stmt = stmt.order_by(desc(sort_col))

    # Pagination
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page)
    scans = (await db.execute(stmt)).scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "items": [
            {
                "id": str(s.id),
                "model_id": str(s.model_id) if s.model_id else None,
                "model_name": s.model.name if s.model else "System Scan",
                "scan_type": s.scan_type,
                "governance_score": s.governance_score,
                "risk_score": s.risk_score,
                "risk_level": s.risk_level,
                "gate_status": s.gate_status,
                "checks_run": s.checks_run,
                "trigger_source": s.trigger_source,
                "duration_ms": s.duration_ms,
                "fairness_risk_score": s.fairness_risk_score,
                "bias_violation_flag": s.bias_violation_flag,
                "artifact_url": s.artifact_url,
                "training_dataset_url": getattr(s, "training_dataset_url", None),
                "validation_dataset_url": getattr(s, "validation_dataset_url", None),
                "created_at": str(s.created_at),
            }
            for s in scans
        ],
    }


async def get_models_paginated(
    db: AsyncSession, page: int = 1, per_page: int = 20,
) -> dict:
    """Paginated model registry."""
    from sqlalchemy import select, func, desc
    
    total = (await db.execute(select(func.count(Model.id)))).scalar() or 0
    offset = (page - 1) * per_page
    
    stmt = select(Model).order_by(desc(Model.created_at)).offset(offset).limit(per_page)
    models = (await db.execute(stmt)).scalars().all()

    # Enrich models with latest scan data
    items = []
    for m in models:
        scan_stmt = (
            select(ScanRecord)
            .filter(ScanRecord.model_id == str(m.id))
            .order_by(desc(ScanRecord.created_at))
            .limit(1)
        )
        latest_scan = (await db.execute(scan_stmt)).scalar_one_or_none()
        
        items.append({
            "id": str(m.id),
            "name": m.name,
            "provider": m.provider,
            "fingerprint": m.fingerprint,
            "version": m.version,
            "metadata": m.metadata_json,
            "artifact_url": m.artifact_url,
            "artifact_size": m.artifact_size,
            "artifact_storage_provider": m.artifact_storage_provider,
            "created_at": str(m.created_at),
            "latest_scan": {
                "governance_score": latest_scan.governance_score,
                "risk_level": latest_scan.risk_level,
                "gate_status": latest_scan.gate_status,
                "created_at": str(latest_scan.created_at),
            } if latest_scan else None,
        })

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "items": items,
    }


async def get_audit_logs_paginated(
    db: AsyncSession, page: int = 1, per_page: int = 30,
    org_id: Optional[str] = None,
) -> dict:
    """Paginated audit log trail."""
    from sqlalchemy import select, func, desc
    
    stmt_count = select(func.count(AuditLog.id))
    if org_id:
        stmt_count = stmt_count.filter(AuditLog.org_id == org_id)
    total = (await db.execute(stmt_count)).scalar() or 0
    
    offset = (page - 1) * per_page
    stmt = select(AuditLog).order_by(desc(AuditLog.created_at))
    if org_id:
        stmt = stmt.filter(AuditLog.org_id == org_id)
    stmt = stmt.offset(offset).limit(per_page)
    logs = (await db.execute(stmt)).scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "items": [
            {
                "id": str(l.id),
                "action": l.action,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "details": l.details,
                "user_id": str(l.user_id) if l.user_id else None,
                "ip_address": l.ip_address,
                "created_at": str(l.created_at),
            }
            for l in logs
        ],
    }
