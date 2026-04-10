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
    # ─── Total Models ───
    q_models = db.query(Model)
    total_models = q_models.count()

    # ─── Total Scans ───
    q_scans = db.query(ScanRecord)
    total_scans = q_scans.count()

    # ─── High Risk Models ───
    high_risk_scans = db.query(ScanRecord).filter(
        ScanRecord.risk_level.in_(["HIGH", "CRITICAL"])
    ).count()

    # ─── Unique high-risk model IDs ───
    high_risk_model_ids = db.query(ScanRecord.model_id).filter(
        ScanRecord.risk_level.in_(["HIGH", "CRITICAL"])
    ).distinct().count()

    # ─── Active Policies ───
    active_policies_v = db.query(PolicyVersion).filter(PolicyVersion.is_active == True).count()
    active_policies_r = db.query(PolicyRule).filter(PolicyRule.is_active == True).count()
    active_policies = active_policies_v + active_policies_r

    # ─── Total Policies ───
    total_policies = db.query(PolicyVersion).count() + db.query(PolicyRule).count()

    # ─── Average Governance Score ───
    avg_score_result = db.query(func.avg(ScanRecord.governance_score)).filter(
        ScanRecord.governance_score.isnot(None)
    ).scalar()
    avg_governance_score = round(float(avg_score_result), 2) if avg_score_result else 0.0

    # ─── Min / Max Governance Score ───
    min_score = db.query(func.min(ScanRecord.governance_score)).filter(
        ScanRecord.governance_score.isnot(None)
    ).scalar()
    max_score = db.query(func.max(ScanRecord.governance_score)).filter(
        ScanRecord.governance_score.isnot(None)
    ).scalar()

    # ─── Gate Status Distribution ───
    gate_passed = db.query(ScanRecord).filter(ScanRecord.gate_status == "PASSED").count()
    gate_warning = db.query(ScanRecord).filter(ScanRecord.gate_status == "WARNING").count()
    gate_critical = db.query(ScanRecord).filter(ScanRecord.gate_status == "CRITICAL").count()

    # ─── Alert Stats ───
    total_alert_rules = db.query(AlertRule).count()
    total_alert_events = db.query(AlertEvent).count()
    undelivered_alerts = db.query(AlertEvent).filter(AlertEvent.delivered == False).count()

    # ─── LLM Scans ───
    total_llm_scans = db.query(LLMScanRecord).count()

    # ─── Organizations ───
    total_orgs = db.query(Organization).count()

    # ─── Recent Activity (last 10 audit log entries) ───
    recent_logs = (await db.execute(select(AuditLog).order_by(desc(AuditLog.created_at)).limit(10))).scalars().all()
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
        "min_governance_score": round(float(min_score), 2) if min_score else None,
        "max_governance_score": round(float(max_score), 2) if max_score else None,
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
    q = db.query(ScanRecord)
    total = q.count()

    # Sorting
    sort_col = getattr(ScanRecord, sort_by, ScanRecord.created_at)
    if sort_dir == "asc":
        q = q.order_by(sort_col.asc())
    else:
        q = q.order_by(sort_col.desc())

    # Pagination
    offset = (page - 1) * per_page
    scans = q.offset(offset).limit(per_page).all()

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
                "training_dataset_url": s.training_dataset_url,
                "validation_dataset_url": s.validation_dataset_url,
                "created_at": str(s.created_at),
            }
            for s in scans
        ],
    }


async def get_models_paginated(
    db: AsyncSession, page: int = 1, per_page: int = 20,
) -> dict:
    """Paginated model registry."""
    q = db.query(Model).order_by(desc(Model.created_at))
    total = q.count()
    offset = (page - 1) * per_page
    models = q.offset(offset).limit(per_page).all()

    # Enrich models with latest scan data
    items = []
    for m in models:
        latest_scan = (
            db.query(ScanRecord)
            .filter(ScanRecord.model_id == str(m.id))
            .order_by(desc(ScanRecord.created_at))
            .first()
        )
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
    q = db.query(AuditLog).order_by(desc(AuditLog.created_at))
    if org_id:
        q = q.filter(AuditLog.org_id == org_id)
    total = q.count()
    offset = (page - 1) * per_page
    logs = q.offset(offset).limit(per_page).all()

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
