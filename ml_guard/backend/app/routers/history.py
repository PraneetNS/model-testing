"""
Scan History + Model Comparison Router
- List scan history with governance score trends
- Compare two scans side by side
- Model registry lookup by fingerprint
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from app.db.session import get_db
from app.db.models import ScanRecord, Model, AuditLog, utcnow

router = APIRouter()


# ══════════════════════════
# SCAN HISTORY
# ══════════════════════════
@router.get("/history")
async def list_scans(
    model_id: str = "",
    scan_type: str = "",
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(ScanRecord).order_by(desc(ScanRecord.created_at))
    if model_id:
        stmt = stmt.filter(ScanRecord.model_id == model_id)
    if scan_type:
        stmt = stmt.filter(ScanRecord.scan_type == scan_type)
    
    scans = (await db.execute(stmt.limit(limit))).scalars().all()
    
    return [
        {
            "id": str(s.id),
            "model_id": str(s.model_id),
            "scan_type": s.scan_type,
            "governance_score": s.governance_score,
            "gate_status": s.gate_status,
            "checks_run": s.checks_run,
            "trigger_source": s.trigger_source,
            "duration_ms": s.duration_ms,
            "created_at": str(s.created_at),
        }
        for s in scans
    ]


@router.get("/history/{scan_id}")
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(ScanRecord, scan_id)
    if not s:
        raise HTTPException(404, "Scan record not found.")
    return {
        "id": str(s.id),
        "model_id": str(s.model_id),
        "scan_type": s.scan_type,
        "governance_score": s.governance_score,
        "gate_status": s.gate_status,
        "checks_run": s.checks_run,
        "results_json": s.results_json,
        "trigger_source": s.trigger_source,
        "duration_ms": s.duration_ms,
        "created_at": str(s.created_at),
    }


# ══════════════════════════
# GOVERNANCE SCORE TRAJECTORY
# ══════════════════════════
@router.get("/history/trajectory/{model_id}")
async def governance_trajectory(model_id: str, limit: int = 20, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(ScanRecord)
        .filter(ScanRecord.model_id == model_id)
        .order_by(ScanRecord.created_at)
        .limit(limit)
    )
    scans = (await db.execute(stmt)).scalars().all()
    
    points = [
        {"scan_id": str(s.id), "score": s.governance_score, "gate": s.gate_status, "ts": str(s.created_at)}
        for s in scans if s.governance_score is not None
    ]
    trend = "stable"
    if len(points) >= 3:
        recent = [p["score"] for p in points[-3:]]
        if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
            trend = "declining"
        elif all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
            trend = "improving"
    return {"model_id": model_id, "data_points": points, "trend": trend}


# ══════════════════════════
# MODEL COMPARISON
# ══════════════════════════
@router.get("/compare")
async def compare_scans(
    scan_a: str = Query(..., description="Scan ID A"),
    scan_b: str = Query(..., description="Scan ID B"),
    db: AsyncSession = Depends(get_db)
):
    a = await db.get(ScanRecord, scan_a)
    b = await db.get(ScanRecord, scan_b)
    if not a or not b:
        raise HTTPException(404, "One or both scan IDs not found.")

    results_a = a.results_json or {}
    results_b = b.results_json or {}

    def _safe_get(d, *keys):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d

    def _delta(val_a, val_b):
        if val_a is None or val_b is None:
            return None
        try:
            return round(float(val_b) - float(val_a), 6)
        except:
            return None

    comparison = {
        "scan_a": {"id": str(a.id), "model_id": str(a.model_id), "score": a.governance_score, "gate": a.gate_status, "created_at": str(a.created_at)},
        "scan_b": {"id": str(b.id), "model_id": str(b.model_id), "score": b.governance_score, "gate": b.gate_status, "created_at": str(b.created_at)},
        "governance_delta": _delta(a.governance_score, b.governance_score),
        "metrics_comparison": {},
        "drift_comparison": {},
    }

    # Compare top-level metrics
    metrics_a = results_a.get("metrics", {})
    metrics_b = results_b.get("metrics", {})
    all_metric_keys = set(list(metrics_a.keys()) + list(metrics_b.keys()))
    for k in all_metric_keys:
        comparison["metrics_comparison"][k] = {
            "scan_a": metrics_a.get(k),
            "scan_b": metrics_b.get(k),
            "delta": _delta(metrics_a.get(k), metrics_b.get(k)),
        }

    # Compare drift
    drift_a = results_a.get("drift", {})
    drift_b = results_b.get("drift", {})
    all_features = set(list(drift_a.keys()) + list(drift_b.keys()))
    for f in list(all_features)[:20]:
        fa = drift_a.get(f, {})
        fb = drift_b.get(f, {})
        comparison["drift_comparison"][f] = {
            "psi_a": fa.get("PSI"),
            "psi_b": fb.get("PSI"),
            "jsd_a": fa.get("JSD"),
            "jsd_b": fb.get("JSD"),
            "psi_delta": _delta(fa.get("PSI"), fb.get("PSI")),
        }

    return comparison


# ══════════════════════════
# MODEL REGISTRY
# ══════════════════════════
@router.get("/models")
async def list_models(project_id: str = "", limit: int = 50, db: AsyncSession = Depends(get_db)):
    stmt = select(Model).order_by(desc(Model.created_at))
    if project_id:
        stmt = stmt.filter(Model.project_id == project_id)
    
    models = (await db.execute(stmt.limit(limit))).scalars().all()
    
    return {
        "items": [
            {
                "id": str(m.id), "name": m.name, "provider": m.provider,
                "fingerprint": m.fingerprint, "version": m.version,
                "metadata": m.metadata_json, "created_at": str(m.created_at),
            }
            for m in models
        ]
    }


@router.get("/models/by-fingerprint/{fingerprint}")
async def find_by_fingerprint(fingerprint: str, db: AsyncSession = Depends(get_db)):
    models = (await db.execute(select(Model).filter(Model.fingerprint == fingerprint))).scalars().all()
    if not models:
        return {"found": False, "message": "No model with this fingerprint has been evaluated."}
    return {
        "found": True,
        "models": [
            {"id": str(m.id), "name": m.name, "version": m.version, "created_at": str(m.created_at)}
            for m in models
        ],
    }


# ══════════════════════════
# AUDIT LOGS
# ══════════════════════════
@router.get("/audit-logs")
async def list_audit_logs(org_id: str = "", limit: int = 100, db: AsyncSession = Depends(get_db)):
    stmt = select(AuditLog).order_by(desc(AuditLog.created_at))
    if org_id:
        stmt = stmt.filter(AuditLog.org_id == org_id)
    
    logs = (await db.execute(stmt.limit(limit))).scalars().all()
    
    return [
        {
            "id": str(l.id), "action": l.action,
            "resource_type": l.resource_type,
            "resource_id": l.resource_id,
            "details": l.details,
            "created_at": str(l.created_at),
        }
        for l in logs
    ]
