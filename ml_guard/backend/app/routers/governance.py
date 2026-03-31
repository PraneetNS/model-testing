"""
governance.py — ML Guard Full Governance Router

Provides the complete governance API surface for ML Guard v7.2.
Incorporates composite score computing, certificate generation, and CI/CD gates.
"""
from __future__ import annotations

import structlog
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.certificate_engine import CertificateEngine
from app.services.governance_engine import GovernanceEngine, GovernanceScoreResult

router = APIRouter()
logger = structlog.get_logger(__name__)

_governance_engine = GovernanceEngine()
_cert_engine = CertificateEngine()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class CertifyRequest(BaseModel):
    force_regenerate: bool = False


class GateRequest(BaseModel):
    """Synchronous CI/CD policy gate check."""
    policy_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None


class RevokeRequest(BaseModel):
    reason: str


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _score_to_dict(result: GovernanceScoreResult) -> Dict[str, Any]:
    return {
        "model_id": result.model_id,
        "overall_score": result.overall_score,
        "live_score": result.live_score,
        "verdict": result.verdict,
        "component_scores": result.component_scores,
        "component_weights": result.component_weights,
        "drift_penalty": result.drift_penalty,
        "perf_penalty": result.perf_penalty,
        "data_freshness_hours": result.data_freshness_hours,
        "computed_at": result.computed_at.isoformat(),
        "recommendations": result.recommendations,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/governance/{model_id}/score")
async def get_governance_score(
    model_id: str = Path(..., description="Model UUID or name"),
    db: Session = Depends(get_db),
):
    """
    Compute the current governance score for a model.
    Returns audit-derived score AND live score with drift/perf decay applied.
    """
    try:
        result = _governance_engine.compute_score(model_id=model_id, db=db)
        return _score_to_dict(result)
    except Exception as e:
        logger.error("governance_score_failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Score computation failed: {str(e)}")


@router.get("/governance/{model_id}/score/live")
async def get_live_governance_score(
    model_id: str = Path(..., description="Model UUID or name"),
    db: Session = Depends(get_db),
):
    """
    Returns the live governance score with real-time drift/performance decay.
    """
    from app.db.models import DriftReport, PerformanceSnapshot, ScanRecord

    base_score = 75.0
    try:
        last_scan = (
            db.query(ScanRecord)
            .filter(ScanRecord.model_id == model_id)
            .order_by(ScanRecord.created_at.desc())
            .first()
        )
        if last_scan and last_scan.governance_score:
            base_score = last_scan.governance_score
    except Exception:
        pass

    last_drift = (
        db.query(DriftReport)
        .filter(DriftReport.model_id == model_id)
        .order_by(DriftReport.created_at.desc())
        .first()
    )

    last_perf = (
        db.query(PerformanceSnapshot)
        .filter(PerformanceSnapshot.model_id == model_id)
        .order_by(PerformanceSnapshot.computed_at.desc())
        .first()
    )

    live_score = _governance_engine.compute_live_score(base_score, last_drift, last_perf)

    drift_penalty = min(0.30, float(last_drift.overall_drift_score or 0)) if last_drift else 0.0
    perf_penalty = 0.0
    if last_perf and last_perf.degradation_report:
        acc = last_perf.degradation_report.get("accuracy", {})
        if acc:
            delta = acc.get("delta", 0) or 0
            perf_penalty = max(0.0, -float(delta) * 2)

    return {
        "model_id": model_id,
        "base_audit_score": base_score,
        "live_score": live_score,
        "drift_penalty": round(drift_penalty, 4),
        "perf_penalty": round(perf_penalty, 4),
        "last_drift_at": last_drift.created_at.isoformat() if last_drift else None,
        "last_perf_at": last_perf.computed_at.isoformat() if last_perf else None,
        "drift_detected": last_drift.drift_detected if last_drift else None,
        "computed_at": datetime.utcnow().isoformat(),
    }


@router.post("/governance/{model_id}/certify")
async def certify_model(
    model_id: str = Path(..., description="Model UUID or name"),
    req: CertifyRequest = CertifyRequest(),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """
    Trigger full governance audit and generate a compliance certificate.
    """
    try:
        result = _governance_engine.compute_score(model_id=model_id, db=db)
    except Exception as e:
        logger.error("governance_certify_failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Governance computation failed: {str(e)}")

    try:
        report_card = _cert_engine.generate_report_card(
            model_id=model_id,
            governance_result=result,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("cert_generation_failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Certificate generation failed: {str(e)}")

    download_url = f"/api/v1/governance/verify/{report_card.cert_hash}"

    return {
        "model_id": model_id,
        "cert_hash": str(report_card.cert_hash),
        "verdict": report_card.verdict,
        "overall_score": report_card.overall_score,
        "live_score": result.live_score,
        "issued_at": report_card.issued_at.isoformat() if report_card.issued_at else None,
        "download_url": download_url,
        "message": f"Certificate generated. Share {download_url} with auditors.",
    }


@router.get("/governance/verify/{cert_hash}")
async def verify_certificate(
    cert_hash: str = Path(..., description="SHA-256 certificate hash"),
    db: Session = Depends(get_db),
):
    """
    PUBLIC endpoint — no authentication required.
    """
    validity = _cert_engine.check_certificate_validity(cert_hash=cert_hash, db=db)
    return {
        "cert_hash": validity.cert_hash,
        "valid": validity.valid,
        "still_compliant": validity.still_compliant,
        "issued_at": validity.issued_at,
        "verdict": validity.verdict,
        "overall_score": validity.overall_score,
        "is_revoked": validity.is_revoked,
        "revocation_reason": validity.revocation_reason,
        "drift_events_since_issue": validity.drift_events_since_issue,
        "message": validity.message,
        "verified_at": datetime.utcnow().isoformat(),
    }


@router.get("/governance/status")
async def governance_status():
    """Health check endpoint for governance module."""
    return {
        "module": "governance",
        "status": "active",
        "version": "7.2.0"
    }


@router.post("/governance/{model_id}/gate")
async def synchronous_gate_check(
    model_id: str = Path(..., description="Model UUID or name"),
    req: GateRequest = GateRequest(),
    db: Session = Depends(get_db),
):
    """
    SYNCHRONOUS CI/CD policy gate check.
    """
    try:
        result = _governance_engine.compute_score(model_id=model_id, db=db)
    except Exception as e:
        logger.error("gate_computation_failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Gate computation failed: {str(e)}")

    all_metrics: Dict[str, float] = {
        "governance_score": result.live_score,
        **result.component_scores,
        **(req.metrics or {}),
    }

    policy_config = req.policy_config or {}
    if not policy_config:
        try:
            from app.db.models import PolicyVersion
            active_policy = (
                db.query(PolicyVersion)
                .filter(PolicyVersion.is_active == True)
                .order_by(PolicyVersion.created_at.desc())
                .first()
            )
            if active_policy:
                policy_config = active_policy.config or {}
        except Exception:
            pass

    class _Policy:
        config = policy_config

    gate_results = _governance_engine.check_policy_gates(all_metrics, _Policy())

    failures = [g for g in gate_results if g.verdict == "FAIL"]
    warnings = [g for g in gate_results if g.verdict == "WARN"]
    passed = len(failures) == 0

    response = {
        "model_id": model_id,
        "passed": passed,
        "score": result.live_score,
        "verdict": result.verdict,
        "gate_results": [asdict(g) for g in gate_results],
        "failures": [g.message for g in failures],
        "warnings": [g.message for g in warnings],
        "checked_at": datetime.utcnow().isoformat(),
    }

    if not passed:
        raise HTTPException(status_code=422, detail=response)

    return response
