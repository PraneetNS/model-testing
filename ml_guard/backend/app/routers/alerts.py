"""
Alert Engine Router.
Create alert rules, evaluate them against scan results, trigger notifications.
"""
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from pydantic import BaseModel
from typing import List, Optional
from app.db.session import get_db
from app.db.models import AlertRule, AlertEvent, ScanRecord, AuditLog, Model, utcnow

router = APIRouter()


class AlertRuleCreate(BaseModel):
    name: str
    condition: dict  # {"metric": "governance_score", "op": "<", "value": 70}
    channels: List[str] = ["webhook"]
    webhook_url: str = ""
    org_id: str = ""


@router.post("/alerts/rules")
async def create_rule(body: AlertRuleCreate, db: AsyncSession = Depends(get_db)):
    rule = AlertRule(
        org_id=body.org_id or None,
        name=body.name,
        condition=body.condition,
        channels=body.channels,
        webhook_url=body.webhook_url,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    db.add(AuditLog(
        org_id=body.org_id or None,
        action="alert_rule.create",
        resource_type="alert_rule",
        resource_id=str(rule.id),
        details={"name": body.name, "condition": body.condition},
    ))
    await db.commit()
    return {"id": str(rule.id), "name": rule.name, "condition": rule.condition, "channels": rule.channels}


@router.get("/alerts/rules")
async def list_rules(org_id: str = "", db: AsyncSession = Depends(get_db)):
    stmt = select(AlertRule)
    if org_id:
        stmt = stmt.filter(AlertRule.org_id == org_id)
    rules = (await db.execute(stmt)).scalars().all()
    return [
        {"id": str(r.id), "name": r.name, "condition": r.condition, "channels": r.channels, "is_active": r.is_active}
        for r in rules
    ]


@router.delete("/alerts/rules/{rule_id}")
async def delete_rule(rule_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AlertRule).filter(AlertRule.id == rule_id))
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found.")
    await db.delete(rule)
    await db.commit()
    return {"deleted": True}


# ─── EVALUATE ALERTS AGAINST A SCAN ───
def _op(actual, op_str, threshold):
    ops = {
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
    }
    fn = ops.get(op_str)
    if not fn:
        return False
    try:
        return fn(float(actual), float(threshold))
    except:
        return False


def _extract_metric(scan_results: dict, metric_name: str):
    """Try to find metric value from scan results."""
    if metric_name == "governance_score":
        gov = scan_results.get("governance", {})
        return gov.get("governance_score") if isinstance(gov, dict) else None
    if metric_name == "robustness_score":
        return scan_results.get("robustness_score")
    if metric_name == "stability_score":
        mc = scan_results.get("stress_results", {}).get("monte_carlo_stability", {})
        return mc.get("stability_score") if isinstance(mc, dict) else None
    metrics = scan_results.get("metrics", {})
    return metrics.get(metric_name)


@router.post("/alerts/evaluate/{scan_id}")
async def evaluate_alerts(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Evaluate all active alert rules against a scan result."""
    scan = await db.get(ScanRecord, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found.")

    rules = (await db.execute(select(AlertRule).filter(AlertRule.is_active == True))).scalars().all()
    triggered = []

    for rule in rules:
        cond = rule.condition or {}
        metric_name = cond.get("metric")
        op_str = cond.get("op")
        threshold = cond.get("value")
        if not metric_name or not op_str or threshold is None:
            continue

        actual = _extract_metric(scan.results_json or {}, metric_name)
        if actual is None:
            continue

        if _op(actual, op_str, threshold):
            event = AlertEvent(
                rule_id=str(rule.id),
                scan_id=str(scan.id),
                severity="CRITICAL" if op_str in ("<", "<=") else "WARNING",
                message=f"Alert '{rule.name}': {metric_name} = {actual} {op_str} {threshold}",
            )
            db.add(event)
            triggered.append({
                "rule": rule.name,
                "metric": metric_name,
                "actual": actual,
                "threshold": threshold,
                "severity": event.severity,
            })

            # Webhook delivery
            if "webhook" in rule.channels and rule.webhook_url:
                try:
                    import requests
                    requests.post(rule.webhook_url, json={
                        "alert": rule.name,
                        "metric": metric_name,
                        "actual": actual,
                        "threshold": threshold,
                        "scan_id": str(scan.id),
                        "severity": event.severity,
                    }, timeout=5)
                    event.delivered = True
                except:
                    pass

            # ML Guard Outbound Plugins (Slack/Teams)
            try:
                from app.tasks.notifications import dispatch_alert
                dispatch_alert.delay(str(scan.model_id), {
                    "severity": event.severity,
                    "breach_type": rule.name,
                    "current_score": actual,
                    "threshold": threshold,
                    "verdict": "FAILED" if event.severity == "CRITICAL" else "WARNING",
                    "dashboard_url": "http://localhost:3000"
                })
            except Exception as e:
                print(f"Notification dispatch failed: {str(e)}")

    await db.commit()
    return {"scan_id": str(scan.id), "alerts_triggered": len(triggered), "details": triggered}


@router.get("/alerts/events")
async def list_events(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """List recent alerts, ordered by model risk tier (Critical first)."""
    # Use a CASE statement for custom sorting of risk tiers
    from sqlalchemy import case
    
    risk_order = case(
        (Model.risk_tier == 'critical', 1),
        (Model.risk_tier == 'high', 2),
        (Model.risk_tier == 'medium', 3),
        (Model.risk_tier == 'low', 4),
        else_=5
    )

    stmt = (
        select(AlertEvent, Model.risk_tier, Model.name.label("model_name"))
        .outerjoin(ScanRecord, AlertEvent.scan_id == ScanRecord.id)
        .outerjoin(Model, ScanRecord.model_id == Model.id)
        .order_by(risk_order, desc(AlertEvent.created_at))
        .limit(limit)
    )
    
    results = (await db.execute(stmt)).all()
    
    return [
        {
            "id": str(e.id), 
            "rule_id": str(e.rule_id),
            "severity": e.severity, 
            "message": e.message,
            "delivered": e.delivered, 
            "created_at": str(e.created_at),
            "risk_tier": risk_tier or "unassigned",
            "model_name": model_name or "Unknown"
        }
        for e, risk_tier, model_name in results
    ]


# ─── Summary endpoint (required by dashboard alert badge) ──────────────────

@router.get("/alerts/summary")
async def get_alerts_summary(db: AsyncSession = Depends(get_db)):
    """
    Returns aggregate counts for the dashboard notification bell.
    Response: {unread_count, critical_count, alerts_last_24h, recent}
    """
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func
    from app.db.models import SecurityAlert

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last_24h = now - timedelta(hours=24)

    # Alert events (rule-based)
    unread_events = (await db.execute(
        select(func.count(AlertEvent.id)).where(AlertEvent.delivered == False)
    )).scalar() or 0

    critical_events = (await db.execute(
        select(func.count(AlertEvent.id)).where(AlertEvent.severity == "CRITICAL")
    )).scalar() or 0

    events_24h = (await db.execute(
        select(func.count(AlertEvent.id))
        .where(AlertEvent.created_at >= last_24h)
    )).scalar() or 0

    # Security alerts (middleware-detected)
    security_24h = 0
    try:
        security_24h = (await db.execute(
            select(func.count(SecurityAlert.id))
            .where(SecurityAlert.created_at >= last_24h)
        )).scalar() or 0
    except Exception:
        pass

    # Most recent alert events
    recent_stmt = (
        select(AlertEvent)
        .order_by(desc(AlertEvent.created_at))
        .limit(5)
    )
    recent_events = (await db.execute(recent_stmt)).scalars().all()

    return {
        "unread_count": unread_events,
        "critical_count": critical_events,
        "alerts_last_24h": events_24h + security_24h,
        "security_alerts_24h": security_24h,
        "recent": [
            {
                "id": str(e.id),
                "severity": e.severity,
                "message": e.message,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_events
        ],
    }


@router.get("/alerts")
async def list_alerts(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
    unread_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Paginated list of all alert events with optional filters."""
    from sqlalchemy import func
    stmt = select(AlertEvent)
    count_stmt = select(func.count(AlertEvent.id))

    if severity:
        stmt = stmt.where(AlertEvent.severity == severity.upper())
        count_stmt = count_stmt.where(AlertEvent.severity == severity.upper())
    if unread_only:
        stmt = stmt.where(AlertEvent.delivered == False)
        count_stmt = count_stmt.where(AlertEvent.delivered == False)

    total = (await db.execute(count_stmt)).scalar() or 0
    events = (await db.execute(
        stmt.order_by(desc(AlertEvent.created_at)).offset(offset).limit(limit)
    )).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "id": str(e.id),
                "rule_id": str(e.rule_id),
                "scan_id": str(e.scan_id) if e.scan_id else None,
                "severity": e.severity,
                "message": e.message,
                "delivered": e.delivered,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


@router.put("/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: str, db: AsyncSession = Depends(get_db)):
    """Mark an alert event as read/delivered."""
    import uuid
    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid alert ID format.")

    event = await db.get(AlertEvent, alert_uuid)
    if not event:
        raise HTTPException(status_code=404, detail="Alert not found.")
    event.delivered = True
    await db.commit()
    return {"id": alert_id, "delivered": True}


class InternalAlertCreate(BaseModel):
    severity: str = "MEDIUM"   # LOW | MEDIUM | HIGH | CRITICAL
    message: str
    source: str = "platform"  # drift | performance | contract | platform
    model_id: Optional[str] = None
    scan_id: Optional[str] = None
    rule_id: Optional[str] = None


@router.post("/alerts/internal")
async def create_internal_alert(
    body: InternalAlertCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Platform-internal alert creation endpoint.
    Called by Celery tasks, drift scanner, contract engine, etc.
    Finds the most relevant alert rule or uses a sentinel rule.
    """
    # Find or create a sentinel rule for this source
    sentinel_name = f"[{body.source.upper()}] Internal Alert Rule"
    rule_stmt = select(AlertRule).where(AlertRule.name == sentinel_name).limit(1)
    rule = (await db.execute(rule_stmt)).scalars().first()

    if not rule:
        rule = AlertRule(
            name=sentinel_name,
            condition={"metric": body.source, "op": "trigger", "value": 0},
            channels=["internal"],
            is_active=True,
        )
        db.add(rule)
        await db.flush()

    # Resolve rule_id override
    rule_id = rule.id
    if body.rule_id:
        try:
            rule_id = uuid.UUID(body.rule_id)
        except ValueError:
            pass

    event = AlertEvent(
        rule_id=rule_id,
        scan_id=uuid.UUID(body.scan_id) if body.scan_id else None,
        severity=body.severity.upper(),
        message=body.message,
        delivered=False,
    )
    db.add(event)
    await db.commit()

    return {
        "id": str(event.id),
        "severity": event.severity,
        "message": event.message,
        "rule_id": str(rule_id),
    }
