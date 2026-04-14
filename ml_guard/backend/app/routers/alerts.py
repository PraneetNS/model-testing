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
from app.db.models import AlertRule, AlertEvent, ScanRecord, AuditLog, utcnow

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
    scan = db.get(ScanRecord, scan_id)
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
    events = (await db.execute(select(AlertEvent).order_by(desc(AlertEvent.created_at)).limit(limit))).scalars().all()
    return [
        {
            "id": str(e.id), "rule_id": str(e.rule_id),
            "severity": e.severity, "message": e.message,
            "delivered": e.delivered, "created_at": str(e.created_at),
        }
        for e in events
    ]
