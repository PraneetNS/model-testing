import uuid
import datetime
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import RedTeamSchedule, RedTeamRun, Model
from app.core.auth import AuthContext, require_role
from app.workers.tasks import run_red_team_task

router = APIRouter()

@router.post("/red-team/{model_id}/schedule")
async def schedule_red_team_endpoint(
    model_id: str,
    payload: Dict = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    """
    Creates or updates a scheduled red teaming penetration test.
    Input: {schedule_cron, attack_profile, notification_on_regression, enabled}
    """
    result = await db.execute(select(RedTeamSchedule).filter(RedTeamSchedule.model_id == model_id))
    sched = result.scalars().first()
    
    if not sched:
        sched = RedTeamSchedule(model_id=model_id)
        db.add(sched)
        
    sched.schedule_cron = payload.get("schedule_cron", sched.schedule_cron)
    sched.attack_profile = payload.get("attack_profile", sched.attack_profile)
    sched.notification_on_regression = payload.get("notification_on_regression", sched.notification_on_regression)
    sched.enabled = payload.get("enabled", True)
    
    await db.commit()
    return {
        "status": "scheduled", 
        "cron": sched.schedule_cron, 
        "profile": sched.attack_profile
    }

@router.post("/red-team/{model_id}/run-now")
async def run_red_team_now(
    model_id: str,
    profile: str = Body("standard", embed=True),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    """Triggers an immediate background red teaming run."""
    task = run_red_team_task.delay(model_id, profile)
    return {"task_id": task.id, "status": "queued"}

@router.get("/red-team/{model_id}/history")
async def get_red_team_history(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    """Returns the last 30 red teaming execution logs."""
    result = await db.execute(
        select(RedTeamRun).filter(RedTeamRun.model_id == model_id).order_by(RedTeamRun.run_at.desc()).limit(30)
    )
    runs = result.scalars().all()
    return [
        {
            "run_at": r.run_at,
            "profile": r.profile,
            "robustness_score": r.robustness_score,
            "attack_results": r.attack_results,
            "regressions_detected": r.regressions_detected
        } for r in runs
    ]

@router.get("/red-team/{model_id}/heatmap")
async def get_red_team_heatmap(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    """Returns a matrix of attack results for dashboard visualization."""
    result = await db.execute(
        select(RedTeamRun).filter(RedTeamRun.model_id == model_id).order_by(RedTeamRun.run_at.desc())
    )
    last_run = result.scalars().first()
    if not last_run:
        return []
        
    heatmap = []
    for attack, res in last_run.attack_results.items():
        severity = "MEDIUM"
        if isinstance(res, dict):
            if res.get("vulnerable") or res.get("risk"):
                severity = "CRITICAL"
            elif res.get("success_rate", 0) > 0.2:
                severity = "HIGH"
                
        heatmap.append({
            "attack_type": attack,
            "severity": severity,
            "success_rate": res.get("success_rate", 0.0) if isinstance(res, dict) else 1.0 if severity == "CRITICAL" else 0.0
        })
    return heatmap

@router.post("/red-team/{model_id}/approve-baseline")
async def approve_baseline(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    """Sets the most recent robustness score as the baseline for future regression checks."""
    r_result = await db.execute(
        select(RedTeamRun).filter(RedTeamRun.model_id == model_id).order_by(RedTeamRun.run_at.desc())
    )
    last_run = r_result.scalars().first()
    if not last_run:
        raise HTTPException(404, "No recent red team runs found to use as baseline")
        
    s_result = await db.execute(select(RedTeamSchedule).filter(RedTeamSchedule.model_id == model_id))
    sched = s_result.scalars().first()
    if not sched:
        sched = RedTeamSchedule(model_id=model_id)
        db.add(sched)
        
    sched.baseline_robustness_score = last_run.robustness_score
    await db.commit()
    
    return {"status": "baseline_approved", "new_baseline": last_run.robustness_score}
