from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel
import uuid

from app.db.session import get_db
from app.db.models import NotificationConfig, Model
from app.tasks.notifications import dispatch_alert

router = APIRouter()

class NotificationConfigSchema(BaseModel):
    model_id: str
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    teams_webhook_url: Optional[str] = None
    notify_on: List[str] = ["CRITICAL", "HIGH", "PREDICTIVE_BREACH", "SCORE_DECAY"]

@router.post("/notifications/config")
async def update_notification_config(body: NotificationConfigSchema, db: AsyncSession = Depends(get_db)):
    config = (await db.execute(select(NotificationConfig).filter(NotificationConfig.model_id == body.model_id))).scalar_one_or_none()
    if not config:
        config = NotificationConfig(model_id=body.model_id)
        db.add(config)
    
    config.slack_webhook_url = body.slack_webhook_url
    config.slack_channel = body.slack_channel
    config.teams_webhook_url = body.teams_webhook_url
    config.notify_on = body.notify_on
    
    await db.commit()
    return {"message": "Notification config updated."}

@router.get("/notifications/config/{model_id}")
async def get_notification_config(model_id: str, db: AsyncSession = Depends(get_db)):
    config = (await db.execute(select(NotificationConfig).filter(NotificationConfig.model_id == model_id))).scalar_one_or_none()
    if not config:
        raise HTTPException(404, "Notification config not found.")
    return config

@router.post("/notifications/test")
async def test_notifications(model_id: str, db: AsyncSession = Depends(get_db)):
    config = (await db.execute(select(NotificationConfig).filter(NotificationConfig.model_id == model_id))).scalar_one_or_none()
    if not config:
        raise HTTPException(404, "Notification config not found for this model.")
    
    # Send a dummy critical alert as test
    alert_data = {
        "severity": "CRITICAL",
        "breach_type": "TEST_ALERT_Breach",
        "current_score": 0.5432,
        "threshold": 0.8000,
        "breaches_in_window": 12,
        "verdict": "FAILED",
        "dashboard_url": "http://localhost:3000"
    }
    
    dispatch_alert.delay(str(model_id), alert_data)
    
    return {"message": "Test notification dispatched to configured channels."}
