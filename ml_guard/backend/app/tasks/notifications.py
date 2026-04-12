import asyncio
from typing import Any, Dict, Optional
from app.core.celery_app import celery_app
from app.db.session import AsyncSessionLocal
from app.db.models import NotificationConfig, Model
from sqlalchemy.future import select
from ml_guard.plugins.slack_notifier import SlackNotifier, AlertSchema
from ml_guard.plugins.teams_notifier import TeamsNotifier
import logging

logger = logging.getLogger(__name__)

async def _dispatch_notifications(model_id: str, alert_data: Dict[str, Any]):
    async with AsyncSessionLocal() as db:
        config = (await db.execute(select(NotificationConfig).filter(NotificationConfig.model_id == model_id))).scalar_one_or_none()
        if not config:
            return

        model = (await db.execute(select(Model).filter(Model.id == model_id))).scalar_one_or_none()
        model_name = model.name if model else "Unknown Model"
        
        # Prepare alert schema
        severity = alert_data.get("severity", "INFO")
        if severity not in (config.notify_on or []):
            if severity != "PREDICTIVE_BREACH" or "PREDICTIVE_BREACH" not in (config.notify_on or []):
                # Check for score decay separately or match severity
                if severity != "SCORE_DECAY" or "SCORE_DECAY" not in (config.notify_on or []):
                    return

        alert = AlertSchema(
            model_id=str(model_id),
            model_name=model_name,
            breach_type=alert_data.get("breach_type", "Rule Triggered"),
            severity=severity,
            current_score=alert_data.get("current_score", 0.0),
            threshold=alert_data.get("threshold", 0.0),
            breaches_in_window=alert_data.get("breaches_in_window", 0),
            verdict=alert_data.get("verdict", "N/A"),
            is_predictive=alert_data.get("is_predictive", False),
            prediction_horizon=alert_data.get("prediction_horizon")
        )

        # Slack
        if config.slack_webhook_url:
            slack = SlackNotifier(config.slack_webhook_url, config.slack_channel or "#alerts")
            if severity == "SCORE_DECAY":
                await slack.send_score_decay_alert(
                    str(model_id), 
                    alert_data.get("old_score", 0.0),
                    alert_data.get("new_score", 0.0),
                    alert_data.get("delta", 0.0),
                    model_name
                )
            else:
                await slack.send_breach_alert(alert)

        # Teams
        if config.teams_webhook_url:
            teams = TeamsNotifier(config.teams_webhook_url)
            if severity == "SCORE_DECAY":
                await teams.send_score_decay_alert(
                    str(model_id),
                    alert_data.get("old_score", 0.0),
                    alert_data.get("new_score", 0.0),
                    alert_data.get("delta", 0.0),
                    model_name
                )
            else:
                await teams.send_breach_alert(alert)

@celery_app.task(name="app.tasks.notifications.dispatch_alert")
def dispatch_alert(model_id: str, alert_data: Dict[str, Any]):
    """Fire-and-forget task to send notifications."""
    asyncio.run(_dispatch_notifications(model_id, alert_data))
