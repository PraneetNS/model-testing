import requests
import structlog
import os
from typing import Dict, Any, Optional

logger = structlog.get_logger(__name__)

class NotificationService:
    """
    Tier 2: Alerting system for ML Guard.
    Supports Slack, Generic Webhooks, and Email (stub).
    """
    
    def __init__(self):
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.default_webhook_url = os.getenv("ALERT_WEBHOOK_URL")

    def send_alert(self, payload: Dict[str, Any], channel: Optional[str] = "slack"):
        """
        Routes alerts to the configured channels.
        """
        logger.info("Sending alert", channel=channel, alert_event=payload.get("event"))
        
        if channel == "slack" and self.slack_webhook_url:
            self._send_to_slack(payload)
        
        if self.default_webhook_url:
            self._send_to_webhook(payload)

    def _send_to_slack(self, payload: Dict[str, Any]):
        try:
            # Format Slack message
            risk_color = "#36a64f" if payload.get("severity") == "Low" else "#ffcc00" if payload.get("severity") == "Medium" else "#ff0000"
            
            slack_data = {
                "attachments": [
                    {
                        "fallback": f"ML Guard Alert: {payload.get('event')}",
                        "color": risk_color,
                        "title": f"🚨 ML Guard Alert: {payload.get('event')}",
                        "text": f"Project: *{payload.get('project')}*\nSeverity: *{payload.get('severity')}*",
                        "fields": [
                            {
                                "title": "Details",
                                "value": str(payload.get("details")),
                                "short": False
                            }
                        ],
                        "footer": "ML Guard Governance Bot",
                        "ts": int(datetime.utcnow().timestamp())
                    }
                ]
            }
            
            response = requests.post(self.slack_webhook_url, json=slack_data)
            response.raise_for_status()
        except Exception as e:
            logger.error("Slack alert failed", error=str(e))

    def _send_to_webhook(self, payload: Dict[str, Any]):
        try:
            response = requests.post(self.default_webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error("Generic webhook failed", error=str(e))

from datetime import datetime
