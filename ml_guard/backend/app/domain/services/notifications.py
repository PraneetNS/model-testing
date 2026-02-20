import requests
import structlog
from typing import Optional
from app.core.config import settings

logger = structlog.get_logger(__name__)

class NotificationService:
    @staticmethod
    def send_slack_alert(webhook_url: str, message: str, severity: str = "critical"):
        """
        Send a formatted slack alert for ML failures.
        """
        color = "#ff0000" if severity == "critical" else "#ffa500"
        payload = {
            "attachments": [
                {
                    "fallback": f"ML Guard Alert: {message}",
                    "color": color,
                    "title": f"🛡️ ML Guard - {severity.upper()} ALERT",
                    "text": message,
                    "fields": [
                        {
                            "title": "Status",
                            "value": "GATE FAILED",
                            "short": True
                        }
                    ],
                    "footer": "ML Guard Governance Platform"
                }
            ]
        }
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))

    @staticmethod
    def send_generic_webhook(url: str, data: dict):
        try:
            requests.post(url, json=data)
        except Exception as e:
            logger.error("Webhook failed", error=str(e))
