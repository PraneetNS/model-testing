import httpx
from typing import Any, Dict, List, Optional
from .slack_notifier import AlertSchema

class TeamsNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_breach_alert(self, alert: AlertSchema):
        severity = alert.severity.upper()
        emoji = "⚪"
        color = "default"
        if severity == "CRITICAL": 
            emoji = "🔴"
            color = "attention"
        elif severity == "HIGH": 
            emoji = "🟡"
            color = "warning"
        elif severity == "LOW": 
            emoji = "🟢"
            color = "good"

        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Large",
                                "weight": "Bolder",
                                "text": f"{emoji} {severity} Alert: {alert.model_name}",
                                "color": color
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Breach Type: {alert.breach_type}",
                                "isSubtle": True,
                                "spacing": "None"
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Current Score", "value": f"{alert.current_score:.4f}"},
                                    {"title": "Threshold", "value": f"{alert.threshold:.4f}"},
                                    {"title": "Breaches", "value": str(alert.breaches_in_window)},
                                    {"title": "Verdict", "value": alert.verdict}
                                ]
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "title": "View Full Report",
                                "url": f"{alert.dashboard_url}/models/{alert.model_id}"
                            }
                        ],
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "version": "1.2"
                    }
                }
            ]
        }

        if alert.is_predictive:
            card["attachments"][0]["content"]["body"].insert(2, {
                "type": "TextBlock",
                "text": f"⚠ Predicted breach in {alert.prediction_horizon or 'X'}h",
                "color": "warning",
                "weight": "Bolder"
            })

        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=card)

    async def send_score_decay_alert(self, model_id: str, old_score: float, new_score: float, delta: float, model_name: str = ""):
        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Large",
                                "weight": "Bolder",
                                "text": f"📉 Governance Score Decay: {model_name or model_id}",
                                "color": "attention"
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Score dropped from {old_score:.2f} to {new_score:.2f} (Delta: {delta:.2f})",
                                "wrap": True
                            }
                        ],
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "version": "1.2"
                    }
                }
            ]
        }
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=card)
