import httpx
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class AlertSchema(BaseModel):
    model_id: str
    model_name: str
    breach_type: str
    severity: str
    current_score: float
    threshold: float
    breaches_in_window: int
    verdict: str
    dashboard_url: str = "http://localhost:3000"
    is_predictive: bool = False
    prediction_horizon: Optional[int] = None

class SlackNotifier:
    def __init__(self, webhook_url: str, channel: str):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_breach_alert(self, alert: AlertSchema):
        severity = alert.severity.upper()
        emoji = "⚪"
        if severity == "CRITICAL": emoji = "🔴"
        elif severity == "HIGH": emoji = "🟡"
        elif severity == "LOW": emoji = "🟢"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {severity} Alert: {alert.model_name} - {alert.breach_type}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Current Score:* {alert.current_score:.4f}"},
                    {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold:.4f}"},
                    {"type": "mrkdwn", "text": f"*Breaches (Window):* {alert.breaches_in_window}"},
                    {"type": "mrkdwn", "text": f"*Governance Verdict:* {alert.verdict}"}
                ]
            }
        ]

        if alert.is_predictive:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":warning: *Predicted breach in {alert.prediction_horizon or 'X'}h*"
                }
            })

        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Full Report"},
                    "url": f"{alert.dashboard_url}/models/{alert.model_id}"
                }
            ]
        })

        payload = {"channel": self.channel, "blocks": blocks}
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=payload)

    async def send_score_decay_alert(self, model_id: str, old_score: float, new_score: float, delta: float, model_name: str = ""):
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📉 Governance Score Decay: {model_name or model_id}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Score dropped from *{old_score:.2f}* to *{new_score:.2f}* (Delta: *{delta:.2f}*)"
                }
            }
        ]
        payload = {"channel": self.channel, "blocks": blocks}
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=payload)
