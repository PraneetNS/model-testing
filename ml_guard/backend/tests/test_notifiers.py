import pytest
import httpx
from unittest.mock import AsyncMock, patch
from ml_guard.plugins.slack_notifier import SlackNotifier, AlertSchema
from ml_guard.plugins.teams_notifier import TeamsNotifier

@pytest.mark.asyncio
async def test_slack_notifier_critical_payload():
    webhook_url = "https://hooks.slack.com/services/test"
    notifier = SlackNotifier(webhook_url, "#alerts")
    
    alert = AlertSchema(
        model_id="model-123",
        model_name="Credit Classifier",
        breach_type="Accuracy Breach",
        severity="CRITICAL",
        current_score=0.75,
        threshold=0.85,
        breaches_in_window=10,
        verdict="FAILED",
        dashboard_url="http://ml-guard.io"
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        await notifier.send_breach_alert(alert)
        
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        
        assert payload["channel"] == "#alerts"
        blocks = payload["blocks"]
        
        # Header block
        assert "🔴 CRITICAL Alert: Credit Classifier - Accuracy Breach" in blocks[0]["text"]["text"]
        
        # Fields block
        fields = blocks[1]["fields"]
        assert "*Current Score:* 0.7500" in [f["text"] for f in fields]
        assert "*Threshold:* 0.8500" in [f["text"] for f in fields]
        assert "*Breaches (Window):* 10" in [f["text"] for f in fields]
        assert "*Governance Verdict:* FAILED" in [f["text"] for f in fields]
        
        # Action block
        assert blocks[2]["elements"][0]["url"] == "http://ml-guard.io/models/model-123"

@pytest.mark.asyncio
async def test_teams_notifier_heavy_payload():
    webhook_url = "https://outlook.office.com/webhook/test"
    notifier = TeamsNotifier(webhook_url)
    
    alert = AlertSchema(
        model_id="model-456",
        model_name="Loan Risk",
        breach_type="Bias Detected",
        severity="HIGH",
        current_score=0.15,
        threshold=0.10,
        breaches_in_window=2,
        verdict="CONDITIONAL",
        is_predictive=True,
        prediction_horizon=24
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        await notifier.send_breach_alert(alert)
        
        args, kwargs = mock_post.call_args
        card = kwargs["json"]
        
        body = card["attachments"][0]["content"]["body"]
        
        # Title
        assert "🟡 HIGH Alert: Loan Risk" in body[0]["text"]
        
        # Predictive banner
        assert "⚠ Predicted breach in 24h" in body[2]["text"]
        
        # Facts
        facts = body[3]["facts"]
        assert any(f["title"] == "Current Score" and f["value"] == "0.1500" for f in facts)
