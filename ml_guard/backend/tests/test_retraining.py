import pytest
import datetime
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import patch, MagicMock

import os
os.environ["MLGUARD_ENV"] = "development"
import app.core.config
app.core.config.settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"

from fastapi import FastAPI
from app.api.v1.endpoints.retraining import router as retraining_router
from ml_guard.core.retraining import evaluate_retrain_trigger, execute_retrain_action
from app.db.models import RetrainingPolicy, DriftReport, PerformanceSnapshot, RetrainingEvent

app = FastAPI()
app.include_router(retraining_router, prefix="/api/models")
client = TestClient(app)

@pytest.mark.anyio
async def test_simulate_returns_false_when_drift_below_threshold():
    mock_db = MagicMock(spec=AsyncSession)
    
    mock_policy = MagicMock(spec=RetrainingPolicy)
    mock_policy.model_id = "test-model-1"
    mock_policy.enabled = True
    mock_policy.trigger_conditions = {
        "psi_threshold": 0.2,
        "ks_stat_threshold": 0.1,
        "performance_degradation_pct": 15,
        "min_days_since_last_retrain": 7,
        "require_all_conditions": False
    }
    mock_policy.last_triggered_at = None
    
    # Mock drift report with PSI = 0.1 (below threshold of 0.2)
    mock_drift = MagicMock(spec=DriftReport)
    mock_drift.method = "psi"
    mock_drift.overall_drift_score = 0.1
    
    async def mock_execute(*args, **kwargs):
        mock_result = MagicMock()
        # The execute call happens multiple times. We need to return different things.
        # First query: policy, second: drift, third: perf
        # We can just return a generic mock that returns our mocks when .first() is called
        return mock_result

    # Let's mock the scalars().first() chain more explicitly
    mock_execute_result = MagicMock()
    mock_scalars = MagicMock()
    
    call_count = [0]
    def get_first():
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0: return mock_policy
        elif idx == 1: return mock_drift
        else: return None # No perf data

    mock_scalars.first.side_effect = get_first
    mock_execute_result.scalars.return_value = mock_scalars
    
    async def mock_execute_func(*args, **kwargs):
        return mock_execute_result
        
    mock_db.execute = mock_execute_func
    
    result = await evaluate_retrain_trigger("test-model-1", mock_db)
    
    assert result["should_trigger"] is False
    assert len(result["triggered_conditions"]) == 0
    assert result["suppressed"] is False

@pytest.mark.anyio
async def test_webhook_called_with_correct_payload_when_psi_above_threshold():
    mock_db = MagicMock(spec=AsyncSession)
    
    mock_policy = MagicMock(spec=RetrainingPolicy)
    mock_policy.id = "policy-1"
    mock_policy.model_id = "test-model-1"
    mock_policy.retrain_action = {
        "action_type": "webhook",
        "webhook_url": "https://example.com/webhook"
    }
    mock_policy.trigger_count = 0
    
    trigger_result = {
        "should_trigger": True,
        "triggered_conditions": ["PSI threshold breached (0.250 >= 0.2)"],
        "suppressed": False,
        "suppression_reason": None
    }
    
    with patch("httpx.AsyncClient.post") as mock_post:
        # Mock successful response
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        
        event = await execute_retrain_action(mock_policy, trigger_result, mock_db)
        
        # Verify db logic
        assert mock_db.add.called
        assert mock_db.commit.called
        
        # Verify event creation
        assert event.action_type == "webhook"
        assert event.action_result == "success"
        
        # Verify payload sent to webhook
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://example.com/webhook"
        payload = kwargs.get("json")
        assert payload["model_id"] == "test-model-1"
        assert "PSI threshold breached" in payload["triggered_conditions"][0]

@pytest.mark.anyio
async def test_suppression_works_when_last_retrain_too_recent():
    mock_db = MagicMock(spec=AsyncSession)
    
    mock_policy = MagicMock(spec=RetrainingPolicy)
    mock_policy.model_id = "test-model-1"
    mock_policy.enabled = True
    mock_policy.trigger_conditions = {
        "psi_threshold": 0.2,
        "min_days_since_last_retrain": 7,
        "require_all_conditions": False
    }
    # Retrained 2 days ago
    mock_policy.last_triggered_at = datetime.datetime.utcnow() - datetime.timedelta(days=2)
    
    # Mock drift report with PSI = 0.3 (above threshold)
    mock_drift = MagicMock(spec=DriftReport)
    mock_drift.method = "psi"
    mock_drift.overall_drift_score = 0.3
    
    mock_execute_result = MagicMock()
    mock_scalars = MagicMock()
    
    call_count = [0]
    def get_first():
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0: return mock_policy
        elif idx == 1: return mock_drift
        else: return None 

    mock_scalars.first.side_effect = get_first
    mock_execute_result.scalars.return_value = mock_scalars
    
    async def mock_execute_func(*args, **kwargs):
        return mock_execute_result
        
    mock_db.execute = mock_execute_func
    
    result = await evaluate_retrain_trigger("test-model-1", mock_db)
    
    # Should technically trigger, but suppressed
    assert result["should_trigger"] is False
    assert len(result["triggered_conditions"]) == 1
    assert "PSI threshold breached" in result["triggered_conditions"][0]
    assert result["suppressed"] is True
    assert "Suppressed: Retrained 2 days ago" in result["suppression_reason"]
