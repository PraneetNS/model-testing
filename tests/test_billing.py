import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException, Request
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits
from app.db.models import Organization, SubscriptionPlan, UsageEvent

@pytest.mark.asyncio
async def test_record_usage_persists_to_db():
    org_id = "test_org"
    event_type = "prediction_logged"
    
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    
    with patch("app.tasks.billing.SessionLocal", return_value=mock_db):
        from app.tasks.billing import record_usage_task
        # Calling the underlying function of the task
        # Celery bound tasks pass 'self' automatically when called via .run() if it's already bound
        await record_usage_task.run(org_id, None, event_type, 1, {})
        
        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, UsageEvent)
        assert added_obj.org_id == org_id
        mock_db.commit.assert_awaited_once()

@pytest.mark.asyncio
async def test_check_billing_limits_allows_under_limit():
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/api/v1/reports/generate"
    mock_request.headers = {"X-Org-ID": "test_org"}
    
    mock_plan = SubscriptionPlan(slug="free", reports_per_month=10)
    mock_org = Organization(id="test_org", plan_id="plan_123")
    
    # Mocking the sequence of DB calls
    mock_db.execute.side_effect = [
        # Call 1: Fetch Org
        MagicMock(scalars=lambda: MagicMock(first=lambda: mock_org)),
        # Call 2: Fetch Plan
        MagicMock(scalars=lambda: MagicMock(first=lambda: mock_plan)),
        # Call 3: Fetch Usage
        MagicMock(scalar=lambda: 5) # 5 < 10
    ]
    
    with patch("app.billing.enforcement.AsyncSessionLocal", return_value=mock_db):
        await check_billing_limits(mock_request)

@pytest.mark.asyncio
async def test_check_billing_limits_blocks_over_limit():
    mock_db = AsyncMock()
    mock_db.__aenter__.return_value = mock_db
    
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/api/v1/reports/generate"
    mock_request.headers = {"X-Org-ID": "test_org"}
    
    mock_plan = SubscriptionPlan(slug="free", reports_per_month=10)
    mock_org = Organization(id="test_org", plan_id="plan_123")
    
    mock_db.execute.side_effect = [
        # Call 1: Fetch Org
        MagicMock(scalars=lambda: MagicMock(first=lambda: mock_org)),
        # Call 2: Fetch Plan
        MagicMock(scalars=lambda: MagicMock(first=lambda: mock_plan)),
        # Call 3: Fetch Usage
        MagicMock(scalar=lambda: 15) # 15 > 10
    ]
    
    with patch("app.billing.enforcement.AsyncSessionLocal", return_value=mock_db):
        with pytest.raises(HTTPException) as excinfo:
            await check_billing_limits(mock_request)
        assert excinfo.value.status_code == 402

@pytest.mark.asyncio
async def test_stripe_usage_reporting():
    from app.billing.stripe_client import StripeClient
    
    with patch("stripe.billing.MeterEvent.create") as mock_create:
        StripeClient.report_usage("cus_123", "prediction_logged", 100)
        
        mock_create.assert_called_once()
        args = mock_create.call_args[1]
        assert args["event_name"] == "prediction_logged"
        assert args["payload"]["value"] == "100"
        assert args["payload"]["stripe_customer_id"] == "cus_123"
