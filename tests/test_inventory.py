import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Model, ScanRecord, AlertEvent
from app.services.inventory_service import auto_tier_model_logic, compute_next_validation
from ml_guard.core.risk_tiering import compute_risk_tier

@pytest.mark.asyncio
async def test_risk_tiering_logic():
    # Test case: High sensitivity model
    metadata = {
        "use_case_category": "medical_diagnosis",
        "training_data_sensitivity": "restricted",
        "deployment_environment": "production",
        "regulatory_jurisdictions": ["US", "EU"],
        "governance_score": 40
    }
    
    result = compute_risk_tier(metadata)
    assert result["tier"] in ["critical", "high"]
    assert result["composite_score"] > 7
    
    # Test case: Low sensitivity model
    metadata_low = {
        "use_case_category": "content_recommendation",
        "training_data_sensitivity": "public",
        "deployment_environment": "development",
        "regulatory_jurisdictions": [],
        "governance_score": 95
    }
    result_low = compute_risk_tier(metadata_low)
    assert result_low["tier"] == "low"
    assert result_low["composite_score"] < 3

@pytest.mark.asyncio
async def test_auto_tier_and_validation_schedule(db: AsyncSession):
    # 1. Setup model
    model = Model(
        id=uuid.uuid4(),
        name="Test Inventory Model",
        use_case_category="credit_scoring",
        training_data_sensitivity="internal",
        deployment_environment="staging"
    )
    db.add(model)
    await db.commit()
    
    # 2. Add a scan record
    scan = ScanRecord(
        model_id=str(model.id),
        governance_score=65.0,
        gate_status="CONDITIONAL",
        scan_type="audit"
    )
    db.add(scan)
    await db.commit()
    
    # 3. Run auto-tier
    result = await auto_tier_model_logic(str(model.id), db)
    assert result is not None
    
    await db.refresh(model)
    assert model.risk_tier is not None
    assert model.next_validation_due_at is not None
    
    # 4. Verify validation frequency
    # Credit scoring (9) * 0.4 + Internal (3) * 0.2 + Staging (3) * 0.15 + 0 + (100-65)/10 * 0.15
    # 3.6 + 0.6 + 0.45 + 0.525 = 5.175 (High risk)
    # High risk frequency = 180 days
    if model.risk_tier == "high":
        assert model.validation_frequency_days == 180

@pytest.mark.asyncio
async def test_validation_overdue_alert(db: AsyncSession):
    # Setup model with overdue validation
    past_date = datetime.utcnow() - timedelta(days=400)
    model = Model(
        id=uuid.uuid4(),
        name="Overdue Model",
        risk_tier="critical",
        last_validated_at=past_date,
        validation_frequency_days=90
    )
    db.add(model)
    await db.commit()
    
    # Compute next validation (should trigger alert)
    await compute_next_validation(str(model.id), db)
    
    # Check for AlertEvent
    from sqlalchemy import select
    res = await db.execute(select(AlertEvent).filter(AlertEvent.message.contains("Overdue Model")))
    alert = res.scalars().first()
    assert alert is not None
    assert alert.severity == "CRITICAL"
