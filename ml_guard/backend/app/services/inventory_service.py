from datetime import datetime, timedelta
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from app.db.models import Model, AlertEvent, AlertRule, ScanRecord, utcnow
from ml_guard.core.risk_tiering import compute_risk_tier

async def auto_tier_model_logic(model_id: str, db: AsyncSession):
    """Business logic for auto-tiering a model."""
    res = await db.execute(select(Model).where(Model.id == model_id))
    model = res.scalar_one_or_none()
    if not model:
        return None

    # Get last governance score
    last_scan = await db.execute(
        select(ScanRecord)
        .where(ScanRecord.model_id == model.id)
        .order_by(desc(ScanRecord.created_at))
        .limit(1)
    )
    scan = last_scan.scalar_one_or_none()
    
    metadata = {
        "use_case_category": model.use_case_category,
        "training_data_sensitivity": model.training_data_sensitivity,
        "deployment_environment": model.deployment_environment,
        "regulatory_jurisdictions": model.regulatory_jurisdictions or [],
        "governance_score": scan.governance_score if scan else 50
    }
    
    result = compute_risk_tier(metadata)
    
    model.risk_tier = result["tier"]
    model.risk_tier_justification = f"Auto-tiered with score {result['composite_score']}. Factors: " + \
        "; ".join([f"{f['name']}: {f['score']}" for f in result["factors"]])
    
    await db.commit()
    await compute_next_validation(str(model.id), db)
    return result

async def compute_next_validation(model_id: str, db: AsyncSession):
    """
    Computes and updates the next validation date for a model based on its risk tier.
    Triggers alerts if validation is due or overdue.
    """
    res = await db.execute(select(Model).where(Model.id == model_id))
    model = res.scalar_one_or_none()
    if not model:
        return None

    # Determine frequency based on risk tier
    tier = model.risk_tier or "low"
    if tier == "critical":
        frequency = 90
    elif tier == "high":
        frequency = 180
    elif tier == "medium":
        frequency = 365
    else:
        frequency = 730

    model.validation_frequency_days = frequency
    
    # Calculate next validation date
    base_date = model.last_validated_at or model.created_at or utcnow()
    next_due = base_date + timedelta(days=frequency)
    model.next_validation_due_at = next_due
    
    await db.commit()
    
    # Check for alerts
    today = utcnow()
    days_remaining = (next_due - today).days
    
    if days_remaining < 0:
        # Overdue
        await create_inventory_alert(
            db, model, "CRITICAL", 
            f"Model validation overdue by {abs(days_remaining)} days."
        )
    elif days_remaining < 30:
        # Due soon
        await create_inventory_alert(
            db, model, "WARNING", 
            f"Model validation due in {days_remaining} days."
        )
    
    return model

async def create_inventory_alert(db: AsyncSession, model: Model, severity: str, message: str):
    """Internal helper to create inventory alerts."""
    # Try to find an inventory alert rule or create a dummy one
    res = await db.execute(select(AlertRule).where(AlertRule.name == "Inventory Monitoring"))
    rule = res.scalar_one_or_none()
    
    if not rule:
        rule = AlertRule(
            id=uuid.uuid4(),
            name="Inventory Monitoring",
            condition={"type": "inventory"},
            channels=["ui"]
        )
        db.add(rule)
        await db.flush()
    
    alert = AlertEvent(
        id=uuid.uuid4(),
        rule_id=rule.id,
        severity=severity,
        message=f"[{model.name}] {message}",
        created_at=utcnow()
    )
    db.add(alert)
    await db.commit()
