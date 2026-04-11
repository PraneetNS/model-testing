import pytest
import uuid
from datetime import datetime, timedelta
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.db.models import Base, ModelContract, ContractBreach
from app.services.contract_engine import ContractEngine

# In-memory SQLite for testing
db_engine = create_engine("sqlite:///:memory:", echo=False)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
def db() -> Generator[Session, None, None]:
    Base.metadata.create_all(bind=db_engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=db_engine)

@pytest.mark.anyio
async def test_grace_period_zero_penalty(db: Session):
    """
    Assert that 2 breaches within the grace period result in zero score penalty.
    """
    contract_id = uuid.uuid4()
    model_id = f"test_model_gp_{uuid.uuid4().hex[:6]}"
    contract = ModelContract(
        id=contract_id,
        model_id=model_id,
        name="Test Grace Period",
        is_active=True,
        breach_grace_period_minutes=5,
        breach_window_minutes=60,
        promises=[
            {"name": "Output >= 0", "type": "output", "metric": "prediction", "operator": "gte", "threshold": 0.0, "severity": "HIGH"}
        ]
    )
    db.add(contract)
    db.commit()

    engine = ContractEngine()
    
    # First breach
    breaches_1 = await engine.check_prediction(
        db, model_id, prediction=-1, prediction_proba=0.9, features={}, latency_ms=10
    )
    assert len(breaches_1) == 1
    assert breaches_1[0]["severity"] == "WARNING"

    # Second breach (within grace period)
    breaches_2 = await engine.check_prediction(
        db, model_id, prediction=-1, prediction_proba=0.9, features={}, latency_ms=10
    )
    assert len(breaches_2) == 1
    assert breaches_2[0]["severity"] == "WARNING"

    # For penalty calculation, count breaches within the last breach_window_minutes.
    summary = await engine.get_contract_breach_summary(db, str(contract_id))
    
    assert summary["current_window_breaches"] == 2
    assert summary["penalty_applied_today"] == 0.0
    assert summary["total_breaches_24h"] == 2
