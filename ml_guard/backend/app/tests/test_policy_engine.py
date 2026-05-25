import pytest
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock
from app.db.models import PolicyRule
from app.domain.services.governance_engine import GovernanceEngine

@pytest.mark.anyio
async def test_evaluate_with_no_policy_found_uses_defaults():
    # Mock database session
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.first.return_value = None
    mock_db.execute.return_value = mock_res

    engine = GovernanceEngine(mock_db)
    metrics = {
        "metrics": {"accuracy": 0.85},
        "drift": {"feat_1": {"PSI": 0.15}},
        "overfitting_gap": {"accuracy_gap": 0.05},
        "governance_score": 75.0
    }
    
    # Should use defaults from ml_guard.core.policy
    result = await engine.evaluate_active_policy(metrics, org_id=str(uuid4()))
    assert result["policy_name"] == "Default (No Active Policy Found)"
    assert result["gate_status"] == "PASSED"
    assert result["deployment_allowed"] is True

@pytest.mark.anyio
async def test_evaluate_with_active_policy_rule():
    # Create an active policy rule that is VERY STRICT
    strict_rules = {
        "min_accuracy": 0.99,  # Impossible for our 0.85
        "max_psi": 0.05        # Impossible for our 0.15
    }
    mock_policy = PolicyRule(
        id=uuid4(),
        org_id=uuid4(),
        name="Strict Policy",
        rules_json=strict_rules,
        is_active=True
    )

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.first.return_value = mock_policy
    mock_db.execute.return_value = mock_res

    engine = GovernanceEngine(mock_db)
    metrics = {
        "metrics": {"accuracy": 0.85},
        "drift": {"feat_1": {"PSI": 0.15}},
    }
    
    result = await engine.evaluate_active_policy(metrics, org_id=str(mock_policy.org_id))
    assert result["policy_name"] == "Strict Policy"
    assert result["gate_status"] == "CRITICAL"
    assert result["deployment_allowed"] is False
    
    # Verify specific check messages
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["Accuracy"]["status"] == "CRITICAL"
    assert checks["Max PSI Drift"]["status"] == "CRITICAL"

@pytest.mark.anyio
async def test_evaluate_scoped_to_org():
    org_id = uuid4()
    policy_a = PolicyRule(
        id=uuid4(),
        org_id=org_id,
        name="Org A Policy",
        rules_json={"min_accuracy": 0.90},
        is_active=True
    )

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.first.return_value = policy_a
    mock_db.execute.return_value = mock_res

    engine = GovernanceEngine(mock_db)
    metrics = {"metrics": {"accuracy": 0.50}}
    
    res_a = await engine.evaluate_active_policy(metrics, org_id=str(org_id))
    assert res_a["policy_name"] == "Org A Policy"
    assert res_a["gate_status"] == "CRITICAL"

@pytest.mark.anyio
async def test_evaluate_global_policy():
    policy_global = PolicyRule(
        id=uuid4(),
        org_id=None,
        name="Global Policy",
        rules_json={"min_accuracy": 0.95},
        is_active=True
    )

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.first.return_value = policy_global
    mock_db.execute.return_value = mock_res

    engine = GovernanceEngine(mock_db)
    metrics = {"metrics": {"accuracy": 0.90}}
    
    # If no org_id passed, it should find global policy
    res = await engine.evaluate_active_policy(metrics, org_id=None)
    assert res["policy_name"] == "Global Policy"
    assert res["gate_status"] == "CRITICAL"
