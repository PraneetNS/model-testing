import pytest
import uuid
import datetime
from unittest.mock import MagicMock, patch
from ml_guard.core.insurance_score import compute_insurance_score

@pytest.mark.asyncio
async def test_platinum_insurance_tier_unlocked():
    """
    Ensures a model with high governance, robustness, and supply chain integrity 
    reaches the 'platinum' insurance tier.
    """
    mock_db = MagicMock()
    model_id = uuid.uuid4()
    
    # Mock return values for DB queries
    mock_model = MagicMock(id=model_id, metadata_json={"risk_tier": "low"})
    mock_aibom = MagicMock(training_datasets=[])
    
    with patch("ml_guard.core.insurance_score.select"), \
         patch("ml_guard.core.insurance_score.evaluate_compliance") as mock_comp:
        
        # 1. Model Result
        # 2. Reliability (ScanRecord score)
        # 3. Robustness (RedTeamRun score)
        # 4. Breach Count (ContractBreach)
        # 5. Compliance Input (ScanRecord results_json)
        # 6. Supply Chain (AIBOM)
        # 7. Incidents (SecurityAlert count)
        mock_db.execute.side_effect = [
            MagicMock(scalars=lambda: MagicMock(first=lambda: mock_model)),
            MagicMock(scalar=lambda: 95.0), # Gov Score -> 200 pts
            MagicMock(scalar=lambda: 98.0), # Robustness -> 196 pts
            MagicMock(scalar=lambda: 10),   # 0.1% Breach Rate -> 150 pts
            MagicMock(scalar=lambda: {"acc": 0.9}),
            MagicMock(scalars=lambda: MagicMock(first=lambda: mock_aibom)), # AIBOM -> 150 pts
            MagicMock(scalar=lambda: 0)     # No Incidents -> 100 pts
        ]
        
        # 100% compliance -> 200 pts
        mock_comp.return_value = [{"status": "pass"}] * 10
        
        report = await compute_insurance_score(model_id, mock_db)
        
        # Total approx: 200 + 196 + 150 + 200 + 150 + 100 = 996
        assert report["total_score"] >= 900
        assert report["tier"] == "platinum"
        assert report["estimated_annual_premium_usd_range"]["min"] < report["estimated_annual_premium_usd_range"]["max"]
        assert report["estimated_annual_premium_usd_range"]["min"] > 0

@pytest.mark.asyncio
async def test_substandard_high_risk_incidents():
    """
    Ensures a model with critical incidents and no AIBOM is downgraded 
    to substandard or uninsurable.
    """
    mock_db = MagicMock()
    model_id = uuid.uuid4()
    
    mock_model = MagicMock(id=model_id, metadata_json={"risk_tier": "high"})
    
    with patch("ml_guard.core.insurance_score.select"), \
         patch("ml_guard.core.insurance_score.evaluate_compliance") as mock_comp:
        
        mock_db.execute.side_effect = [
            MagicMock(scalars=lambda: MagicMock(first=lambda: mock_model)),
            MagicMock(scalar=lambda: 20.0), # Reliability -> 0 pts
            MagicMock(scalar=lambda: 10.0), # Robustness -> 20 pts
            MagicMock(scalar=lambda: 900),  # 9.0% Breach Rate -> 50 pts (-75 penalty = 0)
            MagicMock(scalar=lambda: {}),
            MagicMock(scalars=lambda: MagicMock(first=lambda: None)), # No AIBOM -> 0 pts
            MagicMock(scalar=lambda: 5)     # 3+ Incidents -> 0 pts
        ]
        
        # 0% compliance -> 0 pts
        mock_comp.return_value = [{"status": "fail"}] * 10
        
        report = await compute_insurance_score(model_id, mock_db)
        
        assert report["total_score"] < 400
        assert report["tier"] in ["substandard", "uninsurable"]
        # Premium should be higher for high risk + bad score
        assert report["estimated_annual_premium_usd_range"]["min"] >= 250000 
