import pandas as pd
from niyantrana.local.contracts import Contract

def test_contract_validation():
    contract = Contract(name="SLA", confidence_min=0.8, latency_max_ms=100)
    
    # Passing
    result = contract.validate(prediction=1, probability=0.85, latency_ms=50)
    assert result.passed
    assert len(result.breaches) == 0
    
    # Failing confidence
    result2 = contract.validate(prediction=1, probability=0.75, latency_ms=50)
    assert not result2.passed
    assert len(result2.breaches) == 1
    assert "Confidence" in result2.breaches[0]
    
    # Failing latency
    result3 = contract.validate(prediction=1, probability=0.85, latency_ms=150)
    assert not result3.passed
    assert len(result3.breaches) == 1
    assert "Latency" in result3.breaches[0]
