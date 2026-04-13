import pytest
from unittest.mock import MagicMock
from ml_guard.core.agent_auditor import audit_step

class MockAgent:
    def __init__(self):
        self.allowed_tools = ["search", "calculator"]
        self.sensitive_topics = ["merger", "acquisition"]
        self.step_sla_ms = 1000

def test_unauthorized_tool_call():
    agent = MockAgent()
    step = {"type": "tool_call", "tool": "rm_rf", "input": "none", "output": "none"}
    violations = audit_step(step, agent, [])
    
    found = [v for v in violations if v["type"] == "unauthorized_tool_call"]
    assert len(found) > 0
    assert found[0]["severity"] == "CRITICAL"

def test_loop_detection_fires_on_fourth():
    agent = MockAgent()
    # 3 existing identical steps in session history
    history = [
        MagicMock(step_type="llm_call", tool_name=None),
        MagicMock(step_type="llm_call", tool_name=None),
        MagicMock(step_type="llm_call", tool_name=None)
    ]
    # The 4th call of the same type
    step = {"type": "llm_call", "tool": None, "output": "..."}
    violations = audit_step(step, agent, history)
    
    found = [v for v in violations if v["type"] == "potential_infinite_loop"]
    assert len(found) > 0
    assert found[0]["severity"] == "HIGH"

def test_pii_exfiltration_pattern():
    agent = MockAgent()
    # Output containing an email and a mock SSN
    step = {
        "type": "output", 
        "output": "User data: test@example.com, SSN: 999-00-1111"
    }
    violations = audit_step(step, agent, [])
    
    v_types = [v["type"] for v in violations]
    assert "potential_pii_exfiltration" in v_types
    # Should find at least two instances (email and SSN)
    exfil_violations = [v for v in violations if v["type"] == "potential_pii_exfiltration"]
    assert len(exfil_violations) >= 2

def test_scope_creep_detection():
    agent = MockAgent()
    step = {
        "type": "llm_call",
        "output": "We are planning a secret merger with a competitor next month."
    }
    violations = audit_step(step, agent, [])
    
    found = [v for v in violations if v["type"] == "scope_creep"]
    assert len(found) > 0
    assert found[0]["severity"] == "HIGH"
