import re
import hashlib
from typing import List, Dict, Any

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"\d{4}-\d{4}-\d{4}-\d{4}"
}

def audit_step(step: Dict[str, Any], agent: Any, session_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Audits a single agent step against defined policies.
    
    a. tool_call_allowlist: if tool_name not in the agent's registered allowed_tools list → violation "unauthorized_tool_call" CRITICAL.
    b. output_scope_creep: if the output of a step contains keywords from a configurable sensitive_topics list → violation "scope_creep" HIGH.
    c. loop_detection: if the same (step_type, tool_name) pair appears > 3 times in a session → violation "potential_infinite_loop" HIGH.
    d. latency_sla: if step latency_ms > agent's configured step_sla_ms → violation "step_sla_breach" LOW.
    e. data_exfiltration_pattern: if output contains patterns matching email, SSN, credit card regex → violation "potential_pii_exfiltration" CRITICAL.
    """
    violations = []
    
    # a. tool_call_allowlist
    if step.get("type") == "tool_call":
        tool_name = step.get("tool")
        if tool_name not in (agent.allowed_tools or []):
            violations.append({
                "type": "unauthorized_tool_call", 
                "severity": "CRITICAL", 
                "details": f"Tool '{tool_name}' not in allowlist"
            })

    # b. output_scope_creep
    output_text = str(step.get("output", ""))
    for topic in (agent.sensitive_topics or []):
        if topic.lower() in output_text.lower():
            violations.append({
                "type": "scope_creep", 
                "severity": "HIGH", 
                "details": f"Sensitive topic '{topic}' detected in output"
            })

    # c. loop_detection
    current_key = (step.get("type"), step.get("tool"))
    repeat_count = 0
    for hist in session_history:
        h_type = hist.step_type if hasattr(hist, "step_type") else hist.get("type")
        h_tool = hist.tool_name if hasattr(hist, "tool_name") else hist.get("tool")
        if (h_type, h_tool) == current_key:
            repeat_count += 1
    
    if repeat_count >= 3: # 4th occurrence triggers it
        violations.append({
            "type": "potential_infinite_loop", 
            "severity": "HIGH", 
            "details": f"Step sequence {current_key} repeated {repeat_count + 1} times"
        })

    # d. latency_sla
    latency = step.get("latency_ms", 0)
    if latency > (agent.step_sla_ms or 5000):
        violations.append({
            "type": "step_sla_breach", 
            "severity": "LOW", 
            "details": f"Latency {latency}ms exceeds SLA {agent.step_sla_ms}ms"
        })

    # e. data_exfiltration_pattern
    for p_name, pattern in PII_PATTERNS.items():
        if re.search(pattern, output_text):
            violations.append({
                "type": "potential_pii_exfiltration", 
                "severity": "CRITICAL", 
                "details": f"Pattern matching {p_name} found in output"
            })

    return violations

def compute_step_risk(violations: List[Dict[str, Any]]) -> int:
    """
    Score = 100 - (CRITICAL_violations * 20) - (HIGH_violations * 8) - (LOW_violations * 2), floored at 0.
    """
    score_sum = 0
    for v in violations:
        if v["severity"] == "CRITICAL": score_sum += 20
        elif v["severity"] == "HIGH": score_sum += 8
        elif v["severity"] == "LOW": score_sum += 2
    
    return max(0, 100 - score_sum)
