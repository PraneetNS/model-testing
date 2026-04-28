import re
from typing import Optional
from .models import GuardrailDecision

def local_evaluate_guardrail(prompt: str, response: Optional[str] = None) -> GuardrailDecision:
    """
    A simple local heuristic guardrail evaluation to use when disconnected from the platform.
    Checks for basic PII patterns and blocked words.
    """
    flags = []
    text_to_check = prompt + " " + (response or "")
    
    # Simple email check
    if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text_to_check):
        flags.append("PII: Email address detected")
        
    # Simple phone check
    if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text_to_check):
        flags.append("PII: Phone number detected")
        
    blocked_words = ["secret", "password", "confidential", "internal only"]
    for word in blocked_words:
        if word in text_to_check.lower():
            flags.append(f"Blocked word detected: {word}")

    passed = len(flags) == 0
    reason = "Failed basic local heuristics." if not passed else None
    
    return GuardrailDecision(
        passed=passed,
        reason=reason,
        flags=flags
    )

class Guardrail:
    def __init__(self, client=None, guardrail_id: str = "default"):
        self.client = client
        self.guardrail_id = guardrail_id

    def evaluate(self, prompt: str, response: Optional[str] = None) -> GuardrailDecision:
        if self.client:
            return self.client.evaluate_guardrail(self.guardrail_id, prompt, response)
        else:
            return local_evaluate_guardrail(prompt, response)
