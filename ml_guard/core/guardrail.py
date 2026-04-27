import re
import json
import uuid
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from ml_guard.core.rag import grounding_fidelity, hallucination_risk as get_hallucination_risk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GuardrailResult(BaseModel):
    flagged: bool
    matched_pattern: Optional[str] = None
    confidence: float = 1.0
    pii_types_found: Optional[List[str]] = None
    redacted_prompt: Optional[str] = None
    redacted_response: Optional[str] = None
    attack_type: Optional[str] = None
    matched_blocked_topic: Optional[str] = None
    toxicity_score: Optional[float] = None
    categories: Optional[List[str]] = None
    hallucination_risk: Optional[str] = None
    grounding_fidelity: Optional[float] = None

class GuardrailConfig(BaseModel):
    model_id: str
    name: str
    enabled_input_checks: List[str] = ["injection", "pii", "jailbreak", "topic_policy"]
    enabled_output_checks: List[str] = ["toxicity", "hallucination", "pii"]
    action_on_block: str = "return_error" # "return_error" | "return_fallback_response"
    fallback_response: Optional[str] = "I'm sorry, but I cannot fulfill this request due to safety policy violations."
    allowed_topics: List[str] = []
    blocked_topics: List[str] = []

class GuardrailDecision(BaseModel):
    action: str # "allow" | "block" | "redact" | "flag_for_review"
    blocked_reason: Optional[str] = None
    input_checks: Dict[str, Any]
    output_checks: Dict[str, Any]
    redacted_prompt: Optional[str] = None
    redacted_response: Optional[str] = None
    latency_ms: int
    trace_id: str

class GuardrailEngine:
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.injection_patterns = self._load_injection_patterns()
        self.pii_regex = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
            "phone": r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
            "aadhaar": r"\b\d{4}-\d{4}-\d{4}\b",
            "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
        }
        self.jailbreak_keywords = [
            "ignore all previous", "you are now in developer mode", "dan", "stay in character",
            "hypothetically", "pretend you are", "your true self", "system prompt", "extract",
            "unrestricted", "bypass", "jailbreak"
        ]
        self.toxicity_patterns = [
            r"hate", r"kill", r"die", r"stupid", r"idiot", r"terrorist", r"bomb", r"attack",
            r"explicit", r"porn", r"sex", r"violence", r"harm", r"suicide"
        ]

    def _load_injection_patterns(self) -> Dict[str, List[str]]:
        try:
            import os
            base_path = os.path.dirname(__file__)
            path = os.path.join(base_path, "guardrail_patterns", "injection.json")
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def check_prompt_injection(self, prompt: str) -> GuardrailResult:
        for category, patterns in self.injection_patterns.items():
            for pattern in patterns:
                if pattern.lower() in prompt.lower():
                    return GuardrailResult(flagged=True, matched_pattern=pattern, confidence=0.9, attack_type=category)
        return GuardrailResult(flagged=False, confidence=1.0)

    def _luhn_check(self, number: str) -> bool:
        digits = [int(d) for d in re.sub(r"\D", "", number)]
        if not digits: return False
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        total = sum(odd_digits)
        for d in even_digits:
            total += sum(divmod(d * 2, 10))
        return total % 10 == 0

    def check_pii_in_prompt(self, prompt: str) -> GuardrailResult:
        found_types = []
        redacted = prompt
        flagged = False
        
        for pii_type, regex in self.pii_regex.items():
            matches = re.finditer(regex, redacted)
            for match in matches:
                val = match.group()
                if pii_type == "credit_card" and not self._luhn_check(val):
                    continue
                flagged = True
                if pii_type not in found_types:
                    found_types.append(pii_type)
                redacted = redacted.replace(val, f"[REDACTED_{pii_type.upper()}]")
        
        return GuardrailResult(flagged=flagged, pii_types_found=found_types, redacted_prompt=redacted)

    def check_jailbreak(self, prompt: str) -> GuardrailResult:
        prompt_lower = prompt.lower()
        for kw in self.jailbreak_keywords:
            if kw in prompt_lower:
                return GuardrailResult(flagged=True, attack_type="keyword_heuristic", matched_pattern=kw)
        
        # Semantic heuristics (simple regex for common patterns)
        jailbreak_regex = [
            r"you are (now|going to be) .*",
            r"ignore (all )?previous (instructions|directives)",
            r"forget (your )?rules",
            r"act as if you are .*"
        ]
        for reg in jailbreak_regex:
            if re.search(reg, prompt_lower):
                return GuardrailResult(flagged=True, attack_type="regex_heuristic")
                
        return GuardrailResult(flagged=False)

    def check_topic_policy(self, prompt: str) -> GuardrailResult:
        if not self.config.blocked_topics:
            return GuardrailResult(flagged=False)
        
        try:
            vectorizer = TfidfVectorizer().fit([prompt] + self.config.blocked_topics)
            prompt_vec = vectorizer.transform([prompt])
            topic_vecs = vectorizer.transform(self.config.blocked_topics)
            similarities = cosine_similarity(prompt_vec, topic_vecs).flatten()
            
            for i, sim in enumerate(similarities):
                if sim > 0.4:
                    return GuardrailResult(flagged=True, matched_blocked_topic=self.config.blocked_topics[i])
        except Exception:
            pass
            
        return GuardrailResult(flagged=False)

    def check_toxicity(self, response: str) -> GuardrailResult:
        response_lower = response.lower()
        categories = []
        score = 0.0
        for pattern in self.toxicity_patterns:
            if re.search(pattern, response_lower):
                categories.append(pattern)
                score += 0.2
        
        score = min(score, 1.0)
        return GuardrailResult(flagged=score > 0.3, toxicity_score=score, categories=categories)

    def check_hallucination_risk(self, response: str, context_chunks: Optional[List[str]] = None) -> GuardrailResult:
        if context_chunks:
            fidelity = grounding_fidelity(response, context_chunks)
            risk = get_hallucination_risk(response, context_chunks)
            return GuardrailResult(flagged=fidelity < 0.3, hallucination_risk=risk, grounding_fidelity=fidelity)
        
        # No context: check for overconfident numeric claims
        numeric_claims = re.findall(r"\d+(\.\d+)?%", response)
        overconfident = ["exactly", "proven fact", "100%", "always", "never"]
        response_lower = response.lower()
        
        is_overconfident = any(word in response_lower for word in overconfident)
        flagged = is_overconfident and len(numeric_claims) > 0
        
        return GuardrailResult(
            flagged=flagged, 
            hallucination_risk="high" if flagged else "low"
        )

    def check_pii_in_response(self, response: str) -> GuardrailResult:
        res = self.check_pii_in_prompt(response)
        res.redacted_response = res.redacted_prompt
        res.redacted_prompt = None
        return res

    def evaluate(self, prompt: str, response: Optional[str] = None, context_chunks: Optional[List[str]] = None) -> GuardrailDecision:
        start_time = time.time()
        input_results = {}
        output_results = {}
        action = "allow"
        blocked_reason = None

        # Input checks
        if "injection" in self.config.enabled_input_checks:
            res = self.check_prompt_injection(prompt)
            input_results["injection"] = res.dict()
            if res.flagged: 
                action = "block"
                blocked_reason = f"Prompt injection detected: {res.matched_pattern}"

        if action == "allow" and "pii" in self.config.enabled_input_checks:
            res = self.check_pii_in_prompt(prompt)
            input_results["pii"] = res.dict()
            if res.flagged:
                action = "redact"
                prompt = res.redacted_prompt

        if action != "block" and "jailbreak" in self.config.enabled_input_checks:
            res = self.check_jailbreak(prompt)
            input_results["jailbreak"] = res.dict()
            if res.flagged:
                action = "block"
                blocked_reason = f"Jailbreak attempt detected ({res.attack_type})"

        if action != "block" and "topic_policy" in self.config.enabled_input_checks:
            res = self.check_topic_policy(prompt)
            input_results["topic_policy"] = res.dict()
            if res.flagged:
                action = "block"
                blocked_reason = f"Blocked topic detected: {res.matched_blocked_topic}"

        # Output checks
        if action != "block" and response:
            if "toxicity" in self.config.enabled_output_checks:
                res = self.check_toxicity(response)
                output_results["toxicity"] = res.dict()
                if res.flagged:
                    action = "block"
                    blocked_reason = "Toxic response detected"

            if action != "block" and "hallucination" in self.config.enabled_output_checks:
                res = self.check_hallucination_risk(response, context_chunks)
                output_results["hallucination"] = res.dict()
                if res.flagged:
                    action = "flag_for_review"

            if action != "block" and "pii" in self.config.enabled_output_checks:
                res = self.check_pii_in_response(response)
                output_results["pii"] = res.dict()
                if res.flagged:
                    if action == "allow": action = "redact"
                    response = res.redacted_response

        latency_ms = int((time.time() - start_time) * 1000)
        
        return GuardrailDecision(
            action=action,
            blocked_reason=blocked_reason,
            input_checks=input_results,
            output_checks=output_results,
            redacted_prompt=prompt if action == "redact" else None,
            redacted_response=response if action == "redact" else None,
            latency_ms=latency_ms,
            trace_id=str(uuid.uuid4())
        )
