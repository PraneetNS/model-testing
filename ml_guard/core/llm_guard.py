"""
LLM Governance Guard Engine.

Implements heuristic-based governance checks for Large Language Models:
  - Prompt Injection Detection
  - Toxicity Scoring
  - Hallucination Risk Heuristic
  - Response Stability Analysis

All checks are deterministic and do NOT require external API calls.
They use pattern matching, statistical heuristics, and text analysis.
"""
import re
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter


# ═══════════════════════════════════════════════
# PROMPT INJECTION DETECTION
# ═══════════════════════════════════════════════

# Known injection patterns (extensible)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+the\s+above",
    r"disregard\s+(all\s+)?prior",
    r"you\s+are\s+now\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(if\s+you\s+are\s+)?",
    r"forget\s+(everything|all|your\s+instructions)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*/?script",
    r"\{\{.*\}\}",          # template injection
    r"```\s*(system|admin)",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"dan\s+mode",
    r"bypass\s+(safety|filter|restriction)",
    r"override\s+(safety|content|policy)",
]


def detect_prompt_injection(prompt: str) -> Dict[str, Any]:
    """
    Scan prompt for known injection patterns.
    Returns injection flag and matched patterns.
    """
    prompt_lower = prompt.lower().strip()
    matches = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt_lower):
            matches.append(pattern)

    # Heuristic: excessively long prompts with role directives
    role_keywords = sum(1 for kw in ["you are", "act as", "pretend", "assume role"]
                        if kw in prompt_lower)
    suspicion_score = min(1.0, (len(matches) * 0.3) + (role_keywords * 0.15))

    return {
        "injection_flag":   len(matches) > 0,
        "matched_patterns": len(matches),
        "suspicion_score":  round(suspicion_score, 4),
        "pattern_details":  matches[:5],  # cap detail output
    }


# ═══════════════════════════════════════════════
# TOXICITY SCORING
# ═══════════════════════════════════════════════

TOXIC_KEYWORDS = [
    "kill", "murder", "attack", "destroy", "hate", "racist", "sexist",
    "slur", "violence", "rape", "abuse", "terrorist", "bomb",
    "suicide", "self-harm", "harass", "threaten", "weapon",
    "illegal", "drugs", "exploit", "assault",
]

PROFANITY_MARKERS = [
    "fuck", "shit", "damn", "hell", "ass", "bitch", "bastard", "crap",
]


def compute_toxicity_score(text: str) -> Dict[str, Any]:
    """
    Heuristic toxicity scoring based on keyword density + pattern analysis.
    Score: 0 (clean) to 1 (highly toxic).
    """
    text_lower = text.lower()
    words = text_lower.split()
    word_count = max(len(words), 1)

    toxic_hits = sum(1 for w in words if any(t in w for t in TOXIC_KEYWORDS))
    profanity_hits = sum(1 for w in words if any(p in w for p in PROFANITY_MARKERS))

    # Normalize by text length
    toxic_density = toxic_hits / word_count
    profanity_density = profanity_hits / word_count

    # Combined score
    score = min(1.0, (toxic_density * 5.0) + (profanity_density * 3.0))

    # Severity classification
    if score > 0.5:
        severity = "HIGH"
    elif score > 0.2:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "toxicity_score":    round(score, 4),
        "toxic_keywords":    toxic_hits,
        "profanity_count":   profanity_hits,
        "severity":          severity,
        "word_count":        word_count,
    }


# ═══════════════════════════════════════════════
# HALLUCINATION RISK HEURISTIC
# ═══════════════════════════════════════════════

HEDGING_PHRASES = [
    "i think", "i believe", "probably", "maybe", "it seems",
    "possibly", "might be", "could be", "not sure", "approximately",
    "i'm not certain", "it's possible", "from what i know",
]

CONFIDENT_ASSERTIONS = [
    "definitely", "certainly", "absolutely", "always", "never",
    "guaranteed", "proven", "100%", "without a doubt", "unquestionably",
]


def compute_hallucination_risk(
    response: str,
    prompt: str = "",
    reference_facts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Heuristic hallucination risk assessment.

    Analyzes:
    1. Confidence vs hedging ratio
    2. Specificity of claims (numbers, dates, names without hedging)
    3. Cross-reference with provided facts (if any)
    4. Response length relative to prompt complexity
    """
    resp_lower = response.lower()
    words = resp_lower.split()
    word_count = max(len(words), 1)

    # 1. Hedging vs confidence ratio
    hedge_count = sum(1 for phrase in HEDGING_PHRASES if phrase in resp_lower)
    confident_count = sum(1 for phrase in CONFIDENT_ASSERTIONS if phrase in resp_lower)

    # 2. Specificity: numbers, dates, proper nouns (capitalized words)
    number_pattern = re.findall(r'\b\d+\.?\d*\b', response)
    date_pattern = re.findall(r'\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', resp_lower)
    specificity_count = len(number_pattern) + len(date_pattern)

    # 3. Cross-reference check
    fact_match_score = 1.0
    if reference_facts and len(reference_facts) > 0:
        matches = sum(1 for fact in reference_facts if fact.lower() in resp_lower)
        fact_match_score = matches / len(reference_facts)

    # 4. Ratio analysis
    overconfidence_ratio = confident_count / max(confident_count + hedge_count, 1)

    # Combined hallucination risk
    # High confidence + high specificity + no facts = risky
    risk = 0.0
    risk += overconfidence_ratio * 0.3           # overconfident = risky
    risk += min(specificity_count / 10, 1.0) * 0.2  # many specific claims
    risk += (1.0 - fact_match_score) * 0.3       # no fact backing
    risk += min(word_count / 500, 1.0) * 0.2     # very long = more room for error

    if hedge_count > confident_count:
        risk *= 0.6  # hedging reduces risk

    risk = float(np.clip(risk, 0.0, 1.0))

    return {
        "hallucination_risk":   round(risk, 4),
        "overconfidence_ratio": round(overconfidence_ratio, 4),
        "hedge_phrases":        hedge_count,
        "confident_assertions": confident_count,
        "specific_claims_count": specificity_count,
        "fact_match_score":     round(fact_match_score, 4) if reference_facts else None,
    }


# ═══════════════════════════════════════════════
# RESPONSE STABILITY ANALYSIS
# ═══════════════════════════════════════════════

def compute_response_stability(
    responses: List[str],
) -> Dict[str, Any]:
    """
    Analyze consistency across multiple responses to the same prompt.
    Uses token overlap (Jaccard similarity) between response pairs.
    """
    if len(responses) < 2:
        return {"stability_score": 1.0, "n_responses": len(responses), "variance": 0.0}

    # Tokenize
    token_sets = [set(r.lower().split()) for r in responses]

    # Pairwise Jaccard similarity
    similarities = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            sim = intersection / max(union, 1)
            similarities.append(sim)

    stability_score = float(np.mean(similarities)) if similarities else 1.0

    # Length variance
    lengths = [len(r.split()) for r in responses]
    length_cv = float(np.std(lengths) / max(np.mean(lengths), 1))

    return {
        "stability_score":  round(stability_score, 4),
        "n_responses":      len(responses),
        "mean_similarity":  round(stability_score, 4),
        "length_cv":        round(length_cv, 4),
        "variance":         round(float(np.var(similarities)), 6) if similarities else 0.0,
    }


# ═══════════════════════════════════════════════
# UNIFIED LLM GOVERNANCE EVALUATION
# ═══════════════════════════════════════════════

def evaluate_llm(
    prompt: str,
    response: str,
    additional_responses: Optional[List[str]] = None,
    reference_facts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full LLM governance evaluation.

    Parameters:
        prompt:              The input prompt.
        response:            The primary model response.
        additional_responses: Optional list of other responses for stability analysis.
        reference_facts:     Optional list of known facts for hallucination checking.

    Returns:
        Complete LLM governance report with risk level classification.
    """
    # Run all checks
    injection = detect_prompt_injection(prompt)
    toxicity_prompt = compute_toxicity_score(prompt)
    toxicity_response = compute_toxicity_score(response)
    hallucination = compute_hallucination_risk(response, prompt, reference_facts)

    all_responses = [response] + (additional_responses or [])
    stability = compute_response_stability(all_responses)

    # Fingerprints
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]

    # Combined risk score [0, 100]
    risk_score = 0.0
    risk_score += injection["suspicion_score"] * 25       # injection: 25% weight
    risk_score += toxicity_response["toxicity_score"] * 25  # toxicity: 25% weight
    risk_score += hallucination["hallucination_risk"] * 30  # hallucination: 30% weight
    risk_score += (1.0 - stability["stability_score"]) * 20  # instability: 20% weight
    risk_score = float(np.clip(risk_score, 0, 100))

    # Risk level
    if risk_score > 60:
        risk_level = "HIGH"
    elif risk_score > 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prompt_injection":     injection,
        "toxicity_prompt":      toxicity_prompt,
        "toxicity_response":    toxicity_response,
        "hallucination":        hallucination,
        "stability":            stability,
        "prompt_hash":          prompt_hash,
        "response_hash":        response_hash,
        "llm_risk_score":       round(risk_score, 2),
        "llm_risk_level":       risk_level,
        "toxicity_score":       toxicity_response["toxicity_score"],
        "prompt_injection_flag": injection["injection_flag"],
        "hallucination_risk":   hallucination["hallucination_risk"],
        "stability_score":      stability["stability_score"],
    }
