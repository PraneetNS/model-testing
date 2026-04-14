"""
AI Advisory Assistant Router.

- Takes structured governance result JSON
- LLM explains failures and suggests mitigations
- LLM NEVER recomputes metrics or overrides policy
- Optional feature toggle (disabled by default)
"""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import Optional
from app.db.session import get_db
from app.db.models import ScanRecord
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()

# ─── Feature toggle ───
AI_ADVISORY_ENABLED = True  # Set to False to disable LLM feature entirely


class AdvisoryRequest(BaseModel):
    scan_id: Optional[str] = None
    results_json: Optional[dict] = None
    question: str = "Why is my governance score low?"


def _build_system_prompt() -> str:
    return """You are ML Guard AI Advisor — an expert ML governance consultant.

STRICT SCOPE / INTENT FILTERING:
- You ONLY answer questions related to the ML Guard platform, ML governance, model issues (drift, overfitting, calibration, etc.), and the provided results.
- If a user asks an unrelated question (e.g., about recipes, movies, general trivia, code for unrelated things, or anything not related to ML governance/platform), you MUST politely refuse.
- IMPORTANT: If the user's question is invalid or out-of-scope, you MUST start your response EXACTLY with the string "INVALID_INPUT:" followed by a brief, polite explanation of your scope.

STRICT RULES:
1. You ONLY explain and advise. You NEVER recompute metrics.
2. You NEVER override deterministic policy decisions.
3. You reference the EXACT numbers from the provided results.
4. You suggest specific, actionable mitigation steps.
5. You keep responses concise and structured.
6. You use markdown formatting for readability.

When analyzing results:
- Identify the weakest governance components
- Explain WHY each metric is concerning
- Provide step-by-step remediation
- Prioritize by severity (CRITICAL → WARNING → INFO)
"""


def _build_analysis_prompt(results: dict, question: str) -> str:
    """Build a structured prompt from governance results."""
    gov = results.get("governance", {})
    score = gov.get("governance_score", "N/A")
    components = gov.get("component_scores", {})
    metrics = results.get("metrics", {})
    gaps = results.get("overfitting_gap", {})
    drift = results.get("drift", {})
    cal = results.get("calibration", {})
    leakage = results.get("leakage", {})
    policy = results.get("policy", {})
    advisories = results.get("advisories", [])

    # Build a structured context block
    context = f"""## Governance Results Summary

**Governance Score**: {score}/100
**Gate Status**: {policy.get('gate_status', 'N/A')}
**Deployment**: {'ALLOWED' if gov.get('deployment_allowed') else 'BLOCKED'}

### Component Scores
{json.dumps(components, indent=2) if components else 'No component breakdown available.'}

### Performance Metrics
{json.dumps(metrics, indent=2) if metrics else 'No metrics computed.'}

### Overfitting Gaps
{json.dumps(gaps, indent=2) if gaps else 'No overfitting gaps detected.'}

### Drift Summary
- Features analyzed: {len(drift)}
- Top drifted: {', '.join(results.get('top5_drifted_features', [])) or 'None'}

### Calibration
{json.dumps(cal, indent=2) if cal else 'Not computed.'}

### Leakage
{json.dumps(leakage, indent=2) if leakage else 'Not computed.'}

### Policy Checks
{json.dumps(policy.get('checks', []), indent=2) if policy.get('checks') else 'No checks.'}

### Existing Advisories
{json.dumps(advisories, indent=2) if advisories else 'None.'}

---

**User Question**: {question}
"""
    return context


def _generate_local_advisory(results: dict, question: str) -> dict:
    """
    Generate advisory WITHOUT calling external LLM.
    Uses structured analysis of the results to produce a deterministic explanation.
    This acts as a fallback when no LLM API key is configured.
    """
    gov = results.get("governance", {})
    score = gov.get("governance_score")
    components = gov.get("component_scores", {})
    policy = results.get("policy", {})
    advisories = results.get("advisories", [])
    gaps = results.get("overfitting_gap", {})
    drift = results.get("drift", {})
    cal = results.get("calibration", {})

    sections = []

    # Score analysis
    if score is not None:
        if score >= 80:
            sections.append(f"### ✅ Governance Score: {score:.0f}/100\nYour model has a **strong** governance score. It meets most deployment criteria.")
        elif score >= 60:
            sections.append(f"### ⚠️ Governance Score: {score:.0f}/100\nYour model has a **moderate** governance score. Some areas need attention before production deployment.")
        else:
            sections.append(f"### ❌ Governance Score: {score:.0f}/100\nYour model has a **low** governance score. **Deployment is blocked.** Critical issues must be resolved.")

    # Weakest component
    if components:
        weakest = min(components.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 100)
        sections.append(f"### Weakest Component: `{weakest[0]}`\nScore: **{weakest[1]}/100** — This is dragging down your overall governance score the most. Focus remediation here first.")

    # Overfitting
    critical_gaps = {k: v for k, v in gaps.items() if isinstance(v, (int, float)) and abs(v) > 0.08}
    if critical_gaps:
        gap_text = "\n".join([f"- **{k}**: gap = {v:+.4f}" for k, v in critical_gaps.items()])
        sections.append(f"### ⚠️ Overfitting Detected\n{gap_text}\n\n**Mitigation**: Use cross-validation, reduce model complexity, increase regularization, or collect more training data.")

    # Drift
    drifted = [feat for feat, vals in drift.items() if isinstance(vals, dict) and vals.get("drift_flag")]
    if drifted:
        sections.append(f"### 🔄 Feature Drift Detected\nDrifted features: **{', '.join(drifted[:5])}**\n\n**Mitigation**: Retrain on recent data, investigate data pipeline changes, or add monitoring for these features.")

    # Calibration
    if cal and isinstance(cal, dict):
        brier = cal.get("brier_score")
        if brier and brier > 0.2:
            sections.append(f"### 📊 Poor Calibration (Brier: {brier:.4f})\nThe model's predicted probabilities don't match actual outcomes.\n\n**Mitigation**: Apply Platt scaling or isotonic regression post-hoc calibration.")
        if cal.get("overconfident_flag"):
            sections.append("### ⚡ Overconfidence Detected\nThe model is systematically overconfident in its predictions.\n\n**Mitigation**: Temperature scaling, label smoothing, or ensembling can reduce overconfidence.")

    # Policy failures
    failed_checks = [c for c in policy.get("checks", []) if c.get("status") in ("CRITICAL", "WARNING")]
    if failed_checks:
        check_text = "\n".join([f"- **{c.get('name', 'Check')}**: {c.get('message', '')}" for c in failed_checks[:5]])
        sections.append(f"### 🛡️ Policy Violations\n{check_text}")

    # Existing advisories
    critical_advisories = [a for a in advisories if a.get("severity") == "CRITICAL"]
    if critical_advisories:
        adv_text = "\n".join([f"- [{a.get('code')}] {a.get('message')}" for a in critical_advisories[:3]])
        sections.append(f"### 🚨 Critical Advisories\n{adv_text}")

    # General recommendations
    recommendations = []
    if score is not None and score < 70:
        recommendations.append("• **Priority 1**: Address all CRITICAL policy violations")
        recommendations.append("• **Priority 2**: Reduce overfitting gaps below 8%")
        recommendations.append("• **Priority 3**: Retrain on recent data to address drift")
        recommendations.append("• **Priority 4**: Apply calibration techniques")

    if recommendations:
        sections.append("### 🎯 Recommended Action Plan\n" + "\n".join(recommendations))

    explanation = "\n\n".join(sections) if sections else "No significant governance issues detected. Your model meets deployment criteria."

    return {
        "advisory_type": "local_analysis",
        "question": question,
        "explanation": explanation,
        "disclaimer": "This analysis is generated deterministically from structured governance results. No external LLM was used.",
        "governance_score": score,
        "gate_status": policy.get("gate_status"),
    }


@router.post("/advisory/explain")
async def explain_governance(
    body: AdvisoryRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    AI Advisory endpoint.
    Takes governance results (from scan_id or direct JSON) + user question.
    Returns structured explanation card.

    LLM is optional — falls back to deterministic structured analysis.
    """
    if not AI_ADVISORY_ENABLED:
        raise HTTPException(403, "AI Advisory feature is disabled.")

    # Get results from scan_id or direct JSON
    results = body.results_json
    if not results and body.scan_id:
        scan = await db.get(ScanRecord, body.scan_id)
        if not scan:
            raise HTTPException(404, "Scan not found.")
        results = scan.results_json

    if not results:
        raise HTTPException(400, "Provide scan_id or results_json.")

    # Generate advisory
    advisory = _generate_local_advisory(results, body.question)

    # Log action
    await log_action(db, auth, "advisory.explain", resource_type="scan", resource_id=body.scan_id, details={"question": body.question})

    return advisory


@router.post("/advisory/explain-with-llm")
async def explain_with_llm(
    body: AdvisoryRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    LLM-powered advisory (requires OPENAI_API_KEY or compatible endpoint).
    Falls back to local analysis if LLM unavailable.
    """
    if not AI_ADVISORY_ENABLED:
        raise HTTPException(403, "AI Advisory feature is disabled.")

    results = body.results_json
    if not results and body.scan_id:
        scan = await db.get(ScanRecord, body.scan_id)
        if not scan:
            raise HTTPException(404, "Scan not found.")
        results = scan.results_json

    if not results:
        raise HTTPException(400, "Provide scan_id or results_json.")

    # Try LLM via Groq
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No GROQ_API_KEY configured.")

        import requests as req
        system_prompt = _build_system_prompt()
        user_prompt = _build_analysis_prompt(results, body.question)

        # Using Groq API endpoint
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        llm_response = resp.json()["choices"][0]["message"]["content"]

        # ─── Intent Filtering ───
        if llm_response.strip().startswith("INVALID_INPUT:"):
            # The LLM flagged this as out-of-scope.
            msg = llm_response.replace("INVALID_INPUT:", "").strip()
            return {
                "advisory_type": "llm_rejected",
                "provider": "groq",
                "question": body.question,
                "explanation": f"**Query Rejected:** {msg}",
                "disclaimer": "This question violates the intent filters. The AI Advisor is strictly limited to ML Guard platform, governance results, and model-related queries.",
                "governance_score": None,
                "gate_status": None,
            }

        # Log action
        await log_action(db, auth, "advisory.explain_llm", resource_type="scan", resource_id=body.scan_id, details={"question": body.question, "provider": "groq"})
        return {
            "advisory_type": "llm",
            "provider": "groq",
            "question": body.question,
            "explanation": llm_response,
            "disclaimer": "This explanation was generated by ML Guard AI Advisor (powered by Groq). It explains existing metrics but does NOT recompute or override any governance decisions.",
            "governance_score": results.get("governance", {}).get("governance_score"),
            "gate_status": results.get("policy", {}).get("gate_status"),
        }

    except Exception as e:
        # Fallback to local analysis
        advisory = _generate_local_advisory(results, body.question)
        advisory["fallback_reason"] = f"LLM unavailable: {str(e)}"
        # Log fallback
        await log_action(db, auth, "advisory.explain_fallback", resource_type="scan", resource_id=body.scan_id, details={"question": body.question, "error": str(e)})
        return advisory
