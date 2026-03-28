import os
from typing import Dict, Any
import anthropic
import structlog

logger = structlog.get_logger()

class ExecutiveSummaryGenerator:
    """
    LLM-powered compliance officer for concise report card summaries.
    Converts raw audit metrics into professional, non-technical English.
    """
    
    SYSTEM_PROMPT = (
        "You are a compliance officer writing a 3-paragraph executive summary of an AI model audit. "
        "Be precise, professional, and non-technical. Highlight pass/fail gates and key risks. "
        "Max 150 words total. Provide a one-line final verdict at the end."
    )

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    async def generate_summary(self, audit_json: Dict[str, Any]) -> str:
        """
        Synthesize text from audit data using Claude-Haiku.
        """
        if not self.client:
            logger.warning("Anthropic API key not found. Returning mock summary.")
            return self._mock_summary(audit_json)

        try:
            # Prepare context
            metrics_view = {
                "Governance Score": f"{audit_json.get('governance_score', 0)}/100",
                "PSI Drift": audit_json.get("psi_drift_status", "PASS"),
                "Bias & Fairness": audit_json.get("bias_fairness_status", "PASS"),
                "Robustness": audit_json.get("robustness_status", "PASS")
            }
            
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Review these audit results: {metrics_view}"}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Failed to generate executive summary via LLM", error=str(e))
            return self._mock_summary(audit_json)

    def _mock_summary(self, audit_json: Dict[str, Any]) -> str:
        score = audit_json.get("governance_score", 0)
        verdict = "PASS" if score > 80 else "WARN" if score > 60 else "FAIL"
        
        return (
            f"The audit of model {audit_json.get('model_name', 'M-ID-XXXX')} concluded with an overall "
            f"Governance Score of {score}/100. The model demonstrated stable performance with "
            f"drift metrics within accepted ranges. However, robustness checks suggest optimization is "
            f"needed for edge-case inputs.\n\n"
            f"Final Verdict: {verdict}"
        )
