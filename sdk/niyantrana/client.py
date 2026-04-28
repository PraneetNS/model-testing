import threading
import json
import logging
from typing import Optional, Dict, Any
import pandas as pd

try:
    import httpx
except ImportError:
    httpx = None

from .models import DriftReport, GovernanceScore, GuardrailDecision
from .local.drift import detect_drift

logger = logging.getLogger(__name__)

class NiyantranaClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.niyantrana.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        
        if self.api_key and httpx is None:
            logger.warning("httpx is not installed. Platform connection requires httpx. Run: pip install httpx")

    def _post_async(self, endpoint: str, data: Dict[str, Any]):
        if not self.api_key or httpx is None:
            return

        def _send():
            try:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                # Use a sync client in the background thread for simplicity
                with httpx.Client() as client:
                    client.post(f"{self.base_url}{endpoint}", json=data, headers=headers)
            except Exception as e:
                logger.error(f"Failed to send async data to Niyantrana: {e}")

        thread = threading.Thread(target=_send)
        thread.daemon = True
        thread.start()

    def _post_sync(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or httpx is None:
            raise ValueError("API key and httpx are required for synchronous platform calls.")
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client() as client:
            response = client.post(f"{self.base_url}{endpoint}", json=data, headers=headers)
            response.raise_for_status()
            return response.json()

    def _get_sync(self, endpoint: str) -> Dict[str, Any]:
        if not self.api_key or httpx is None:
            raise ValueError("API key and httpx are required for synchronous platform calls.")
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client() as client:
            response = client.get(f"{self.base_url}{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json()

    def log_prediction(self, model_id: str, features: Dict[str, Any], prediction: Any, probability: float, latency_ms: float) -> None:
        """
        POSTs to /api/ingest asynchronously (background thread, non-blocking).
        """
        if not self.api_key:
            logger.info(f"[Local] Logged prediction for {model_id}: pred={prediction}, prob={probability}, latency={latency_ms}ms")
            return
            
        data = {
            "model_id": model_id,
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": latency_ms
        }
        self._post_async("/api/ingest", data)

    def run_drift_check(self, model_id: str, current_df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> DriftReport:
        """
        Calls /api/drift/{model_id} on the platform.
        Falls back to local `detect_drift` if api_key is missing.
        """
        if not self.api_key:
            if reference_df is None:
                raise ValueError("reference_df is required for local drift checking.")
            logger.info(f"[Local] Running local drift check for {model_id}")
            return detect_drift(reference_df, current_df)
            
        # Platform call
        # Assuming platform expects JSON records
        data = {
            "current_data": current_df.to_dict(orient="records")
        }
        result = self._post_sync(f"/api/drift/{model_id}", data)
        return DriftReport(**result)

    def get_governance_score(self, model_id: str) -> GovernanceScore:
        """
        Calls /api/governance/{model_id}/score
        """
        if not self.api_key:
            logger.info(f"[Local] Returning dummy governance score for {model_id}")
            return GovernanceScore(model_id=model_id, score=0.85, details={"status": "local mode dummy score"})
            
        result = self._get_sync(f"/api/governance/{model_id}/score")
        return GovernanceScore(**result)

    def evaluate_guardrail(self, guardrail_id: str, prompt: str, response: Optional[str] = None) -> GuardrailDecision:
        """
        Calls /api/guardrail/{guardrail_id}/evaluate
        """
        if not self.api_key:
            from .guardrail import local_evaluate_guardrail
            return local_evaluate_guardrail(prompt, response)
            
        data = {
            "prompt": prompt,
            "response": response
        }
        result = self._post_sync(f"/api/guardrail/{guardrail_id}/evaluate", data)
        return GuardrailDecision(**result)
