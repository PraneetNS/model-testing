import uuid
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import Model, ScanRecord, AuditLog
# CORRECT - imports from the single source of truth
from app.db.models import ReportCard
import structlog

logger = structlog.get_logger()

class ReportCardBuilder:
    """
    Core synthesis engine for Governance Report Cards.
    Aggregates metrics, computes weighted scores, and generates certificate hashes.
    """
    
    # Weight configuration for Governance Score (0-100)
    WEIGHTS = {
        "psi_drift": 0.20,
        "performance": 0.25,
        "bias_fairness": 0.25,
        "ll_m_safety": 0.20,
        "robustness": 0.10
    }

    def __init__(self, db: AsyncSession, model_id: str):
        self.db = db
        self.model_id = model_id
        self.model = None

    async def initialize(self):
        self.model = await self.db.get(Model, self.model_id)
        if not self.model:
            raise ValueError(f"Model {self.model_id} not found.")

    async def aggregate_audit_data(self) -> Dict[str, Any]:
        """
        Pulls latest audit metrics for the model across all categories.
        """
        # Fetch latest scan result (most recent audit)
        query = select(ScanRecord).filter(ScanRecord.model_id == self.model_id).order_by(ScanRecord.created_at.desc())
        result = await self.db.execute(query)
        latest_audit = result.scalars().first()
            
        if not latest_audit:
            logger.warning("No audit results found for model", model_id=self.model_id)
            return {}

        # Synthesize normalized scores (0-100)
        # Mocking/assuming structure from typical ML Guard scans
        return {
            "psi_drift": self._normalize_score(latest_audit.metrics.get("drift_psi", 0.0), lower_is_better=True),
            "performance": self._normalize_score(latest_audit.metrics.get("accuracy", 0.0), lower_is_better=False),
            "bias_fairness": self._normalize_score(latest_audit.metrics.get("bias_score", 0.0), lower_is_better=True),
            "llm_safety": self._normalize_score(latest_audit.metrics.get("safety_violation_rate", 0.0), lower_is_better=True) if self.model.is_llm else 100.0,
            "robustness": self._normalize_score(latest_audit.metrics.get("robustness_index", 0.0), lower_is_better=False),
            "audit_timestamp": latest_audit.created_at.isoformat(),
            "scan_id": str(latest_audit.id)
        }

    async def compute_governance_score(self, aggregated_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Computes weighted total score and determines final verdict.
        Also checks ModelExplanation to flag 'SHAP-Fairness Alert' if top drift is inside sensitive_features.
        """
        # Fetch explanation
        from app.db.models import ExplainabilityResult # Changed to ExplainabilityResult based on current schema
        query = select(ExplainabilityResult).filter(ExplainabilityResult.model_id == self.model_id).order_by(ExplainabilityResult.created_at.desc())
        result = await self.db.execute(query)
        explanation = result.scalars().first()
        
        if explanation and explanation.summary_metrics:
            top_features = explanation.summary_metrics.get("top_features", [])
            if top_features:
                top_drift = top_features[0]
                sensitive_features = (self.model.metadata_json or {}).get("sensitive_features", [])
                if top_drift in sensitive_features:
                    aggregated_data["shap_fairness_alert"] = True
        total = 0.0
        # Re-adjust weights if LLM safety is not applicable
        active_weights = self.WEIGHTS.copy()
        if not self.model.is_llm:
            # Rebalance LLM Safety weight into others
            extra = active_weights.pop("llm_safety") / len(active_weights)
            for k in active_weights:
                active_weights[k] += extra

        for key, weight in active_weights.items():
            # Normalized score map
            metric_key = key if key != "llm_safety" else "llm_safety" # Key normalization
            total += aggregated_data.get(metric_key, 0.0) * weight

        # Verdict logic
        verdict = "CERTIFIED"
        if total < 60:
            verdict = "FAILED"
        elif total < 85:
            verdict = "CONDITIONAL"

        return round(total, 2), verdict

    def generate_cert_hash(self, model_id: str, timestamp: str, score: float) -> str:
        """
        Deterministic SHA-256 hash for certificate identification.
        """
        payload = f"{model_id}:{timestamp}:{score}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def determine_gate_status(self, metric: str, value: float) -> str:
        """
        PASS/WARN/FAIL based on internal threshold policy.
        """
        if value > 90: return "PASS"
        if value > 75: return "WARN"
        return "FAIL"

    def _normalize_score(self, value: float, lower_is_better: bool = False) -> float:
        """
        Ensure metric is on 0-100 scale where higher is better.
        """
        # Simple normalization logic for mock purposes:
        # If higher is better: scale 0.0-1.0 to 0-100.
        # If lower is better (like PSI): 100 - (scaled 0-1.0)
        norm = max(0.0, min(1.0, value)) * 100
        return 100 - norm if lower_is_better else norm
