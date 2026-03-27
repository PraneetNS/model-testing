from pydantic import BaseModel, Field
from typing import Optional, Dict

class MLGuardPolicy(BaseModel):
    """
    Pydantic model for mlguard.yaml policy-as-code schema.
    """
    version: str = Field(default="1.0", description="Policy schema version")
    model_name: str = Field(description="Name of the model being gated")
    
    # Tabular ML Thresholds
    max_psi: float = Field(default=0.2, description="Maximum Population Stability Index (PSI) allowed")
    max_jsd: float = Field(default=0.1, description="Maximum Jensen-Shannon Divergence allowed")
    min_accuracy: float = Field(default=0.85, description="Minimum Accuracy score allowed")
    max_overfit_gap: float = Field(default=0.08, description="Maximum allowed gap between Train and Val metrics")
    
    # Fairness Thresholds
    bias_parity_threshold: float = Field(default=0.1, description="Max absolute difference for Statistical Parity")
    min_disparate_impact: float = Field(default=0.8, description="Min Disparate Impact Ratio (DIR)")
    
    # LLM Thresholds
    max_hallucination_rate: float = Field(default=0.05, description="Max allowed hallucination risk score")
    max_toxicity_score: float = Field(default=0.1, description="Max allowed toxicity score")
    max_injection_risk: float = Field(default=0.0, description="Max allowed injection risk (0=none)")

    # Metadata
    contacts: Optional[Dict[str, str]] = None

class GateRequest(BaseModel):
    """
    Request model for the /v1/gate/evaluate endpoint.
    """
    artifact_path: Optional[str] = None
    inference_url: Optional[str] = None
    policy: MLGuardPolicy
    context: Optional[Dict] = None

class GateVerdict(BaseModel):
    """
    Structured JSON verdict returned by the gate endpoint.
    """
    passed: bool
    score: float
    gate_status: str  # PASSED | WARNING | CRITICAL
    failures: list[str]
    badge_url: str
    details: Dict
