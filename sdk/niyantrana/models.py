from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FeatureDrift(BaseModel):
    feature: str
    statistic: float
    threshold: float
    drifted: bool

class DriftReport(BaseModel):
    overall_drift_detected: bool
    per_feature: List[FeatureDrift]
    method: str
    reference_rows: int
    current_rows: int

class FeatureFairness(BaseModel):
    feature: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    disparate_impact_ratio: float
    flags: List[str]

class FairnessReport(BaseModel):
    overall_fair: bool
    per_feature: List[FeatureFairness]

class FeatureImportance(BaseModel):
    feature: str
    importance: float # For SHAP: mean_abs_shap. For LIME: importance
    rank: int

class ExplainReport(BaseModel):
    method: str
    feature_importances: List[FeatureImportance]

class ContractResult(BaseModel):
    passed: bool
    breaches: List[str]

class GovernanceScore(BaseModel):
    model_id: str
    score: float
    details: Dict[str, Any]

class GuardrailDecision(BaseModel):
    passed: bool
    reason: Optional[str] = None
    flags: List[str] = []
