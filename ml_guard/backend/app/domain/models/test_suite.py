
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime

class TestCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    STATISTICAL_STABILITY = "statistical_stability"
    MODEL_PERFORMANCE = "model_performance"
    BIAS_FAIRNESS = "bias_fairness"
    ROBUSTNESS = "robustness"

class TestSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestConfig(BaseModel):
    name: str
    category: TestCategory
    type: str # e.g., "psi_drift", "accuracy_threshold"
    severity: TestSeverity
    config: Dict[str, Any] # Flexible config for specific test logic
    description: Optional[str] = None

class TestResult(BaseModel):
    test_id: str
    test_name: str
    category: Union[TestCategory, str]
    status: str # "passed", "failed", "warning"
    severity: Union[TestSeverity, str]
    message: str
    execution_time_seconds: float = 0.0
    actual_value: Optional[Any] = None
    details: Dict[str, Any] = {}
    description: Optional[str] = "No description available."
    explanation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TestSuite(BaseModel):
    id: str
    name: str
    description: str
    tests: List[TestConfig]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelMetadata(BaseModel):
    id: str
    name: str
    version: str
    framework: str # sklearn, xgboost, torch
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]

class QualityGateResult(BaseModel):
    run_id: str
    project_id: str
    model_version: str
    test_suite: str
    score: float # 0-100
    deployment_allowed: bool
    results: List[TestResult]
    risk_level: Optional[str] = "Low"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
