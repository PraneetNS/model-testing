from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class LLMModelPullRequest(BaseModel):
    model_name: str = Field(..., example="mistralai/Mistral-7B")
    provider: str = Field(..., example="HuggingFace") # HuggingFace, OpenAI, External
    api_key: Optional[str] = None
    encryption_key_id: Optional[str] = None # For encrypted storage

class LLMEvaluationConfig(BaseModel):
    knowledge_benchmark: bool = True
    hallucination_test: bool = True
    consistency_test: bool = True
    toxicity_test: bool = True
    bias_test: bool = True
    jailbreak_test: bool = True
    max_tokens: int = 500
    timeout: int = 30

class LLMEvalJobResponse(BaseModel):
    job_id: str
    status: str
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class LLMMetrics(BaseModel):
    knowledge_score: float = 0.0
    hallucination_rate: float = 0.0
    toxicity_score: float = 0.0
    bias_score: float = 0.0
    consistency_score: float = 0.0
    jailbreak_score: float = 0.0
    governance_score: float = 0.0
    deployment_status: str = "PENDING"

class LLMFullReport(BaseModel):
    job_id: str
    model_name: str
    provider: str
    metrics: LLMMetrics
    detailed_results: List[Dict[str, Any]]
    completed_at: datetime = Field(default_factory=datetime.utcnow)
