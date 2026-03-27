from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class ForecastPoint(BaseModel):
    date: str
    value: float
    lower: Optional[float] = None
    upper: Optional[float] = None

class ForecastResult(BaseModel):
    model_id: str
    metric: str
    forecast_points: List[ForecastPoint]
    breach_date: Optional[str] = None
    breach_confidence: float = 0.0
    trend: str  # IMPROVING | STABLE | DEGRADING
    recommendation: str
    status: str # SUCCESS | INSUFFICIENT_DATA | ERROR

class ModelForecastSummary(BaseModel):
    model_id: str
    summary: str
    forecasts: Dict[str, ForecastResult]
