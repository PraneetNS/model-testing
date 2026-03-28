from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import UUID

class ReportCardBase(BaseModel):
    model_id: UUID
    overall_score: float = Field(..., ge=0, le=100)
    verdict: str = Field(..., example="CERTIFIED")
    executive_summary: Optional[str] = None
    metric_snapshots: Dict[str, Any]

class ReportCardCreate(ReportCardBase):
    cert_hash: str

class ReportCardUpdate(BaseModel):
    is_revoked: bool = False
    revocation_reason: Optional[str] = None

class ReportCardResponse(ReportCardBase):
    id: UUID
    cert_hash: str
    issued_at: datetime
    is_revoked: bool
    pdf_path: Optional[str] = None

    class Config:
        from_attributes = True

class ReportCardVerification(BaseModel):
    valid: bool
    model_name: str
    issued_at: datetime
    overall_score: float
    verdict: str
    revoked: bool
    revocation_reason: Optional[str] = None
