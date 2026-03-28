import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Boolean, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from app.db.session import Base

class ReportCard(Base):
    """
    Certified Governance Report Card for an ML Model.
    Acts as a professional certificate of compliance.
    """
    __tablename__ = "report_cards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Audit Identification
    cert_hash = Column(String(64), unique=True, nullable=False, index=True)
    issued_at = Column(DateTime, default=datetime.utcnow)
    
    # High Level Results
    overall_score = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False) # CERTIFIED, CONDITIONAL, FAILED
    
    # Summary & Content
    executive_summary = Column(String, nullable=True)
    metric_snapshots = Column(JSON, nullable=False) # Snapshot of audit data used
    
    # Status & Revocation
    is_revoked = Column(Boolean, default=False)
    revocation_reason = Column(String, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    
    # Storage Reference
    pdf_path = Column(String, nullable=True) # MinIO path: reports/{model_id}/{cert_hash}.pdf

    def __repr__(self):
        return f"<ReportCard(id={self.id}, model_id={self.model_id}, hash={self.cert_hash})>"
