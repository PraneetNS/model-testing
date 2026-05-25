import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Boolean, Integer, Index
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.infrastructure.database import Base


class UUID(TypeDecorator):
    """
    Platform-independent UUID type.
    Uses PostgreSQL's UUID natively, falls back to CHAR(36) for SQLite.
    This fixes the crash when running with SQLite locally.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            try:
                return str(uuid.UUID(str(value)))
            except Exception:
                return str(value)
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            try:
                return uuid.UUID(str(value))
            except Exception:
                return value
        return value


def utcnow():
    """Timezone-aware UTC now — compatible with Python 3.12+."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), index=True, unique=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    users = relationship("User", back_populates="tenant")
    projects = relationship("Project", back_populates="tenant")


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    firebase_uid = Column(String(128), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=True) # Optional with Firebase
    full_name = Column(String(255))
    role = Column(String(50), default="developer")  # admin, auditor, developer
    is_active = Column(Boolean(), default=True)
    tenant_id = Column(UUID(), ForeignKey("tenants.id"), index=True)

    tenant = relationship("Tenant", back_populates="users")


class Project(Base):
    __tablename__ = "projects"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), index=True)
    tenant_id = Column(UUID(), ForeignKey("tenants.id"), index=True)
    created_at = Column(DateTime, default=utcnow)

    tenant = relationship("Tenant", back_populates="projects")
    runs = relationship("TestRun", back_populates="project")
    artifacts = relationship("ModelArtifact", back_populates="project")


class ModelArtifact(Base):
    """Tier 1: Model Registry Metadata"""
    __tablename__ = "model_artifacts"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.id"), index=True)
    version = Column(String(50), nullable=False)
    artifact_uri = Column(String(512), nullable=False)  # S3/Local Path
    model_type = Column(String(100))
    parameters = Column(JSON)  # Hyperparameters
    signature = Column(JSON)   # Input/Output schema
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="artifacts")


class Dataset(Base):
    """Tier 1: Dataset Fingerprinting & Versioning"""
    __tablename__ = "datasets"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    fingerprint = Column(String(64), unique=True, index=True)  # SHA-256
    schema_definition = Column(JSON)
    storage_uri = Column(String(512))
    row_count = Column(Integer)
    created_at = Column(DateTime, default=utcnow)


class TestRun(Base):
    __tablename__ = "test_runs"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.id"), index=True)
    model_artifact_id = Column(UUID(), ForeignKey("model_artifacts.id"), nullable=True)
    dataset_id = Column(UUID(), ForeignKey("datasets.id"), nullable=True)
    
    suite_name = Column(String(255))
    score = Column(Float)
    deployment_allowed = Column(Boolean)
    risk_level = Column(String(20))  # Tier 3: Low/Medium/Critical
    
    # Tier 1: Reproducibility & Metadata
    reproducibility_token = Column(String(64), index=True)
    environment_config = Column(JSON)  # Python version, lib versions
    execution_metadata = Column(JSON)  # Worker ID, duration
    
    summary_metrics = Column(JSON)
    results_raw = Column(JSON)
    created_at = Column(DateTime, default=utcnow, index=True)

    project = relationship("Project", back_populates="runs")
    drift_logs = relationship("DriftLog", back_populates="test_run")


class DriftLog(Base):
    __tablename__ = "drift_logs"
    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(UUID(), ForeignKey("test_runs.id"), index=True, nullable=True)
    monitoring_job_id = Column(UUID(), ForeignKey("monitoring_jobs.id"), index=True, nullable=True)
    feature_name = Column(String(255), index=True)
    metric_type = Column(String(50))  # PSI, KS, JS
    metric_value = Column(Float)
    is_drifted = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=utcnow, index=True)

    test_run = relationship("TestRun", back_populates="drift_logs")
    monitoring_job = relationship("MonitoringJob", back_populates="drift_results")


class PredictionLog(Base):
    """Production: Efficient logging of live predictions for drift analysis."""
    __tablename__ = "prediction_logs"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.id"), index=True)
    model_version = Column(String(50), index=True)
    features = Column(JSON)  # High-cardinality feature data
    prediction = Column(JSON) # Model output
    actual = Column(JSON, nullable=True) # Ground truth (if available later)
    timestamp = Column(DateTime, default=utcnow, index=True)


class MonitoringJob(Base):
    """Production: Scheduled background monitoring configurations."""
    __tablename__ = "monitoring_jobs"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.id"), index=True)
    name = Column(String(255))
    cron_expression = Column(String(100)) # e.g., "0 * * * *" (hourly)
    last_run = Column(DateTime)
    drift_threshold = Column(Float, default=0.1)
    alert_config = Column(JSON) # Slack channel, email, etc.
    is_active = Column(Boolean, default=True)

    drift_results = relationship("DriftLog", back_populates="monitoring_job")


class FeatureBaseline(Base):
    """Phase 2: Store baseline distribution profile for each feature."""
    __tablename__ = "feature_baselines"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.id"), index=True)
    model_version = Column(String(50), index=True)
    feature_name = Column(String(255), index=True)
    distribution_type = Column(String(50)) # "numeric" or "categorical"
    histogram_bins = Column(JSON)
    percentiles = Column(JSON)
    created_at = Column(DateTime, default=utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(), ForeignKey("users.id"), index=True)
    action = Column(String(100), index=True)
    details = Column(JSON)
    timestamp = Column(DateTime, default=utcnow, index=True)

class LLMEvaluation(Base):
    __tablename__ = "llm_evaluations"
    id = Column(String(64), primary_key=True, index=True)
    model_name = Column(String(255), index=True)
    provider = Column(String(50))
    metrics = Column(JSON)
    detailed_results = Column(JSON)
    status = Column(String(20)) # IN_PROGRESS, COMPLETED, FAILED
    error = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)


class PolicyRule(Base):
    __tablename__ = "policy_rules"
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    rules_json = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
