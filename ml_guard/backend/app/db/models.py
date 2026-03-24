import uuid
import hashlib
import secrets
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Integer, Boolean, Text, Index, BigInteger
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.db.session import Base


# ── Portable JSON type: uses JSONB on PostgreSQL, JSON on SQLite ──────────────
class PortableJSON(TypeDecorator):
    """Use JSONB on PostgreSQL for better indexing/query, plain JSON elsewhere."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB)
        return dialect.type_descriptor(JSON)


class UUID(TypeDecorator):
    impl = CHAR
    cache_ok = True
    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))
    def process_bind_param(self, value, dialect):
        if value is None: return value
        return str(value)
    def process_result_value(self, value, dialect):
        if value is None: return value
        if not isinstance(value, uuid.UUID):
            try: return uuid.UUID(str(value))
            except: return value
        return value


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def generate_api_key():
    return f"mlg_{secrets.token_urlsafe(32)}"


# ══════════════════════════════════════════════════════════
# ENTERPRISE LAYER — MULTI-TENANT + ORGANIZATIONS
# ══════════════════════════════════════════════════════════

class Organization(Base):
    __tablename__ = "organizations"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name        = Column(String(255), unique=True, nullable=False)
    slug        = Column(String(255), unique=True, nullable=False, index=True)
    plan        = Column(String(50), default="free")  # free, pro, enterprise
    settings    = Column(PortableJSON, default=dict)
    created_at  = Column(DateTime, default=utcnow)

    users       = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    projects    = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    api_keys    = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False)
    email       = Column(String(255), unique=True, nullable=False, index=True)
    name        = Column(String(255), nullable=False)
    role        = Column(String(50), default="viewer")  # admin, ml_engineer, auditor, viewer
    auth_provider = Column(String(50), default="local")  # local, google, github, azure_ad
    auth_id     = Column(String(255), nullable=True)  # OAuth provider ID
    password_hash = Column(String(512), nullable=True)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=utcnow)
    last_login  = Column(DateTime, nullable=True)

    organization = relationship("Organization", back_populates="users")


class Project(Base):
    __tablename__ = "projects"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False)
    name        = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_by  = Column(UUID(), ForeignKey("users.id"), nullable=True)
    created_at  = Column(DateTime, default=utcnow)

    organization = relationship("Organization", back_populates="projects")
    models       = relationship("Model", back_populates="project", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False)
    key_hash    = Column(String(128), unique=True, nullable=False, index=True)
    label       = Column(String(255), nullable=False)
    scopes      = Column(PortableJSON, default=list)  # ["audit", "behavior", "monitor"]
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=utcnow)
    last_used   = Column(DateTime, nullable=True)

    organization = relationship("Organization", back_populates="api_keys")


# ══════════════════════════════════════════════════════════
# MODEL REGISTRY + VERSIONING
# ══════════════════════════════════════════════════════════

class Model(Base):
    __tablename__ = "models"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id  = Column(UUID(), ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=True)
    name        = Column(String(255), index=True, nullable=False)
    provider    = Column(String(255), default="Local", nullable=False)
    fingerprint = Column(String(128), nullable=True, index=True)  # SHA-256
    metadata_json = Column(PortableJSON, nullable=True)  # model_class, framework, task, etc.
    complexity  = Column(PortableJSON, nullable=True)
    version     = Column(Integer, default=1)
    parent_id   = Column(UUID(), ForeignKey("models.id"), nullable=True)  # for version chains
    # ─── Cloud Storage Fields ────────────────────────────────────────────────
    artifact_url             = Column(String(1024), nullable=True)  # R2 object key / URL
    artifact_size            = Column(BigInteger, nullable=True)     # bytes
    artifact_storage_provider = Column(String(50), nullable=True)   # "cloudflare_r2"
    # ────────────────────────────────────────────────────────────────────────
    created_at  = Column(DateTime, default=utcnow)
    created_by  = Column(UUID(), ForeignKey("users.id"), nullable=True)

    project     = relationship("Project", back_populates="models")
    datasets    = relationship("Dataset", back_populates="model", cascade="all, delete-orphan")
    nlp_intents = relationship("NLPIntent", back_populates="model", cascade="all, delete-orphan")
    jobs        = relationship("Job", back_populates="model", cascade="all, delete-orphan")
    scans       = relationship("ScanRecord", back_populates="model", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    type        = Column(String(50), nullable=False)
    metadata_json = Column(PortableJSON, nullable=False)
    row_count   = Column(Integer, nullable=False)
    fingerprint = Column(String(128), nullable=True)
    created_at  = Column(DateTime, default=utcnow)

    model       = relationship("Model", back_populates="datasets")


class NLPIntent(Base):
    __tablename__ = "nlp_intents"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    raw_intent  = Column(String(1024), nullable=False)
    parsed_constraints = Column(PortableJSON, nullable=False)
    created_at  = Column(DateTime, default=utcnow)

    model       = relationship("Model", back_populates="nlp_intents")


# ══════════════════════════════════════════════════════════
# JOBS + SCAN RECORDS (FULL AUDIT TRAIL)
# ══════════════════════════════════════════════════════════

class Job(Base):
    __tablename__ = "jobs"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    status      = Column(String(50), default="RUNNING")
    error       = Column(String(1024), nullable=True)
    created_at  = Column(DateTime, default=utcnow)

    model       = relationship("Model", back_populates="jobs")


class ScanRecord(Base):
    """Full audit log of every governance scan ever run."""
    __tablename__ = "scan_records"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=True)
    job_id          = Column(String(50), index=True, nullable=True) # UUID of the job
    scan_type       = Column(String(50), nullable=False)  # audit, behavior, monitor
    policy_version_id = Column(UUID(), ForeignKey("policy_versions.id"), nullable=True)
    checks_run      = Column(PortableJSON, nullable=False)  # list of check names
    results_json    = Column(PortableJSON, nullable=False)  # full results snapshot
    governance_score = Column(Float, nullable=True)
    gate_status     = Column(String(20), nullable=True)  # PASSED, WARNING, CRITICAL
    # ─── Feature 1: Risk Score ───────────────────────────────────────────────
    risk_score      = Column(Integer, nullable=True)           # 0–100 deterministic risk
    risk_level      = Column(String(20), nullable=True)        # LOW / MEDIUM / HIGH / CRITICAL
    # ─── Feature 2: Top Drifted Features ─────────────────────────────────────
    top_drifted_features = Column(PortableJSON, nullable=True) # [{feature, psi, severity}]
    # ─── Feature 3: Fairness Metrics ──────────────────────────────────────────
    fairness_metrics     = Column(PortableJSON, nullable=True)  # full fairness report
    bias_violation_flag  = Column(Boolean, nullable=True)       # True if any fairness threshold violated
    fairness_risk_score  = Column(Float, nullable=True)         # 0-100 fairness-based risk
    # ─── Cloud Storage References ────────────────────────────────────────────
    artifact_url            = Column(String(1024), nullable=True)  # R2 key for model artifact
    training_dataset_url    = Column(String(1024), nullable=True)  # R2 key for training data
    validation_dataset_url  = Column(String(1024), nullable=True)  # R2 key for validation data
    # ─── Feature 1: Model Versioning ─────────────────────────────────────────
    model_version_id        = Column(UUID(), ForeignKey("model_versions.id"), nullable=True)
    # ─── Feature 9: Model Security ───────────────────────────────────────────
    security_checks         = Column(PortableJSON, nullable=True)  # security audit results
    # ─────────────────────────────────────────────────────────────────────────
    triggered_by    = Column(UUID(), ForeignKey("users.id"), nullable=True)
    trigger_source  = Column(String(50), default="ui")  # ui, api, ci, github
    duration_ms     = Column(Integer, nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    model           = relationship("Model", back_populates="scans")



class AuditLog(Base):
    """Generic audit trail for all platform actions."""
    __tablename__ = "audit_logs"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    user_id     = Column(UUID(), ForeignKey("users.id"), nullable=True)
    action      = Column(String(100), nullable=False, index=True)  # model.upload, scan.run, policy.update, etc.
    resource_type = Column(String(50), nullable=True)  # model, policy, project, etc.
    resource_id = Column(String(64), nullable=True)
    details     = Column(PortableJSON, nullable=True)
    ip_address  = Column(String(45), nullable=True)
    created_at  = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# VERSIONED GOVERNANCE POLICIES
# ══════════════════════════════════════════════════════════

class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    name        = Column(String(255), default="Default Policy", nullable=False)
    version     = Column(Integer, default=1, nullable=False)
    config      = Column(PortableJSON, nullable=False)  # threshold config dict
    is_active   = Column(Boolean, default=True)
    created_by  = Column(UUID(), ForeignKey("users.id"), nullable=True)
    created_at  = Column(DateTime, default=utcnow)
    notes       = Column(Text, nullable=True)


# ══════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════

class AlertRule(Base):
    __tablename__ = "alert_rules"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    name        = Column(String(255), nullable=False)
    condition   = Column(PortableJSON, nullable=False)  # {"metric": "governance_score", "op": "<", "value": 70}
    channels    = Column(PortableJSON, nullable=False)   # ["slack", "email", "webhook"]
    webhook_url = Column(String(1024), nullable=True)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=utcnow)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    rule_id     = Column(UUID(), ForeignKey("alert_rules.id", ondelete="CASCADE"), index=True, nullable=False)
    scan_id     = Column(UUID(), ForeignKey("scan_records.id"), nullable=True)
    severity    = Column(String(20), nullable=False)
    message     = Column(Text, nullable=False)
    delivered   = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# GITHUB / CI INTEGRATION
# ══════════════════════════════════════════════════════════

class CIIntegration(Base):
    __tablename__ = "ci_integrations"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False)
    provider    = Column(String(50), nullable=False)  # github, gitlab, jenkins
    repo_url    = Column(String(1024), nullable=True)
    access_token_hash = Column(String(256), nullable=True)
    webhook_secret    = Column(String(256), nullable=True)
    settings    = Column(PortableJSON, default=dict)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# EXISTING MODULE RESULT TABLES (unchanged contract)
# ══════════════════════════════════════════════════════════

class PreflightResult(Base):
    __tablename__ = "preflight_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


class DriftResult(Base):
    __tablename__ = "drift_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


class PerformanceResult(Base):
    __tablename__ = "performance_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


class FairnessResult(Base):
    __tablename__ = "fairness_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


class LLMResult(Base):
    __tablename__ = "llm_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


class GovernanceResult(Base):
    __tablename__ = "governance_results"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    job_id      = Column(UUID(), ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_metrics_json = Column(PortableJSON, nullable=False)
    severity_counts = Column(PortableJSON, nullable=False)
    status      = Column(String(50), nullable=False)
    created_at  = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# NEW: EXPLICIT POLICY RULE MODEL
# ══════════════════════════════════════════════════════════

class PolicyRule(Base):
    """Explicit model for governance policy rules as per user request."""
    __tablename__ = "policy_rules"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id      = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    name        = Column(String(255), nullable=False)
    rules_json  = Column(PortableJSON, nullable=False)  # The actual policy rules
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# NEW: LLM SCAN RECORDS (v7.0)
# ══════════════════════════════════════════════════════════

class LLMScanRecord(Base):
    """Audit trail for LLM governance evaluations."""
    __tablename__ = "llm_scan_records"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    prompt_hash     = Column(String(64), nullable=False, index=True)
    response_hash   = Column(String(64), nullable=False)
    prompt_text     = Column(Text, nullable=True)  # stored for audit (optional)
    response_text   = Column(Text, nullable=True)
    results_json    = Column(PortableJSON, nullable=False)  # full evaluation results
    llm_risk_score  = Column(Float, nullable=True)
    llm_risk_level  = Column(String(20), nullable=True)  # LOW / MEDIUM / HIGH
    toxicity_score  = Column(Float, nullable=True)
    hallucination_risk = Column(Float, nullable=True)
    injection_flag  = Column(Boolean, default=False)
    stability_score = Column(Float, nullable=True)
    triggered_by    = Column(UUID(), ForeignKey("users.id"), nullable=True)
    created_at      = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# NEW: STREAMING DRIFT RECORDS (v7.0)
# ══════════════════════════════════════════════════════════

class StreamDriftRecord(Base):
    """Snapshot of streaming drift evaluation per window."""
    __tablename__ = "stream_drift_records"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(String(255), nullable=False, index=True)
    window_psi      = Column(Float, nullable=False)
    window_jsd      = Column(Float, nullable=True)
    trend           = Column(String(20), nullable=True)  # stable / increasing / critical
    alert           = Column(Boolean, default=False)
    severity        = Column(String(20), nullable=True)  # LOW / MEDIUM / HIGH / CRITICAL
    window_size     = Column(Integer, nullable=True)
    total_events    = Column(Integer, nullable=True)
    adaptive_threshold = Column(Float, nullable=True)
    created_at      = Column(DateTime, default=utcnow)


# ══════════════════════════════════════════════════════════
# FEATURE 1: MODEL VERSIONS (extends existing Model table)
# ══════════════════════════════════════════════════════════

class ModelVersion(Base):
    """Versioned model artifacts with governance scores."""
    __tablename__ = "model_versions"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    version_number  = Column(Integer, nullable=False, default=1)
    artifact_url    = Column(String(1024), nullable=True)
    framework       = Column(String(100), nullable=True)  # sklearn, pytorch, tensorflow, onnx
    parameters_count = Column(BigInteger, nullable=True)
    training_dataset = Column(String(1024), nullable=True)
    governance_score = Column(Float, nullable=True)
    risk_class      = Column(String(20), nullable=True)  # LOW / MEDIUM / HIGH / CRITICAL
    description     = Column(Text, nullable=True)
    metadata_json   = Column(PortableJSON, nullable=True)
    created_by      = Column(UUID(), ForeignKey("users.id"), nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    model           = relationship("Model", backref="versions")

    __table_args__ = (
        Index("ix_model_versions_model_version", "model_id", "version_number", unique=True),
    )


# ══════════════════════════════════════════════════════════
# FEATURE 8: ENVIRONMENT MANAGEMENT
# ══════════════════════════════════════════════════════════

class Environment(Base):
    """Deployment environments: DEV, STAGING, PRODUCTION."""
    __tablename__ = "environments"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    org_id          = Column(UUID(), ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=True)
    name            = Column(String(50), nullable=False)  # DEV, STAGING, PRODUCTION
    description     = Column(Text, nullable=True)
    is_active       = Column(Boolean, default=True)
    config          = Column(PortableJSON, default=dict)
    created_at      = Column(DateTime, default=utcnow)


class Deployment(Base):
    """Track model version deployments to environments."""
    __tablename__ = "deployments"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    version_id      = Column(UUID(), ForeignKey("model_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    environment_id  = Column(UUID(), ForeignKey("environments.id"), index=True, nullable=True)
    environment     = Column(String(50), nullable=False)  # DEV / STAGING / PRODUCTION
    status          = Column(String(50), default="ACTIVE")  # ACTIVE, ROLLED_BACK, ARCHIVED
    deployed_by     = Column(UUID(), ForeignKey("users.id"), nullable=True)
    deployment_date = Column(DateTime, default=utcnow)
    metadata_json   = Column(PortableJSON, nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    model_version   = relationship("ModelVersion", backref="deployments")


# ══════════════════════════════════════════════════════════
# FEATURE 2: DATASET LINEAGE
# ══════════════════════════════════════════════════════════

class DatasetVersion(Base):
    """Versioned dataset snapshots with schema tracking."""
    __tablename__ = "dataset_versions"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    dataset_id      = Column(UUID(), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False)
    version_number  = Column(Integer, nullable=False, default=1)
    storage_url     = Column(String(1024), nullable=True)
    schema_hash     = Column(String(128), nullable=True)
    row_count       = Column(Integer, nullable=True)
    feature_count   = Column(Integer, nullable=True)
    quality_score   = Column(Float, nullable=True)
    metadata_json   = Column(PortableJSON, nullable=True)
    created_by      = Column(UUID(), ForeignKey("users.id"), nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    dataset         = relationship("Dataset", backref="versions")


class LineageLink(Base):
    """Links dataset versions to model versions for governance compliance."""
    __tablename__ = "lineage_links"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    dataset_version_id = Column(UUID(), ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    model_version_id   = Column(UUID(), ForeignKey("model_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    training_run_id    = Column(UUID(), ForeignKey("experiments.id"), nullable=True)
    link_type       = Column(String(50), default="training")  # training, validation, testing
    created_at      = Column(DateTime, default=utcnow)

    dataset_version = relationship("DatasetVersion", backref="lineage_links")
    model_version   = relationship("ModelVersion", backref="lineage_links")


# ══════════════════════════════════════════════════════════
# FEATURE 3: EXPERIMENT TRACKING
# ══════════════════════════════════════════════════════════

class Experiment(Base):
    """Track ML training runs, hyperparameters, and metrics."""
    __tablename__ = "experiments"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    dataset_version_id = Column(UUID(), ForeignKey("dataset_versions.id"), nullable=True)
    name            = Column(String(255), nullable=True)
    parameters      = Column(PortableJSON, nullable=True)  # hyperparameters
    metrics         = Column(PortableJSON, nullable=True)   # accuracy, loss, f1, etc.
    artifact_url    = Column(String(1024), nullable=True)
    status          = Column(String(50), default="RUNNING")  # RUNNING, COMPLETED, FAILED
    training_time_ms = Column(Integer, nullable=True)
    framework       = Column(String(100), nullable=True)
    tags            = Column(PortableJSON, nullable=True)
    created_by      = Column(UUID(), ForeignKey("users.id"), nullable=True)
    started_at      = Column(DateTime, default=utcnow)
    completed_at    = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    model           = relationship("Model", backref="experiments")


# ══════════════════════════════════════════════════════════
# FEATURE 7: PREDICTION LOGGING (monitoring extension)
# ══════════════════════════════════════════════════════════

class PredictionLog(Base):
    """Log individual predictions for drift and performance monitoring."""
    __tablename__ = "prediction_logs"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(UUID(), ForeignKey("model_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    features        = Column(PortableJSON, nullable=True)
    prediction      = Column(PortableJSON, nullable=True)
    actual          = Column(PortableJSON, nullable=True)  # ground truth, if available
    confidence      = Column(Float, nullable=True)
    latency_ms      = Column(Integer, nullable=True)
    created_at      = Column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_prediction_logs_version_created", "model_version_id", "created_at"),
    )


# ══════════════════════════════════════════════════════════
# FEATURE 4: EXPLAINABILITY RESULTS
# ══════════════════════════════════════════════════════════

class ExplainabilityResult(Base):
    """Store SHAP/LIME explainability results per scan."""
    __tablename__ = "explainability_results"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    scan_id         = Column(UUID(), ForeignKey("scan_records.id", ondelete="CASCADE"), index=True, nullable=True)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    method          = Column(String(50), nullable=False)  # shap, lime, feature_importance
    global_importance = Column(PortableJSON, nullable=True)  # {feature: importance, ...}
    local_explanations = Column(PortableJSON, nullable=True)  # sample-level explanations
    summary_metrics = Column(PortableJSON, nullable=True)  # interpretability score, etc.
    created_at      = Column(DateTime, default=utcnow)

