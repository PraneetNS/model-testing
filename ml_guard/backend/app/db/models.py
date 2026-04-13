import uuid
import hashlib
import secrets
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Integer, Boolean, Text, Index, BigInteger, LargeBinary
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship, backref
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
    expires_at  = Column(DateTime, nullable=True)
    rate_limit_rpm = Column(Integer, default=120)
    request_count = Column(BigInteger, default=0)
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
    parent_model_id = Column(UUID(), ForeignKey("models.id"), nullable=True)  # for version chains
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
    
    # DAG Relationship
    children = relationship("Model", backref=backref("parent", remote_side=[id]))

class ModelExplanation(Base):
    __tablename__ = "model_explanations"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    computed_at = Column(DateTime, default=utcnow)
    feature_importances = Column(PortableJSON, nullable=False)
    top_drift_contributors = Column(PortableJSON, nullable=False)
    
    model       = relationship("Model", backref=backref("explanations", cascade="all, delete-orphan"))


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
    submission_token = Column(String(255), unique=True, index=True, nullable=True)
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
    actor_key_id = Column(UUID(), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    actor_ip    = Column(String(45), nullable=True)
    action      = Column(String(100), nullable=False, index=True)  # model.upload, scan.run, policy.update, etc.
    resource_type = Column(String(50), nullable=True)  # model, policy, project, etc.
    resource_id = Column(String(64), nullable=True)
    payload_hash = Column(String(64), nullable=True)
    result      = Column(String(20), nullable=True)
    details     = Column(PortableJSON, nullable=True)
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
# NEW: REAL-TIME DRIFT SENTINEL (v7.2)
# ══════════════════════════════════════════════════════════

class SentinelRecord(Base):
    """Real-time drift persistence for sliding-window Sentinel scans."""
    __tablename__ = "sentinel_records"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    avg_psi         = Column(Float, nullable=False)
    feature_psi     = Column(PortableJSON, nullable=True)  # {feature: psi, ...}
    window_size     = Column(Integer, nullable=True)
    threshold       = Column(Float, nullable=True)
    is_breached     = Column(Boolean, default=False)
    metadata_json   = Column(PortableJSON, nullable=True)
    created_at      = Column(DateTime, default=utcnow, index=True)

    model           = relationship("Model", backref="sentinel_scans")


# ══════════════════════════════════════════════════════════
# NEW: LLM RED TEAMING (v7.2)
# ══════════════════════════════════════════════════════════

class RedTeamSession(Base):
    """Container for a red-teaming campaign against an LLM."""
    __tablename__ = "red_team_sessions"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id        = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    status          = Column(String, default="RUNNING") # RUNNING, COMPLETED, FAILED
    total_attacks   = Column(Integer, default=0)
    success_count   = Column(Integer, default=0)
    metadata_json   = Column(PortableJSON, nullable=True)
    created_at      = Column(DateTime, default=utcnow)
    completed_at    = Column(DateTime, nullable=True)

    model           = relationship("Model", backref="red_team_sessions")
    attacks         = relationship("RedTeamAttack", backref="session", cascade="all, delete-orphan")

class RedTeamAttack(Base):
    """Detailed log of a single adversarial attempt (possibly multi-round)."""
    __tablename__ = "red_team_attacks"
    id              = Column(UUID(), primary_key=True, default=uuid.uuid4)
    session_id      = Column(UUID(), ForeignKey("red_team_sessions.id", ondelete="CASCADE"), index=True)
    category        = Column(String, nullable=False) # jailbreak, pii, bias, etc.
    severity        = Column(String, default="MEDIUM") # CRITICAL, HIGH, MEDIUM
    rounds          = Column(Integer, default=1)
    is_successful   = Column(Boolean, default=False)
    
    # Encrypted fields (Ferrent)
    encrypted_prompt    = Column(LargeBinary, nullable=False)
    encrypted_response  = Column(LargeBinary, nullable=True)
    judge_reasoning     = Column(String, nullable=True)
    
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


# FEATURE 7: PREDICTION LOGGING (observability foundation)
# ══════════════════════════════════════════════════════════

class PredictionLog(Base):
    """
    DEFINITIVE merged prediction log.
    Serves both versioned prediction tracking (Feature 7)
    and production observability ingestion (v7.2 layer).
    This is the single source of truth — no duplicates.
    """
    __tablename__ = "prediction_logs"

    __table_args__ = (
        Index("ix_predlog_model_ts",   "model_id",    "timestamp"),
        Index("ix_predlog_env",        "environment"),
        Index("ix_predlog_version_ts", "model_version_id", "timestamp"),
    )

    id                = Column(UUID(), primary_key=True, default=uuid.uuid4)

    # Observability fields (v7.2) — flexible string model_id for SDK ingestion
    model_id          = Column(String(255), nullable=False, index=True)

    # Versioned tracking FK (Feature 7) — optional, set when version is known
    model_version_id  = Column(UUID(), ForeignKey("model_versions.id",
                          ondelete="SET NULL"), nullable=True, index=True)

    timestamp         = Column(DateTime, default=utcnow, nullable=False, index=True)
    features          = Column(PortableJSON, nullable=True)       # raw input feature dict
    prediction        = Column(String(255), nullable=True)        # predicted value (str)
    prediction_proba  = Column(Float, nullable=True)              # confidence / proba
    actual            = Column(String(255), nullable=True)        # ground truth (Feature 7 name)
    ground_truth      = Column(String(255), nullable=True)        # alias filled via label endpoint
    confidence        = Column(Float, nullable=True)              # additional confidence field
    latency_ms        = Column(Float, nullable=True)
    data_source       = Column(String(50), default="api")         # api | batch | sdk
    environment       = Column(String(50), default="production")  # production | staging
    tags              = Column(PortableJSON, nullable=True)        # optional metadata dict

    def __repr__(self) -> str:
        return f"<PredictionLog(id={self.id}, model_id={self.model_id}, ts={self.timestamp})>"


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

class ReportCard(Base):
    """
    Certified Governance Report Card for an ML Model.
    Acts as a professional certificate of compliance.
    """
    __tablename__ = "report_cards"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Audit Identification
    cert_hash = Column(String(64), unique=True, nullable=False, index=True)
    issued_at = Column(DateTime, default=utcnow)
    
    # High Level Results
    overall_score = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False) # CERTIFIED, CONDITIONAL, FAILED
    
    # Summary & Content
    executive_summary = Column(Text, nullable=True)
    metric_snapshots = Column(PortableJSON, nullable=False) # Snapshot of audit data used
    
    # Status & Revocation
    is_revoked = Column(Boolean, default=False)
    revocation_reason = Column(String, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    
    # Storage Reference
    pdf_path = Column(String, nullable=True) # MinIO path: reports/{model_id}/{cert_hash}.pdf

    model = relationship("Model", backref="report_cards")

    def __repr__(self):
        return f"<ReportCard(id={self.id}, model_id={self.model_id}, hash={self.cert_hash})>"


# ══════════════════════════════════════════════════════════
# OBSERVABILITY LAYER — DRIFT + PERFORMANCE MODELS (v7.2)
# ══════════════════════════════════════════════════════════

class DriftReport(Base):
    """
    Per-feature drift analysis result. Generated hourly by Celery beat.
    Captures KS, PSI, chi2, and Wasserstein metrics per feature.
    """
    __tablename__ = "drift_reports"

    __table_args__ = (
        Index("ix_driftreport_model_ts", "model_id", "created_at"),
    )

    id                       = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id                 = Column(String(255), nullable=False, index=True)
    created_at               = Column(DateTime, default=utcnow, nullable=False, index=True)
    reference_window_start   = Column(DateTime, nullable=True)
    reference_window_end     = Column(DateTime, nullable=True)
    current_window_start     = Column(DateTime, nullable=True)
    current_window_end       = Column(DateTime, nullable=True)
    feature_results          = Column(PortableJSON, nullable=False)   # list of per-feature result dicts
    overall_drift_score      = Column(Float, nullable=False, default=0.0)
    drift_detected           = Column(Boolean, default=False)
    method                   = Column(String(50), default="ks")       # psi | ks | chi2 | wasserstein
    sample_count             = Column(Integer, nullable=True)
    alert_triggered          = Column(Boolean, default=False)

    def __repr__(self):
        return f"<DriftReport(id={self.id}, model_id={self.model_id}, drift={self.drift_detected})>"


class PerformanceSnapshot(Base):
    """
    Live model performance metrics computed from labeled PredictionLogs.
    Captures classification or regression metrics and computes degradation delta.
    """
    __tablename__ = "performance_snapshots"

    __table_args__ = (
        Index("ix_perfsnapshot_model_ts", "model_id", "computed_at"),
    )

    id                   = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id             = Column(String(255), nullable=False, index=True)
    computed_at          = Column(DateTime, default=utcnow, nullable=False, index=True)
    window_start         = Column(DateTime, nullable=True)
    window_end           = Column(DateTime, nullable=True)
    task_type            = Column(String(50), default="classification")  # classification | regression | ranking
    metrics              = Column(PortableJSON, nullable=False)           # computed metrics dict
    baseline_metrics     = Column(PortableJSON, nullable=True)            # baseline for delta computation
    degradation_report   = Column(PortableJSON, nullable=True)            # per-metric delta/alert
    sample_count         = Column(Integer, nullable=True)
    labeled_count        = Column(Integer, nullable=True)
    label_coverage_pct   = Column(Float, nullable=True)

    def __repr__(self):
        return f"<PerformanceSnapshot(id={self.id}, model_id={self.model_id}, computed_at={self.computed_at})>"


# ══════════════════════════════════════════════════════════
# MODEL BEHAVIOR CONTRACT SYSTEM
# ══════════════════════════════════════════════════════════

class ModelContract(Base):
    """
    A behavioral contract for an ML model.
    Defines promises the model must keep on every prediction.
    Breaches trigger alerts and governance score penalties.

    Promise types: output | latency | distribution | feature_range | fairness
    Operators:     lte | gte | lt | gt | eq | neq
    """
    __tablename__ = "model_contracts"

    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(String(255), nullable=False, index=True)
    name        = Column(String(255), nullable=False)
    version     = Column(String(50), default="1.0")
    description = Column(Text, nullable=True)
    promises    = Column(PortableJSON, nullable=False)  # list[PromiseDict]
    is_active   = Column(Boolean, default=True)
    breach_grace_period_minutes = Column(Integer, default=5)
    breach_window_minutes       = Column(Integer, default=60)
    created_at  = Column(DateTime, default=utcnow)
    created_by  = Column(UUID(), ForeignKey("users.id"), nullable=True)

    __table_args__ = (
        Index("ix_contract_model_active", "model_id", "is_active"),
    )

    breaches = relationship(
        "ContractBreach",
        back_populates="contract",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<ModelContract(id={self.id}, model_id={self.model_id!r}, "
            f"name={self.name!r}, active={self.is_active})>"
        )


class ContractBreach(Base):
    """
    Records every time a model promise is violated.
    Linked to the specific prediction that caused the breach.
    """
    __tablename__ = "contract_breaches"

    id                = Column(UUID(), primary_key=True, default=uuid.uuid4)
    contract_id       = Column(
        UUID(),
        ForeignKey("model_contracts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id          = Column(String(255), nullable=False, index=True)
    promise_name      = Column(String(255), nullable=False)
    promise_type      = Column(String(50), nullable=False)
    expected          = Column(String(255), nullable=True)
    actual            = Column(String(255), nullable=True)
    prediction_log_id = Column(UUID(), nullable=True)
    severity          = Column(String(20), default="HIGH")  # LOW/MEDIUM/HIGH/CRITICAL
    resolved          = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_breach_model_ts", "model_id", "created_at"),
    )

    contract = relationship("ModelContract", back_populates="breaches")

    def __repr__(self) -> str:
        return (
            f"<ContractBreach(id={self.id}, model_id={self.model_id!r}, "
            f"promise={self.promise_name!r}, severity={self.severity})>"
        )


class EmbeddingBatch(Base):
    """Stores batches of embeddings for model drift detection."""
    __tablename__ = "embedding_batches"
    
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
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

class ReportCard(Base):
    """
    Certified Governance Report Card for an ML Model.
    Acts as a professional certificate of compliance.
    """
    __tablename__ = "report_cards"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Audit Identification
    cert_hash = Column(String(64), unique=True, nullable=False, index=True)
    issued_at = Column(DateTime, default=utcnow)
    
    # High Level Results
    overall_score = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False) # CERTIFIED, CONDITIONAL, FAILED
    
    # Summary & Content
    executive_summary = Column(Text, nullable=True)
    metric_snapshots = Column(PortableJSON, nullable=False) # Snapshot of audit data used
    
    # Status & Revocation
    is_revoked = Column(Boolean, default=False)
    revocation_reason = Column(String, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    
    # Storage Reference
    pdf_path = Column(String, nullable=True) # MinIO path: reports/{model_id}/{cert_hash}.pdf

    model = relationship("Model", backref="report_cards")

    def __repr__(self):
        return f"<ReportCard(id={self.id}, model_id={self.model_id}, hash={self.cert_hash})>"


# ══════════════════════════════════════════════════════════
# OBSERVABILITY LAYER — DRIFT + PERFORMANCE MODELS (v7.2)
# ══════════════════════════════════════════════════════════

class DriftReport(Base):
    """
    Per-feature drift analysis result. Generated hourly by Celery beat.
    Captures KS, PSI, chi2, and Wasserstein metrics per feature.
    """
    __tablename__ = "drift_reports"

    __table_args__ = (
        Index("ix_driftreport_model_ts", "model_id", "created_at"),
    )

    id                       = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id                 = Column(String(255), nullable=False, index=True)
    created_at               = Column(DateTime, default=utcnow, nullable=False, index=True)
    reference_window_start   = Column(DateTime, nullable=True)
    reference_window_end     = Column(DateTime, nullable=True)
    current_window_start     = Column(DateTime, nullable=True)
    current_window_end       = Column(DateTime, nullable=True)
    feature_results          = Column(PortableJSON, nullable=False)   # list of per-feature result dicts
    overall_drift_score      = Column(Float, nullable=False, default=0.0)
    drift_detected           = Column(Boolean, default=False)
    method                   = Column(String(50), default="ks")       # psi | ks | chi2 | wasserstein
    sample_count             = Column(Integer, nullable=True)
    alert_triggered          = Column(Boolean, default=False)

    def __repr__(self):
        return f"<DriftReport(id={self.id}, model_id={self.model_id}, drift={self.drift_detected})>"


class PerformanceSnapshot(Base):
    """
    Live model performance metrics computed from labeled PredictionLogs.
    Captures classification or regression metrics and computes degradation delta.
    """
    __tablename__ = "performance_snapshots"

    __table_args__ = (
        Index("ix_perfsnapshot_model_ts", "model_id", "computed_at"),
    )

    id                   = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id             = Column(String(255), nullable=False, index=True)
    computed_at          = Column(DateTime, default=utcnow, nullable=False, index=True)
    window_start         = Column(DateTime, nullable=True)
    window_end           = Column(DateTime, nullable=True)
    task_type            = Column(String(50), default="classification")  # classification | regression | ranking
    metrics              = Column(PortableJSON, nullable=False)           # computed metrics dict
    baseline_metrics     = Column(PortableJSON, nullable=True)            # baseline for delta computation
    degradation_report   = Column(PortableJSON, nullable=True)            # per-metric delta/alert
    sample_count         = Column(Integer, nullable=True)
    labeled_count        = Column(Integer, nullable=True)
    label_coverage_pct   = Column(Float, nullable=True)

    def __repr__(self):
        return f"<PerformanceSnapshot(id={self.id}, model_id={self.model_id}, computed_at={self.computed_at})>"


# ══════════════════════════════════════════════════════════
# MODEL BEHAVIOR CONTRACT SYSTEM
# ══════════════════════════════════════════════════════════

class ModelContract(Base):
    """
    A behavioral contract for an ML model.
    Defines promises the model must keep on every prediction.
    Breaches trigger alerts and governance score penalties.

    Promise types: output | latency | distribution | feature_range | fairness
    Operators:     lte | gte | lt | gt | eq | neq
    """
    __tablename__ = "model_contracts"

    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id    = Column(String(255), nullable=False, index=True)
    name        = Column(String(255), nullable=False)
    version     = Column(String(50), default="1.0")
    description = Column(Text, nullable=True)
    promises    = Column(PortableJSON, nullable=False)  # list[PromiseDict]
    is_active   = Column(Boolean, default=True)
    breach_grace_period_minutes = Column(Integer, default=5)
    breach_window_minutes       = Column(Integer, default=60)
    created_at  = Column(DateTime, default=utcnow)
    created_by  = Column(UUID(), ForeignKey("users.id"), nullable=True)

    __table_args__ = (
        Index("ix_contract_model_active", "model_id", "is_active"),
    )

    breaches = relationship(
        "ContractBreach",
        back_populates="contract",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<ModelContract(id={self.id}, model_id={self.model_id!r}, "
            f"name={self.name!r}, active={self.is_active})>"
        )


class ContractBreach(Base):
    """
    Records every time a model promise is violated.
    Linked to the specific prediction that caused the breach.
    """
    __tablename__ = "contract_breaches"

    id                = Column(UUID(), primary_key=True, default=uuid.uuid4)
    contract_id       = Column(
        UUID(),
        ForeignKey("model_contracts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id          = Column(String(255), nullable=False, index=True)
    promise_name      = Column(String(255), nullable=False)
    promise_type      = Column(String(50), nullable=False)
    expected          = Column(String(255), nullable=True)
    actual            = Column(String(255), nullable=True)
    prediction_log_id = Column(UUID(), nullable=True)
    severity          = Column(String(20), default="HIGH")  # LOW/MEDIUM/HIGH/CRITICAL
    resolved          = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_breach_model_ts", "model_id", "created_at"),
    )

    contract = relationship("ModelContract", back_populates="breaches")

    def __repr__(self) -> str:
        return (
            f"<ContractBreach(id={self.id}, model_id={self.model_id!r}, "
            f"promise={self.promise_name!r}, severity={self.severity})>"
        )


class EmbeddingBatch(Base):
    """Stores batches of embeddings for model drift detection."""
    __tablename__ = "embedding_batches"
    
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    batch_id = Column(String(255), nullable=False, index=True)
    embeddings = Column(PortableJSON, nullable=False)
    timestamp = Column(DateTime, default=utcnow, index=True)

    def __repr__(self):
        return f"<EmbeddingBatch(id={self.id}, model_id={self.model_id}, batch_id={self.batch_id})>"

class RagTrace(Base):
    """Stores RAG evaluation traces."""
    __tablename__ = "rag_traces"
    
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_chunks = Column(PortableJSON, nullable=False) # list of str
    retrieved_doc_ids = Column(PortableJSON, nullable=False) # list of str
    latency_ms = Column(Float, nullable=True)
    
    # Precomputed metrics to save time
    context_relevance = Column(Float, nullable=True)
    grounding_fidelity = Column(Float, nullable=True)
    hallucination_risk = Column(String(50), nullable=True)
    
    timestamp = Column(DateTime, default=utcnow, index=True)

    def __repr__(self):
        return f"<RagTrace(id={self.id}, model_id={self.model_id}, risk={self.hallucination_risk})>"

class NotificationConfig(Base):
    """Configuration for outbound notifications (Slack/Teams)."""
    __tablename__ = "notification_configs"
    
    id                  = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id            = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False, unique=True)
    slack_webhook_url   = Column(String(512), nullable=True)
    slack_channel       = Column(String(100), nullable=True)
    teams_webhook_url   = Column(String(512), nullable=True)
    # notify_on: ["CRITICAL", "HIGH", "PREDICTIVE_BREACH", "SCORE_DECAY"]
    notify_on           = Column(PortableJSON, default=list) 
    created_at          = Column(DateTime, default=utcnow)
    
    def __repr__(self):
        return f"<NotificationConfig(model_id={self.model_id})>"

class SecurityAlert(Base):
    """Storage for detected injection attempts and security anomalies."""
    __tablename__ = "alerts"
    id          = Column(UUID(), primary_key=True, default=uuid.uuid4)
    timestamp   = Column(DateTime, default=utcnow, index=True)
    alert_type  = Column(String(100), nullable=False) # injection_attempt
    endpoint    = Column(String(255), nullable=True)
    payload_hash = Column(String(64), nullable=True)
    ip          = Column(String(45), nullable=True)
    key_id      = Column(UUID(), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    details     = Column(PortableJSON, nullable=True)


class AIBOM(Base):
    """AI Bill of Materials (AIBOM) — a cryptographically verifiable manifest of everything a model depends on."""
    __tablename__ = "aibom"
    id               = Column(UUID(), primary_key=True, default=uuid.uuid4)
    model_id         = Column(UUID(), ForeignKey("models.id", ondelete="CASCADE"), index=True, nullable=False)
    generated_at     = Column(DateTime, default=utcnow)
    schema_version   = Column(String(10), default="1.0")
    base_model       = Column(PortableJSON, nullable=False)
    training_datasets = Column(PortableJSON, nullable=False)
    dependencies     = Column(PortableJSON, nullable=False)
    training_framework = Column(PortableJSON, nullable=False)
    aibom_hash       = Column(String(64), nullable=False, index=True)

    model = relationship("Model", backref=backref("aiboms", cascade="all, delete-orphan", order_by=generated_at.desc()))

    def __repr__(self):
        return f"<AIBOM(model_id={self.model_id}, hash={self.aibom_hash})>"

