import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
from app.infrastructure.database import Base

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, index=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    users = relationship("User", back_populates="tenant")
    projects = relationship("Project", back_populates="tenant")

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="developer") # admin, auditor, developer
    is_active = Column(Boolean(), default=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    
    tenant = relationship("Tenant", back_populates="users")

class Project(Base):
    __tablename__ = "projects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    tenant = relationship("Tenant", back_populates="projects")
    runs = relationship("TestRun", back_populates="project")

class TestRun(Base):
    __tablename__ = "test_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"))
    model_version = Column(String)
    suite_name = Column(String)
    score = Column(Float)
    deployment_allowed = Column(Boolean)
    summary_metrics = Column(JSON) # Aggregated metrics
    results_raw = Column(JSON) # Detailed list of TestResult objects
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="runs")
    drift_logs = relationship("DriftLog", back_populates="test_run")

class DriftLog(Base):
    __tablename__ = "drift_logs"
    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(UUID(as_uuid=True), ForeignKey("test_runs.id"))
    feature_name = Column(String)
    psi_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    test_run = relationship("TestRun", back_populates="drift_logs")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String) # e.g. "RUN_TEST", "CREATE_PROJECT"
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
