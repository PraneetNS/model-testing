import asyncio
import os
import sys
import uuid
import random
from datetime import datetime, timedelta

# Add backend to path
base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if os.path.exists(backend_dir):
    sys.path.append(backend_dir)
else:
    sys.path.append(os.getcwd())

from sqlalchemy import select
from app.db.session import SessionLocal, engine, Base
from app.db.models import (
    Organization, User, Project, Model, ModelVersion, 
    Deployment, Dataset, DatasetVersion, Experiment, PredictionLog,
    ScanRecord, AuditLog, PolicyVersion, AlertRule, AlertEvent, Environment,
    CIIntegration, RetrainingPolicy, ModelContract, ContractBreach, AIBOM, SecurityAlert
)


async def seed():
    # Force fresh schema for development seeding
    print("Cleaning database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
    
    async with SessionLocal() as db:
        try:
            # 1. Create Org and User
            org = Organization(name="Fireflink Enterprise", slug="fireflink", plan="enterprise")
            db.add(org)
            await db.commit()
            await db.refresh(org)

            user = User(org_id=org.id, email="admin@fireflink.com", name="System Admin", role="admin")
            db.add(user)
            await db.commit()
            await db.refresh(user)

            project = Project(org_id=org.id, name="Financial Security", created_by=user.id)
            db.add(project)
            await db.commit()
            await db.refresh(project)

            # 2. Environments
            envs = []
            for e_name in ["DEV", "STAGING", "PRODUCTION"]:
                env = Environment(org_id=org.id, name=e_name, description=f"{e_name} environment")
                db.add(env)
                envs.append(env)
            await db.commit()

            # 3. Models
            model_configs = [
                ("CreditRiskPredictor", "XGBoost"),
                ("FraudDetectorV7", "PyTorch"),
                ("ChurnForecaster", "RandomForest")
            ]

            for m_name, framework in model_configs:
                m = Model(project_id=project.id, name=m_name, provider=framework, created_by=user.id)
                db.add(m)
                await db.commit()
                await db.refresh(m)

                # Versions
                pred_logs = []
                for v_num in range(1, 4):
                    score = random.uniform(60, 95)
                    mv = ModelVersion(
                        model_id=m.id,
                        version_number=v_num,
                        framework=framework.lower(),
                        governance_score=score,
                        risk_class="CRITICAL" if score < 60 else "HIGH" if score < 75 else "MEDIUM" if score < 85 else "LOW",
                        created_by=user.id
                    )
                    db.add(mv)
                    await db.commit()
                    await db.refresh(mv)

                    # Scans
                    scan = ScanRecord(
                        model_id=m.id,
                        model_version_id=mv.id,
                        scan_type="audit",
                        checks_run=["accuracy", "drift", "fairness", "security"],
                        results_json={"metrics": {"accuracy": random.uniform(0.7, 0.99)}},
                        governance_score=score,
                        risk_level=mv.risk_class,
                        gate_status="PASSED" if score > 70 else "WARNING",
                        security_checks=[
                            {"test_name": "Membership Inference", "status": "PASS", "score": 0.98, "risk_level": "LOW"},
                            {"test_name": "Model Inversion", "status": "PASS", "score": 0.95, "risk_level": "LOW"},
                            {"test_name": "Evasion Attack Resistance", "status": "FAIL", "score": 0.42, "risk_level": "HIGH"},
                        ]
                    )
                    db.add(scan)

                    # Deployments for latest versions
                    if v_num == 3:
                        for env in envs:
                            deploy = Deployment(
                                version_id=mv.id,
                                environment=env.name,
                                status="ACTIVE",
                                deployed_by=user.id
                            )
                            db.add(deploy)

                    # Prediction Logs (for the latest version)
                    if v_num == 3:
                        for i in range(50):
                            log = PredictionLog(
                                model_id=m.name,
                                model_version_id=mv.id,
                                prediction=str(random.randint(0, 1)),
                                confidence=random.uniform(0.6, 0.99),
                                latency_ms=random.randint(10, 100),
                                features={"gender": random.choice(["male", "female"]), "age": random.randint(18, 70)},
                                timestamp=datetime.now() - timedelta(minutes=i*15)
                            )
                            db.add(log)
                            pred_logs.append(log)
                await db.commit()

                # Seed Contracts, Breaches, AIBOM, and Security Alerts
                contract_latency = ModelContract(
                    id=uuid.uuid4(),
                    model_id=str(m.id),
                    name="Latency SLA Contract",
                    version="1.0",
                    description="Ensures production API response times remain within SLA thresholds.",
                    is_active=True,
                    breach_grace_period_minutes=5,
                    breach_window_minutes=60,
                    promises=[
                        {
                            "name": "Max Latency SLA",
                            "type": "latency",
                            "metric": "latency_ms",
                            "operator": "lte",
                            "threshold": 50.0,
                            "severity": "HIGH",
                            "action": "alert",
                            "window_hours": 24
                        }
                    ]
                )
                db.add(contract_latency)
                
                contract_fairness = ModelContract(
                    id=uuid.uuid4(),
                    model_id=str(m.id),
                    name="Demographic Parity Contract",
                    version="1.0",
                    description="Monitors output bias and parity difference across demographic attributes.",
                    is_active=True,
                    breach_grace_period_minutes=15,
                    breach_window_minutes=120,
                    promises=[
                        {
                            "name": "Demographic Parity Delta Limit",
                            "type": "fairness",
                            "metric": "prediction",
                            "operator": "lte",
                            "threshold": 0.1,
                            "severity": "CRITICAL",
                            "action": "alert",
                            "window_hours": 24,
                            "protected_attribute": "gender"
                        }
                    ]
                )
                db.add(contract_fairness)
                await db.commit()
                await db.refresh(contract_latency)
                await db.refresh(contract_fairness)

                # Seed historical breaches
                for i, log in enumerate(pred_logs[:5]):
                    breach = ContractBreach(
                        id=uuid.uuid4(),
                        contract_id=contract_latency.id,
                        model_id=str(m.id),
                        promise_name="Max Latency SLA",
                        promise_type="latency",
                        expected="50.0",
                        actual=str(log.latency_ms),
                        prediction_log_id=log.id,
                        severity="HIGH",
                        resolved=(i % 2 == 0),
                        created_at=log.timestamp
                    )
                    db.add(breach)
                
                # Seed AIBOM manifest
                aibom = AIBOM(
                    id=uuid.uuid4(),
                    model_id=m.id,
                    schema_version="1.0",
                    base_model={
                        "name": f"{m_name}_Base",
                        "repo_id": f"fireflink/{m_name.lower()}-base",
                        "sha256": "8a7c6f5d4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b"
                    },
                    training_datasets=[
                        {
                            "name": f"{m_name}_Train_Set",
                            "version": "1.0.0",
                            "fingerprint": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
                        }
                    ],
                    dependencies=[
                        {"name": "numpy", "version": "1.24.3", "hash": "sha256:7b6a..."},
                        {"name": "scikit-learn", "version": "1.2.2", "hash": "sha256:3d2c..."},
                        {"name": "transformers", "version": "4.30.2", "hash": "sha256:4f3a..."}
                    ],
                    training_framework={
                        "name": framework.lower(),
                        "version": "2.1.0"
                    },
                    aibom_hash="f5c6b7a890123456789abcdef0123456789abcdef0123456789abcdef0123456"
                )
                db.add(aibom)

                # Seed SecurityAlerts supply_chain_cve alerts
                alert1 = SecurityAlert(
                    id=uuid.uuid4(),
                    model_id=m.id,
                    alert_type="supply_chain_cve",
                    severity="HIGH",
                    endpoint="N/A",
                    details={
                        "package": "transformers",
                        "cve_id": "CVE-2023-38320",
                        "description": "Arbitrary code execution via unsafe deserialization in HuggingFace Transformers."
                    }
                )
                db.add(alert1)

                alert2 = SecurityAlert(
                    id=uuid.uuid4(),
                    model_id=m.id,
                    alert_type="supply_chain_cve",
                    severity="MEDIUM",
                    endpoint="N/A",
                    details={
                        "package": "numpy",
                        "cve_id": "CVE-2023-37290",
                        "description": "Buffer overflow in numpy.array conversion."
                    }
                )
                db.add(alert2)
                await db.commit()


                # Datasets
                ds = Dataset(
                    model_id=m.id,
                    type="training",
                    metadata_json={"name": f"{m_name}_Dataset", "source": "S3"},
                    row_count=100000
                )
                db.add(ds)
                await db.commit()
                await db.refresh(ds)

                dv = DatasetVersion(
                    dataset_id=ds.id,
                    version_number=1,
                    row_count=100000,
                    quality_score=random.uniform(80, 100),
                    created_by=user.id
                )
                db.add(dv)

                # Experiments
                for i in range(5):
                    exp = Experiment(
                        model_id=m.id,
                        name=f"sweep_iteration_{i}",
                        parameters={"lr": 0.01, "layers": [64, 32]},
                        metrics={"loss": 0.1, "accuracy": 0.9},
                        status="COMPLETED",
                        started_at=datetime.now() - timedelta(days=i),
                        completed_at=datetime.now() - timedelta(days=i, hours=-1)
                    )
                    db.add(exp)

            # 4. Policies
            policy = PolicyVersion(
                org_id=org.id,
                name="Standard Governance Policy",
                config={"min_accuracy": 0.8, "max_drift": 0.2, "security_threshold": "MEDIUM"},
                is_active=True
            )
            db.add(policy)

            # 5. Alert Rules & Events
            rule = AlertRule(
                org_id=org.id,
                name="Critical Drift Alert",
                condition={"metric": "drift", "op": ">", "value": 0.25},
                channels=["slack", "email"]
            )
            db.add(rule)
            await db.commit()
            await db.refresh(rule)

            alert = AlertEvent(rule_id=rule.id, severity="CRITICAL", message="High drift detected in CreditRiskPredictor v3")
            db.add(alert)

            # 6. Audit Logs
            actions = ["user.login", "model.upload", "scan.run", "policy.update", "deployment.promote"]
            for i in range(20):
                audit_log = AuditLog(
                    org_id=org.id,
                    action=random.choice(actions),
                    details={"ip": "192.168.1.1", "browser": "Chrome"}
                )
                db.add(audit_log)

            # 8. Retraining Policy
            m_first = (await db.execute(select(Model).limit(1))).scalars().first()
            if m_first:
                rp = RetrainingPolicy(
                    model_id=str(m_first.id),
                    enabled=True,
                    trigger_conditions={
                        "psi_threshold": 0.2,
                        "ks_stat_threshold": 0.1,
                        "performance_degradation_pct": 10.0,
                        "min_days_since_last_retrain": 3,
                        "require_all_conditions": False
                    },
                    retrain_action={
                        "action_type": "webhook",
                        "webhook_url": "https://hooks.slack.com/services/sample-trigger"
                    }
                )
                db.add(rp)

            await db.commit()
            print("Success: ML Guard Enterprise Database seeded/updated with full lifecycle data.")

        except Exception as e:
            await db.rollback()
            import traceback
            traceback.print_exc()
            print(f"Error during seeding: {e}")
        finally:
            await db.close()

if __name__ == "__main__":
    asyncio.run(seed())
