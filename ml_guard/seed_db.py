
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

from app.db.session import SessionLocal, engine, Base
from app.db.models import (
    Organization, User, Project, Model, ModelVersion, 
    Deployment, Dataset, DatasetVersion, Experiment, PredictionLog,
    ScanRecord, AuditLog, PolicyVersion, AlertRule, AlertEvent, Environment,
    CIIntegration
)

def seed():
    # Force fresh schema for development seeding
    print("🧹 Cleaning database...")
    try:
        Base.metadata.drop_all(bind=engine)
    except:
        pass
    print("🏗️ Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # 1. Create Org and User
        org = Organization(name="Fireflink Enterprise", slug="fireflink", plan="enterprise")
        db.add(org)
        db.commit()
        db.refresh(org)

        user = User(org_id=org.id, email="admin@fireflink.com", name="System Admin", role="admin")
        db.add(user)
        db.commit()
        db.refresh(user)

        project = Project(org_id=org.id, name="Financial Security", created_by=user.id)
        db.add(project)
        db.commit()
        db.refresh(project)

        # 2. Environments
        envs = []
        for e_name in ["DEV", "STAGING", "PRODUCTION"]:
            env = Environment(org_id=org.id, name=e_name, description=f"{e_name} environment")
            db.add(env)
            envs.append(env)
        db.commit()

        # 3. Models
        model_configs = [
            ("CreditRiskPredictor", "XGBoost"),
            ("FraudDetectorV7", "PyTorch"),
            ("ChurnForecaster", "RandomForest")
        ]

        for m_name, framework in model_configs:
            m = Model(project_id=project.id, name=m_name, provider=framework, created_by=user.id)
            db.add(m)
            db.commit()
            db.refresh(m)

            # Versions
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
                db.commit()
                db.refresh(mv)

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
                            model_version_id=mv.id,
                            prediction={"class": random.randint(0, 1)},
                            confidence=random.uniform(0.6, 0.99),
                            latency_ms=random.randint(10, 100),
                            created_at=datetime.now() - timedelta(minutes=i*15)
                        )
                        db.add(log)

            # Datasets
            ds = Dataset(
                model_id=m.id,
                type="training",
                metadata_json={"name": f"{m_name}_Dataset", "source": "S3"},
                row_count=100000
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)

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
        db.commit()
        db.refresh(rule)

        alert = AlertEvent(rule_id=rule.id, severity="CRITICAL", message="High drift detected in CreditRiskPredictor v3")
        db.add(alert)

        # 6. Audit Logs
        actions = ["user.login", "model.upload", "scan.run", "policy.update", "deployment.promote"]
        for i in range(20):
            audit = AuditLog(
                org_id=org.id,
                user_id=user.id,
                action=random.choice(actions),
                details={"ip": "192.168.1.1", "browser": "Chrome"}
            )
            db.add(audit)

        # 7. CI/CD Integrations
        ci = CIIntegration(
            org_id=org.id,
            provider="github",
            repo_url="https://github.com/fireflink/credit-risk-model",
            is_active=True,
            settings={"gate_policy": "standard"}
        )
        db.add(ci)

        db.commit()
        print("✅ Success: ML Guard Enterprise Database seeded/updated with full lifecycle data.")

    except Exception as e:
        db.rollback()
        print(f"❌ Error during seeding: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
