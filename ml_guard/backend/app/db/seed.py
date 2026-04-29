"""
app/db/seed.py — Idempotent database seed.

Runs on every startup but only inserts data if the DB is empty.
Seeds:
  - Admin Organization
  - Admin User
  - Admin API Key (prints raw key to stdout once)
  - Default Governance Policy
  - Subscription Plans (free / pro / enterprise)
  - Demo Model record
"""
import secrets
import uuid
import hashlib
import asyncio
from datetime import datetime, timezone

import bcrypt
import structlog
from sqlalchemy import text
from sqlalchemy.future import select

from app.db.session import AsyncSessionLocal
from app.db.models import (
    APIKey, Organization, User, Model, Project,
    PolicyVersion, SubscriptionPlan,
)

logger = structlog.get_logger(__name__)


async def seed_if_empty():
    """Idempotent seed: checks each category independently."""
    async with AsyncSessionLocal() as db:
        logger.info("seed_check_starting")

        # ── 1. Subscription Plans ────────────────────────────────────────────
        res_plans = await db.execute(select(SubscriptionPlan).limit(1))
        plans = res_plans.scalars().all()
        if not plans:
            logger.info("seeding_plans")
            plans = [
                SubscriptionPlan(
                    id=uuid.uuid4(), name="Free", slug="free",
                    price_monthly_usd=0, predictions_per_month=1_000,
                    models_limit=2, reports_per_month=1,
                    compliance_packs_limit=0, guardrail_eval_limit=0,
                ),
                SubscriptionPlan(
                    id=uuid.uuid4(), name="Pro", slug="pro",
                    price_monthly_usd=29900, predictions_per_month=100_000,
                    models_limit=10, reports_per_month=10,
                    compliance_packs_limit=5, guardrail_eval_limit=1_000,
                ),
                SubscriptionPlan(
                    id=uuid.uuid4(), name="Enterprise", slug="enterprise",
                    price_monthly_usd=0, predictions_per_month=999_999_999,
                    models_limit=-1, reports_per_month=-1,
                    compliance_packs_limit=-1, guardrail_eval_limit=-1,
                    is_custom_price=True,
                ),
            ]
            for p in plans:
                db.add(p)
            await db.flush()
        
        enterprise_plan = next((p for p in plans if p.slug == "enterprise"), plans[0])

        # ── 2. Admin Organization ────────────────────────────────────────────
        res_org = await db.execute(select(Organization).limit(1))
        org = res_org.scalars().first()
        if not org:
            logger.info("seeding_org")
            org = Organization(
                id=uuid.uuid4(),
                name="Niyantrana Admin",
                slug="niyantrana-admin",
                plan="enterprise",
                plan_id=enterprise_plan.id,
                subscription_status="active",
            )
            db.add(org)
            await db.flush()

        # ── 3. Admin User ────────────────────────────────────────────────────
        res_user = await db.execute(select(User).limit(1))
        admin_user = res_user.scalars().first()
        if not admin_user:
            logger.info("seeding_user")
            hashed_pw = bcrypt.hashpw(b"change-me-immediately", bcrypt.gensalt()).decode("utf-8")
            admin_user = User(
                id=uuid.uuid4(),
                org_id=org.id,
                email="admin@niyantrana.ai",
                name="Platform Admin",
                role="admin",
                password_hash=hashed_pw,
                is_active=True,
            )
            db.add(admin_user)
            await db.flush()

        # ── 4. Admin API Key ─────────────────────────────────────────────────
        res_key = await db.execute(select(APIKey).limit(1))
        if not res_key.scalars().first():
            logger.info("seeding_api_key")
            raw_key = f"mlg_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            api_key = APIKey(
                id=uuid.uuid4(),
                org_id=org.id,
                label="Admin Master Key",
                key_hash=key_hash,
                is_active=True,
                scopes=["admin", "ml_engineer", "auditor", "viewer"],
                rate_limit_rpm=600,
            )
            db.add(api_key)
            await db.flush()
            print(f"  🔑 Admin API key:  {raw_key}")
        
        # ── 5. Default Governance Policy ─────────────────────────────────────
        res_pol = await db.execute(select(PolicyVersion).limit(1))
        if not res_pol.scalars().first():
            logger.info("seeding_policy")
            default_policy = PolicyVersion(
                id=uuid.uuid4(),
                org_id=org.id,
                name="Default Governance Policy",
                version=1,
                config={
                    "min_accuracy": 0.75,
                    "min_f1": 0.70,
                    "max_psi": 0.25,
                    "max_ks": 0.30,
                    "min_governance_score": 70.0,
                    "min_fairness_score": 0.80,
                    "max_drift_penalty": 0.30,
                    "max_brier_score": 0.25,
                    "max_missing_pct": 0.15,
                },
                is_active=True,
                notes="Seeded default policy.",
            )
            db.add(default_policy)
            await db.flush()

        # ── 6. Demo Project & Model ──────────────────────────────────────────
        res_proj = await db.execute(select(Project).limit(1))
        if not res_proj.scalars().first():
            logger.info("seeding_demo_data")
            project = Project(
                id=uuid.uuid4(),
                org_id=org.id,
                name="Demo Project",
                description="Starter project seeded automatically",
                created_by=admin_user.id,
            )
            db.add(project)
            await db.flush()

            demo_model = Model(
                id=uuid.uuid4(),
                project_id=project.id,
                name="Demo Credit Scoring Model",
                provider="Local",
                version=1,
                risk_tier="high",
                risk_tier_justification="Financial decisions affecting creditworthiness",
                use_case_category="credit_scoring",
                business_owner="admin@niyantrana.ai",
                technical_owner="admin@niyantrana.ai",
                deployment_environment="staging",
                model_type="classification",
                training_data_sensitivity="confidential",
                regulatory_jurisdictions=["US", "EU"],
                validation_frequency_days=90,
                created_by=admin_user.id,
            )
            db.add(demo_model)

        await db.commit()
        logger.info("seed_finished")
        return
