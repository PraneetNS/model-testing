import asyncio
import os
import sys
import uuid

# Add current directory to path so we can import app
sys.path.insert(0, os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import PolicyRule, Organization
from sqlalchemy.future import select

async def create_good_policy():
    async with AsyncSessionLocal() as db:
        # Get org
        res = await db.execute(select(Organization).limit(1))
        org = res.scalars().first()
        if not org:
            print("No organization found!")
            return

        print(f"Creating Enterprise Financial Policy for org {org.id}")
        
        # Deactivate existing rules for this org
        from sqlalchemy import update
        await db.execute(update(PolicyRule).where(PolicyRule.org_id == org.id).values(is_active=False))
        
        policy = PolicyRule(
            id=uuid.uuid4(),
            org_id=org.id,
            name="Enterprise Financial Governance v2.1",
            rules_json={
                "min_accuracy": 0.88,
                "min_f1_score": 0.85,
                "max_drift_psi": 0.12,
                "max_ks_statistic": 0.15,
                "min_governance_score": 85.0,
                "min_fairness_demographic_parity": 0.92,
                "max_brier_score": 0.18,
                "max_missing_values_pct": 0.02,
                "max_latency_p95_ms": 150,
                "require_explainability": True,
                "require_model_card": True
            },
            is_active=True,
        )
        db.add(policy)
        await db.commit()
        print("Good policy created and activated successfully.")

if __name__ == "__main__":
    asyncio.run(create_good_policy())
