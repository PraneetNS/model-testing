import asyncio
import os
import sys
import uuid

# Add current directory to path so we can import app
sys.path.insert(0, os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import PolicyVersion, Organization
from sqlalchemy.future import select

async def seed_policy():
    async with AsyncSessionLocal() as db:
        # Get org
        res = await db.execute(select(Organization).limit(1))
        org = res.scalars().first()
        if not org:
            print("No organization found!")
            return

        # Check if policy exists
        res_p = await db.execute(select(PolicyVersion).filter(PolicyVersion.is_active == True))
        if res_p.scalars().first():
            print("Active policy already exists.")
            return

        print(f"Seeding policy for org {org.id}")
        policy = PolicyVersion(
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
        )
        db.add(policy)
        await db.commit()
        print("Policy seeded successfully.")

if __name__ == "__main__":
    asyncio.run(seed_policy())
