import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import SubscriptionPlan
from sqlalchemy import update, select

async def run():
    async with AsyncSessionLocal() as db:
        # Check if plans exist
        plans = (await db.execute(select(SubscriptionPlan))).scalars().all()
        if not plans:
            print("No plans found in DB. Seeding might be needed.")
            return
            
        print(f"Found {len(plans)} plans.")
        for p in plans:
            print(f"Plan: {p.slug}, Limit: {p.compliance_packs_limit}")
            
        # Update free plan
        await db.execute(
            update(SubscriptionPlan)
            .where(SubscriptionPlan.slug == 'free')
            .values(compliance_packs_limit=100)
        )
        await db.commit()
        print("Free plan compliance limit bumped to 100.")

if __name__ == "__main__":
    asyncio.run(run())
