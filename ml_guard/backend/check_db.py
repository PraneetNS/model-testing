import asyncio
import os
import sys

# Add current directory to path so we can import app
sys.path.insert(0, os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import PolicyVersion, PolicyRule, APIKey
from sqlalchemy.future import select

async def check_policies():
    async with AsyncSessionLocal() as db:
        res_ak = await db.execute(select(APIKey))
        api_keys = res_ak.scalars().all()
        print(f"APIKeys: {len(api_keys)}")
        for ak in api_keys:
            print(f"  - {ak.label} ({ak.id}), org={ak.org_id}")

        res_v = await db.execute(select(PolicyVersion))
        versions = res_v.scalars().all()
        print(f"PolicyVersions: {len(versions)}")
        for v in versions:
            print(f"  - {v.name} (v{v.version}), active={v.is_active}, org={v.org_id}")

        res_r = await db.execute(select(PolicyRule))
        rules = res_r.scalars().all()
        print(f"PolicyRules: {len(rules)}")
        for r in rules:
            print(f"  - {r.name}, active={r.is_active}, org={r.org_id}")

if __name__ == "__main__":
    asyncio.run(check_policies())
