import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import APIKey
from sqlalchemy.future import select

async def check_keys():
    async with AsyncSessionLocal() as db:
        stmt = select(APIKey).filter(APIKey.is_active == True)
        result = await db.execute(stmt)
        keys = result.scalars().all()
        if not keys:
            print("No active keys found.")
        for k in keys:
            print(f"Label: {k.label}, KeyHash: {k.key_hash}")

if __name__ == "__main__":
    asyncio.run(check_keys())
