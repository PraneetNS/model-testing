import asyncio
import sys, os
sys.path.insert(0, os.getcwd())

from app.db.session import SessionLocal
from app.db.models import APIKey, Organization
from sqlalchemy.future import select

async def get_keys():
    async with SessionLocal() as db:
        res = await db.execute(select(APIKey).where(APIKey.is_active == True))
        keys = res.scalars().all()
        print(f"Active API keys: {len(keys)}")
        for k in keys:
            print(f"  ID: {k.id}")
            print(f"  Label: {k.label}")
            print(f"  Scopes: {k.scopes}")
            print(f"  Key hash (first 16): {k.key_hash[:16]}...")
            print()

if __name__ == "__main__":
    asyncio.run(get_keys())
