import asyncio
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.db.models import Dataset

async def check():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Dataset))
        ds = res.scalars().all()
        print(f"Found {len(ds)} datasets")
        for d in ds:
            name = (d.metadata_json or {}).get("name", "Unnamed")
            print(f"ID: {d.id} | Name: {name} | Type: {d.type}")

if __name__ == "__main__":
    asyncio.run(check())
